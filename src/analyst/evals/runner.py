"""Run the eval suite across one or more models and report.

    python -m analyst.evals.runner
    python -m analyst.evals.runner --models qwen2.5-coder:14b,qwen2.5-coder:1.5b
    python -m analyst.evals.runner --repeat 3        # measure run-to-run variance

Local models are not deterministic even at temperature 0, so a single run is a
sample, not a score. `--repeat` reports the spread.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from analyst.evals.cases import SUITE, EvalCase
from analyst.evals.grade import Result, grade
from analyst.llm import OllamaProvider, ProviderError
from analyst.prepare import prepare_frame
from analyst.profiler import load_and_profile
from analyst.session import AnalystSession

DIM, BOLD, RED, GREEN, YELLOW, RESET = (
    ("\033[2m", "\033[1m", "\033[31m", "\033[32m", "\033[33m", "\033[0m")
    if sys.stdout.isatty()
    else ("", "", "", "", "", "")
)


@dataclass
class RunRecord:
    case_id: str
    model: str
    attempt: int
    results: list[Result] = field(default_factory=list)
    narrative: str = ""
    route: str = ""
    seconds: float = 0.0
    output_tokens: int = 0
    error: str | None = None

    @property
    def passed(self) -> int:
        return sum(r.passed for r in self.results)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def score(self) -> float:
        return self.passed / self.total if self.total else 0.0


def run_case(session: AnalystSession, case: EvalCase, attempt: int) -> RunRecord:
    record = RunRecord(case_id=case.id, model=session.provider.info.model, attempt=attempt)
    started = time.monotonic()
    try:
        answer = session.ask(case.question)
    except (ProviderError, ValueError, RuntimeError) as exc:
        record.error = f"{type(exc).__name__}: {exc}"
        record.seconds = time.monotonic() - started
        return record

    record.results = grade(answer, case.assertions)
    record.narrative = answer.narrative
    record.route = answer.route
    record.seconds = time.monotonic() - started
    record.output_tokens = answer.output_tokens
    return record


def run_suite(models: list[str], repeat: int, only: list[str] | None) -> list[RunRecord]:
    cases = [c for c in SUITE if not only or c.id in only]
    if not cases:
        raise SystemExit(f"no cases matched {only}")

    prepared_cache: dict[Path, object] = {}
    records: list[RunRecord] = []

    for model in models:
        provider = OllamaProvider(model)
        reachable, status = provider.health()
        if not reachable:
            print(f"{RED}skipping {model}: {status}{RESET}", file=sys.stderr)
            continue

        print(f"\n{BOLD}{model}{RESET}")
        sessions: dict[Path, AnalystSession] = {}
        try:
            for case in cases:
                if case.dataset not in prepared_cache:
                    raw, profile = load_and_profile(case.dataset)
                    prepared_cache[case.dataset] = prepare_frame(raw, profile)
                if case.dataset not in sessions:
                    # One session (and one sandbox kernel) per dataset, reused
                    # across cases so the worker start-up is not paid per case.
                    sessions[case.dataset] = AnalystSession(
                        provider,
                        prepared_cache[case.dataset],
                        workdir=Path(__file__).resolve().parents[3] / "outputs",
                    )
                session = sessions[case.dataset]

                for attempt in range(1, repeat + 1):
                    record = run_case(session, case, attempt)
                    records.append(record)

                    suffix = f" (run {attempt}/{repeat})" if repeat > 1 else ""
                    if record.error:
                        print(f"  {RED}ERROR{RESET} {case.id}{suffix}: {record.error}")
                    else:
                        colour = (
                            GREEN if record.score == 1 else YELLOW if record.score >= 0.7 else RED
                        )
                        print(
                            f"  {colour}{record.passed:>2}/{record.total}{RESET} "
                            f"{case.id}{suffix} {DIM}[{record.route}] "
                            f"{record.seconds:.0f}s{RESET}"
                        )
                        for failure in (r for r in record.results if not r.passed):
                            print(f"       {RED}x{RESET} {failure.name} {DIM}— {failure.detail}{RESET}")
        finally:
            for session in sessions.values():
                session.close()

    return records


def report(records: list[RunRecord], models: list[str]) -> str:
    """Markdown summary — paste-able straight into the README."""
    by_model: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        by_model[record.model].append(record)

    lines = ["## Eval results", ""]
    lines.append(
        "| model | score | routing | interpretation | correctness | communication "
        "| median s | cost |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")

    for model in models:
        runs = by_model.get(model)
        if not runs:
            continue
        graded = [r for r in runs if not r.error]
        if not graded:
            lines.append(f"| `{model}` | all runs errored | | | | | | free |")
            continue

        passed = sum(r.passed for r in graded)
        total = sum(r.total for r in graded)
        family: dict[str, list[bool]] = defaultdict(list)
        for run in graded:
            for result in run.results:
                family[result.family].append(result.passed)

        def pct(values: list[bool]) -> str:
            return f"{100 * sum(values) / len(values):.0f}%" if values else "—"

        lines.append(
            f"| `{model}` | **{passed}/{total}** ({100 * passed / total:.0f}%) "
            f"| {pct(family['routing'])} "
            f"| {pct(family['interpretation'])} "
            f"| {pct(family['correctness'])} "
            f"| {pct(family['communication'])} "
            f"| {statistics.median(r.seconds for r in graded):.0f} | free |"
        )

    lines += ["", "### Per case", "", "| case | " + " | ".join(f"`{m}`" for m in models) + " |"]
    lines.append("|---" * (len(models) + 1) + "|")
    for case in SUITE:
        row = [case.id]
        for model in models:
            runs = [r for r in by_model.get(model, []) if r.case_id == case.id and not r.error]
            if not runs:
                row.append("—")
            elif len(runs) == 1:
                row.append(f"{runs[0].passed}/{runs[0].total}")
            else:
                scores = [r.passed for r in runs]
                row.append(f"{min(scores)}–{max(scores)}/{runs[0].total}")
        lines.append("| " + " | ".join(row) + " |")

    # --- variance -----------------------------------------------------------
    repeats = max(
        (len([r for r in by_model[m] if r.case_id == c.id]) for m in models for c in SUITE),
        default=1,
    )
    if repeats > 1:
        lines += ["", "### Run-to-run variance", ""]
        lines.append("| model | mean score | sd | worst run | best run | unstable cases |")
        lines.append("|---|---|---|---|---|---|")
        for model in models:
            graded = [r for r in by_model.get(model, []) if not r.error]
            if not graded:
                continue
            # One "run" is a full pass over the suite: group by attempt index.
            by_attempt: dict[int, list[RunRecord]] = defaultdict(list)
            for record in graded:
                by_attempt[record.attempt].append(record)
            totals = [
                100 * sum(r.passed for r in rs) / sum(r.total for r in rs)
                for rs in by_attempt.values()
                if sum(r.total for r in rs)
            ]
            unstable = [
                c.id
                for c in SUITE
                if len({r.passed for r in graded if r.case_id == c.id}) > 1
            ]
            sd = statistics.stdev(totals) if len(totals) > 1 else 0.0
            lines.append(
                f"| `{model}` | {statistics.mean(totals):.1f}% | {sd:.1f} pp "
                f"| {min(totals):.1f}% | {max(totals):.1f}% "
                f"| {', '.join(f'`{u}`' for u in unstable) if unstable else 'none'} |"
            )
        lines += [
            "",
            f"Each model ran the full suite {repeats} times. A case listed as unstable "
            "scored differently across runs — local models are not deterministic even "
            "at temperature 0, so a single pass is a sample, not a score.",
        ]

    failures: dict[str, int] = defaultdict(int)
    for record in records:
        for result in record.results:
            if not result.passed:
                failures[f"{result.family} / {result.name}"] += 1
    if failures:
        lines += ["", "### Most common failures", ""]
        for name, count in sorted(failures.items(), key=lambda kv: -kv[1])[:8]:
            lines.append(f"- `{name}` — failed {count}x")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="analyst.evals.runner", description=__doc__)
    parser.add_argument("--models", default="qwen2.5-coder:14b")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--only", help="comma-separated case ids")
    parser.add_argument("--out", type=Path, default=Path("evals_report.md"))
    parser.add_argument(
        "--json", type=Path, help="also write raw per-run records here, for re-analysis"
    )
    args = parser.parse_args(argv)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    only = [c.strip() for c in args.only.split(",")] if args.only else None

    started = time.monotonic()
    records = run_suite(models, args.repeat, only)
    if not records:
        print(f"{RED}no runs completed{RESET}", file=sys.stderr)
        return 1

    markdown = report(records, models)
    args.out.write_text(markdown + "\n")

    if args.json:
        args.json.write_text(
            json.dumps(
                [
                    {
                        "case": r.case_id,
                        "model": r.model,
                        "attempt": r.attempt,
                        "route": r.route,
                        "passed": r.passed,
                        "total": r.total,
                        "seconds": round(r.seconds, 2),
                        "output_tokens": r.output_tokens,
                        "error": r.error,
                        "failures": [
                            {"family": x.family, "name": x.name, "detail": x.detail}
                            for x in r.results
                            if not x.passed
                        ],
                        "narrative": r.narrative,
                    }
                    for r in records
                ],
                indent=2,
            )
        )

    print(f"\n{markdown}\n")
    print(f"{DIM}{len(records)} runs in {time.monotonic() - started:.0f}s -> {args.out}{RESET}")

    graded = [r for r in records if not r.error]
    overall = sum(r.passed for r in graded) / max(1, sum(r.total for r in graded))
    return 0 if overall >= 0.8 else 1


if __name__ == "__main__":
    raise SystemExit(main())
