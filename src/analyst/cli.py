"""Command-line entry point. Runs locally and free by default.

    python -m analyst.cli data/sales_2026.csv "Why did revenue drop in March?"
    python -m analyst.cli data/sales_2026.csv --profile-only     # no model needed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from analyst.llm import OllamaProvider, ProviderError
from analyst.llm.ollama import DEFAULT_MODEL
from analyst.prepare import prepare_frame
from analyst.profiler import load_and_profile
from analyst.session import AnalystSession, RoutedAnswer

DIM, BOLD, RED, GREEN, YELLOW, RESET = (
    ("\033[2m", "\033[1m", "\033[31m", "\033[32m", "\033[33m", "\033[0m")
    if sys.stdout.isatty()
    else ("", "", "", "", "", "")
)


def render(answer: RoutedAnswer, *, show_detail: bool) -> None:
    print(f"\n{BOLD}{answer.narrative}{RESET}\n")

    if answer.route == "code":
        _render_code(answer, show_detail=show_detail)
        return

    analysis = answer.change.analysis

    if show_detail:
        source = analysis.intersection.dimension if analysis.intersection else "dimension"
        print(f"{BOLD}Contributions by {source}{RESET}  {DIM}(percentage points of the total change){RESET}")
        breakdown = analysis.intersection or next(iter(analysis.breakdowns.values()))
        for s in breakdown.ranked(analysis.direction)[:6]:
            bar = "█" * max(1, int(abs(s.contribution_pp) * 2))
            flag = f"  {YELLOW}<- big % move, small share{RESET}" if s.is_decoy else ""
            print(f"  {s.segment:<28} {s.contribution_pp:>7.2f}  {bar}{flag}")

        print(f"\n{BOLD}Verification{RESET}")
        for check in analysis.checks:
            mark = f"{GREEN}PASS{RESET}" if check.passed else f"{RED}FAIL{RESET}"
            print(f"  {mark}  {check.name}")
            print(f"        {DIM}{check.detail}{RESET}")

        if analysis.notes:
            print(f"\n{BOLD}Normalisation applied{RESET}")
            for note in analysis.notes:
                print(f"  {DIM}- {note}{RESET}")

    for warning in answer.change.plan.warnings:
        print(f"{YELLOW}  ! {warning}{RESET}")

    _render_footer(answer, verified_label=f"{GREEN}verified{RESET}" if answer.verified
                   else f"{RED}CHECKS FAILED{RESET}")


def _render_code(answer: RoutedAnswer, *, show_detail: bool) -> None:
    run = answer.code
    if show_detail and run.attempts:
        print(f"{BOLD}Generated code{RESET}")
        for i, attempt in enumerate(run.attempts, 1):
            mark = f"{GREEN}ok{RESET}" if attempt.ok else f"{RED}failed{RESET}"
            print(f"\n{DIM}--- step {i}/{len(run.attempts)} [{mark}{DIM}] "
                  f"{attempt.result.duration_ms}ms{RESET}")
            for line in attempt.code.splitlines():
                print(f"{DIM}|{RESET} {line}")
            output = attempt.result.to_model_text()
            print(f"{DIM}{output[:600].rstrip()}{RESET}")

    if run.stop_reason == "no_execution":
        print(f"{RED}The model tried to answer without running code — refused.{RESET}")

    _render_footer(
        answer,
        verified_label=f"{YELLOW}unverified — figures come from generated code{RESET}",
    )


def _render_footer(answer: RoutedAnswer, *, verified_label: str) -> None:
    print(
        f"\n{DIM}route={answer.route}, {len(answer.completions)} local model calls, "
        f"{answer.output_tokens:,} output tokens, {answer.seconds:.1f}s, $0.00 — "
        f"{verified_label}{RESET}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="analyst", description=__doc__)
    parser.add_argument("data", type=Path, help="CSV/Excel/Parquet/JSON file to analyse")
    parser.add_argument("question", nargs="?", help="question to ask; omit for interactive mode")
    parser.add_argument("--sheet", help="Excel sheet name or index")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model")
    parser.add_argument("--profile-only", action="store_true", help="print the profile and exit")
    parser.add_argument("--keep-duplicates", action="store_true")
    parser.add_argument("--brief", action="store_true", help="answer only, no detail")
    parser.add_argument("--workdir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--no-code",
        action="store_true",
        help="verified change analysis only; refuse questions needing generated code",
    )
    args = parser.parse_args(argv)

    if not args.data.exists():
        parser.error(f"no such file: {args.data}")

    sheet: str | int | None = args.sheet
    if isinstance(sheet, str) and sheet.isdigit():
        sheet = int(sheet)

    raw, profile = load_and_profile(args.data, sheet=sheet)

    if args.profile_only:
        print(profile.to_prompt())
        return 0

    provider = OllamaProvider(args.model)
    reachable, status = provider.health()
    if not reachable:
        print(f"{RED}{status}{RESET}", file=sys.stderr)
        print("\nOr run with --profile-only, which needs no model.", file=sys.stderr)
        return 2

    data = prepare_frame(raw, profile, drop_duplicates=not args.keep_duplicates)
    print(f"{DIM}{data.rows_after:,} rows ready ({data.rows_removed:,} removed){RESET}")
    for action in data.actions:
        print(f"{DIM}  {'!' if not action.reversible else '+'} {action.description}{RESET}")

    questions = [args.question] if args.question else None

    with AnalystSession(
        provider, data, workdir=args.workdir, allow_codegen=not args.no_code
    ) as session:
        while True:
            if questions is None:
                try:
                    question = input(f"\n{BOLD}?{RESET} ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    return 0
                if not question:
                    return 0
            else:
                question = questions[0]

            try:
                render(session.ask(question), show_detail=not args.brief)
            except ProviderError as exc:
                print(f"{RED}{exc}{RESET}", file=sys.stderr)
                return 1
            except (ValueError, RuntimeError) as exc:
                print(f"{RED}Could not answer that: {exc}{RESET}", file=sys.stderr)
                if questions is not None:
                    return 1

            if questions is not None:
                return 0


if __name__ == "__main__":
    raise SystemExit(main())
