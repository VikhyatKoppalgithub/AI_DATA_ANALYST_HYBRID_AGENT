"""Routes a question to the path that can answer it, and owns the shared state.

Two paths with different guarantees:

  change   The deterministic engine. Exact, reconciled, verifiable. Only covers
           "how did X change between periods, and what accounts for it".
  code     Generated Python in the sandbox. Answers anything, but its numbers
           come from code the model wrote, so they carry no reconciliation.

The difference is surfaced rather than smoothed over: `verified` is True only on
the first path. Presenting both as equally trustworthy would undo the point of
computing anything exactly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from analyst.codegen import CodegenRun, CodegenSession
from analyst.interpret import Answer, answer_question
from analyst.llm.base import Completion, Provider
from analyst.prepare import PreparedData
from analyst.sandbox import Kernel

ROUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "route": {"type": "string", "enum": ["change_over_time", "other"]},
        "reason": {"type": "string"},
    },
    "required": ["route", "reason"],
}

ROUTE_SYSTEM = """\
You classify a question about a dataset into one of two routes. You do not \
answer it.

`change_over_time` is a narrow, exact tool. It compares **one summable metric** \
between **exactly two periods** and works out which segments account for the \
difference. Choose it only when all three hold:
- the question is about a single metric, not several;
- it implies one comparison, not a series of them;
- the metric is a total that can be summed, not an average, median, or rate.

Examples that fit: "why did revenue drop in March", "what happened to sales \
last month", "which region drove the decline".

`other` is everything else, and it is the safer default. Choose it when the \
question asks for any of:
- more than one metric ("revenue and profit");
- more than one kind of comparison ("month-over-month and year-over-year");
- a trend, a series, or a history across many periods;
- unusual periods, outliers, anomalies, or spikes;
- an average, median, rate, share, correlation, distribution, or ranking;
- anything the narrow tool above cannot express.

A question that mentions a change is not automatically `change_over_time`. If \
answering it fully would need several separate comparisons, choose `other` — a \
partial answer to a broader question is worse than a complete one.
"""


# Statistics the change engine cannot express. It sums; an average, median, or
# correlation is not a sum, so answering one with a total is not an approximation
# but a different number — on one fixture a question about *average* resolution
# time yields -1.3% where the true change in the mean is +0.2%, the opposite sign.
# Prompting the router not to pick it was unreliable, so this is enforced in code.
NON_SUMMABLE = re.compile(
    r"\b(average|avg|mean|median|typical|correlation|distribution|"
    r"variance|std|standard deviation|percentile|quantile|per capita)\b",
    re.I,
)


def _bootstrap_for(path: Path, reader: str) -> str:
    """The snippet the kernel replays on every restart to bring `df` back."""
    return (
        "import pandas as pd, numpy as np\n"
        f"df = pd.{reader}(r'{path.resolve()}')\n"
        "len(df)"
    )


@dataclass
class RoutedAnswer:
    question: str
    route: str
    narrative: str
    verified: bool
    reason: str = ""
    change: Answer | None = None
    code: CodegenRun | None = None
    completions: list[Completion] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.completions is None:
            self.completions = []

    @property
    def charts(self) -> list[str]:
        return self.code.charts if self.code else []

    @property
    def seconds(self) -> float:
        return sum(c.duration_s for c in self.completions)

    @property
    def output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.completions)


def route_question(provider: Provider, question: str) -> tuple[str, str, Completion]:
    completion = provider.complete(
        system=ROUTE_SYSTEM,
        prompt=f"Question: {question!r}",
        schema=ROUTE_SCHEMA,
        max_tokens=150,
    )
    data = completion.data or {}
    route = data.get("route", "other")
    reason = data.get("reason", "")
    if route not in ("change_over_time", "other"):
        route = "other"

    if route == "change_over_time" and (match := NON_SUMMABLE.search(question)):
        route = "other"
        reason = (
            f"asks for {match.group(0).lower()!r}, which the summing engine "
            f"cannot express — routed to generated code instead"
        )

    return route, reason, completion


class AnalystSession:
    """A prepared dataset plus the two ways of asking about it."""

    def __init__(
        self,
        provider: Provider,
        data: PreparedData,
        *,
        workdir: str | Path = "outputs",
        allow_codegen: bool = True,
    ) -> None:
        self.provider = provider
        self.data = data
        self.workdir = Path(workdir)
        self.allow_codegen = allow_codegen
        self._kernel: Kernel | None = None
        self._codegen: CodegenSession | None = None

    # ------------------------------------------------------------- sandbox

    def _ensure_codegen(self) -> CodegenSession:
        """Start the sandbox on first use — most questions never need it."""
        if self._codegen is not None:
            return self._codegen

        self.workdir.mkdir(parents=True, exist_ok=True)
        snapshot, reader = self._write_snapshot()

        kernel = Kernel(self.workdir, timeout=90)
        boot = kernel.set_bootstrap(_bootstrap_for(snapshot, reader))

        # Writing parquet successfully does not mean reading it will succeed.
        # The write happens here and is checked; the read happens inside the
        # worker, and on a managed host an Arrow/pandas skew broke the read while
        # the write was fine — taking the entire code route down with it. Pickle
        # is permissive and equally faithful to dtypes, so retry through it
        # rather than lose the route.
        if not boot.ok and reader == "read_parquet":
            pickled = self.workdir / "_prepared.pkl"
            self.data.frame.to_pickle(pickled)
            boot = kernel.set_bootstrap(_bootstrap_for(pickled, "read_pickle"))

        if not boot.ok:
            kernel.shutdown()
            raise RuntimeError(f"sandbox failed to load the data:\n{boot.to_model_text()}")

        self._kernel = kernel
        self._codegen = CodegenSession(
            self.provider,
            kernel,
            self.data.profile,
            preparation_log=[a.description for a in self.data.actions],
        )
        return self._codegen

    def _write_snapshot(self) -> tuple[Path, str]:
        """Hand the sandbox the *prepared* frame, so generated code sees exactly
        what the deterministic engine sees.

        Parquet first: compact, and it preserves dtypes where a CSV round-trip
        would turn parsed dates back into strings. But Arrow requires one type
        per column, and real spreadsheets produce object columns holding a mix
        of str and int — a title row read as a header is enough to do it. Those
        raise ArrowTypeError, so pickle is the fallback: permissive about object
        columns and equally faithful to dtypes.
        """
        parquet = self.workdir / "_prepared.parquet"
        try:
            self.data.frame.to_parquet(parquet, index=False)
            return parquet, "read_parquet"
        except Exception:  # noqa: BLE001 — any serialisation failure falls back
            pickled = self.workdir / "_prepared.pkl"
            self.data.frame.to_pickle(pickled)
            return pickled, "read_pickle"

    # ----------------------------------------------------------------- ask

    def ask(self, question: str, *, force_route: str | None = None) -> RoutedAnswer:
        completions: list[Completion] = []

        if force_route:
            route, reason = force_route, "forced by caller"
        else:
            route, reason, routing = route_question(self.provider, question)
            completions.append(routing)

        if route == "change_over_time":
            try:
                answer = answer_question(
                    self.provider, self.data.frame, self.data.profile, question
                )
            except ValueError as exc:
                # The engine cannot express it after all (no date column, one
                # period only). Fall through rather than failing the question.
                if not self.allow_codegen:
                    raise
                reason = f"change analysis not applicable ({exc}); used generated code"
                route = "other"
            else:
                completions += answer.completions
                return RoutedAnswer(
                    question=question,
                    route="change",
                    narrative=answer.narrative,
                    verified=answer.verified,
                    reason=reason,
                    change=answer,
                    completions=completions,
                )

        if not self.allow_codegen:
            raise ValueError(
                "This question needs generated code, which is disabled for this session."
            )

        run = self._ensure_codegen().ask(question)
        completions += run.completions
        return RoutedAnswer(
            question=question,
            route="code",
            narrative=run.answer,
            verified=False,
            reason=reason,
            code=run,
            completions=completions,
        )

    def close(self) -> None:
        if self._kernel is not None:
            self._kernel.shutdown()
            self._kernel = None
            self._codegen = None

    def __enter__(self) -> AnalystSession:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
