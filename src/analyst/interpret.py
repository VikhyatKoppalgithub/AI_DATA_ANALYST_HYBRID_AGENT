"""Question interpretation and narration — the two jobs a model is actually good at.

Everything numeric happens in analyst.analysis. The model reads a question and
picks columns; later it reads finished numbers and writes prose. It never
computes anything, because it is measurably bad at that: asked to split a 16%
decline across two segments, a 14B model returned 13pp and 3pp — plausible,
summing correctly to 16, and wrong by 5x on the smaller one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from analyst.analysis import ChangeAnalysis, analyze_change
from analyst.llm.base import Completion, Provider, ProviderError
from analyst.profiler import DatasetProfile

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "metric": {"type": "string"},
        "date_column": {"type": "string"},
        "period_after": {"type": "string"},
        "dimensions": {"type": "array", "items": {"type": "string"}},
        "filters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"column": {"type": "string"}, "value": {"type": "string"}},
                "required": ["column", "value"],
            },
        },
    },
    "required": ["metric", "date_column", "period_after", "dimensions", "filters"],
}

PLAN_SYSTEM = """\
You map a business question onto columns of a dataset. You do not analyse \
anything and you do not compute numbers — you only choose columns and a period.

Rules:
- `metric` must be the quantity the question is about. Prefer a money or volume \
column over an identifier. Never choose an ID column.
- `date_column` must be the column holding dates, even if it is stored as text.
- `period_after` is the period the question asks about, as YYYY-MM. Infer the \
year from the data range you are shown if the question does not state one. \
If the question refers to the latest, most recent, or current period without \
naming one, return an empty string — the system knows which period is last and \
will fill it in exactly. Do not guess at which month that is.
- `dimensions` are the categorical columns worth breaking the change down by. \
Include every plausible one; exclude identifiers and free text.
- Use exact column names as given. Do not invent columns.

`filters` narrows the rows before analysis. Usually leave it empty. It matters \
for datasets in long format, where a single value column holds many different \
measures identified by another column — a `Value` column alongside a \
`Variable_name` column holding "Total income", "Total expenditure", and so on. \
Summing such a column across measures is meaningless, and mixing units is \
worse. In that shape:
- set `metric` to the value column,
- add a filter selecting the one measure the question is about,
- add a filter on any units column so only one unit is summed.

Filters also matter when a dataset stores several levels of aggregation in the \
same table. A column whose values look like "Level 1", "Level 2", "Total", \
"All industries", or "Subtotal" means rows are nested: the totals already \
contain the detail, so summing every row counts the same money several times. \
Add a filter pinning one level. Prefer the most detailed level when the \
question asks which segment drove a change, since the top level has only one \
row and explains nothing.

Use exact values as shown in the profile.
"""

NARRATE_SYSTEM = """\
You explain a completed analysis to a business reader.

Every number has already been computed and verified. Copy figures exactly as \
written. Do not calculate, re-derive, round differently, sum anything, or state \
any number that is not in the input. If a figure is not given, you do not have it.

Never add contributions together. The input shows the same decline sliced \
several different ways; those slices overlap, so adding them double-counts. \
Never write that several segments "each contributed" or "together account for" \
some amount.

Structure your answer:
1. One sentence naming the single largest contributor from THE ANSWER section, \
with its exact percentage-point figure and the total change.
2. One or two sentences of support drawn only from the supporting facts.
3. If the input flags a large move that is not the story, one sentence warning \
that this segment's percentage move explains very little of the total.

Write about what *accounts for* a change, never what caused it — this is \
observational data showing which segments moved, not why they moved. Say "X was \
the largest contributor, accounting for N points of the decline". Never say the \
metric fell "due to", "because of", "driven by", or "caused by" a segment.

That restriction is about the metric only. Explaining the arithmetic is fine and \
encouraged: a segment explains little of the total *because* it is a small share \
of the starting figure, and saying so is accurate.

The input is organised under headings in capital letters. They exist to tell \
you which figures to use and are not part of the answer. Never quote or refer \
to a heading in your reply; write as if you had worked the numbers out \
yourself. Naming the structure of your input is the single most common mistake \
here.

No headings, no bullets, no preamble. Four sentences at most.
"""


@dataclass
class QuestionPlan:
    metric: str
    date_column: str
    period_after: str | None
    dimensions: list[str]
    filters: list[tuple[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def describe_filters(self) -> str:
        return ", ".join(f"{c} = {v!r}" for c, v in self.filters)


@dataclass
class Answer:
    question: str
    narrative: str
    analysis: ChangeAnalysis
    plan: QuestionPlan
    completions: list[Completion] = field(default_factory=list)

    @property
    def verified(self) -> bool:
        return self.analysis.passed


def _numeric_candidates(profile: DatasetProfile) -> list[str]:
    return [
        c.name
        for c in profile.columns
        if c.semantic_type in ("numeric", "numeric_as_string")
    ]


def _date_candidates(profile: DatasetProfile) -> list[str]:
    """Datetimes first; an integer year is a valid time axis when there is no date."""
    dates = [c.name for c in profile.columns if c.semantic_type in ("date", "date_as_string")]
    return dates + [c.name for c in profile.columns if c.semantic_type == "year"]


def _dimension_candidates(profile: DatasetProfile) -> list[str]:
    return [c.name for c in profile.columns if c.semantic_type in ("categorical", "boolean")]


def resolve_question(
    provider: Provider,
    profile: DatasetProfile,
    question: str,
    *,
    date_range: str = "",
    frame: pd.DataFrame | None = None,
) -> tuple[QuestionPlan, Completion]:
    """Ask the model which columns the question refers to, then validate the answer."""
    lines = [f"Columns in `{profile.source}`:"]
    for column in profile.columns:
        detail = f"- {column.name} ({column.semantic_type}"
        if column.stats.get("top_values"):
            detail += f", values: {', '.join(list(column.stats['top_values'])[:5])}"
        lines.append(detail + ")")
    if date_range:
        lines.append(f"\nData covers: {date_range}")

    # The profiler already detects nested aggregation levels, mis-stored types,
    # and duplicate rows. Computing that and then not showing it to the planner
    # was why a hierarchical dataset kept getting summed across its own totals.
    if profile.warnings:
        lines.append("\nThings the profiler noticed about this data:")
        lines += [f"- {w}" for w in profile.warnings]

    lines.append(f"\nUser question: {question!r}")

    completion = provider.complete(
        system=PLAN_SYSTEM, prompt="\n".join(lines), schema=PLAN_SCHEMA, max_tokens=500
    )
    raw = completion.data or {}
    plan = _validate_plan(raw, profile, frame=frame)
    return plan, completion


def _validate_plan(
    raw: dict, profile: DatasetProfile, frame: pd.DataFrame | None = None
) -> QuestionPlan:
    """Never trust returned column names — a hallucinated one fails deep in pandas."""
    known = {c.name for c in profile.columns}
    warnings: list[str] = []

    metric = raw.get("metric", "")
    if metric not in known:
        fallback = _numeric_candidates(profile)
        if not fallback:
            raise ProviderError(f"No numeric column to analyse; model suggested {metric!r}.")
        warnings.append(f"Model chose unknown metric {metric!r}; using '{fallback[0]}'.")
        metric = fallback[0]

    date_column = raw.get("date_column", "")
    if date_column not in known:
        fallback = _date_candidates(profile)
        if not fallback:
            raise ProviderError("No date column detected, so change over time cannot be computed.")
        warnings.append(f"Model chose unknown date column {date_column!r}; using '{fallback[0]}'.")
        date_column = fallback[0]

    dimensions = [d for d in raw.get("dimensions", []) if d in known and d != metric]
    dropped = [d for d in raw.get("dimensions", []) if d not in known]
    if dropped:
        warnings.append(f"Ignored unknown dimension(s): {', '.join(dropped)}.")
    if not dimensions:
        dimensions = [d for d in _dimension_candidates(profile) if d != metric]
        warnings.append("Model named no usable dimensions; using all categorical columns.")

    period = raw.get("period_after") or None
    if period and not _looks_like_period(period, frame=frame):
        warnings.append(f"Ignored malformed period {period!r}; using the latest period.")
        period = None

    filters: list[tuple[str, str]] = []
    for entry in raw.get("filters") or []:
        column, value = entry.get("column", ""), entry.get("value", "")
        if column not in known:
            warnings.append(f"Ignored filter on unknown column {column!r}.")
            continue
        if frame is not None and not _value_exists(frame, column, value):
            # Filtering to zero rows produces an empty analysis rather than an
            # error, so a value that is not present must be dropped here.
            warnings.append(f"Ignored filter {column} = {value!r}: no such value in the data.")
            continue
        filters.append((column, value))

    return QuestionPlan(
        metric=metric,
        date_column=date_column,
        period_after=period,
        dimensions=[d for d in dimensions if d not in {c for c, _ in filters}],
        filters=filters,
        warnings=warnings,
    )


def _value_exists(frame: pd.DataFrame, column: str, value: str) -> bool:
    series = frame[column].astype("string").str.strip().str.casefold()
    return bool((series == str(value).strip().casefold()).any())


def apply_filters(frame: pd.DataFrame, filters: list[tuple[str, str]]) -> pd.DataFrame:
    """Narrow to the rows a long-format question is actually about."""
    for column, value in filters:
        series = frame[column].astype("string").str.strip().str.casefold()
        frame = frame[series == str(value).strip().casefold()]
    return frame


def _looks_like_period(value: str, frame: pd.DataFrame | None = None) -> bool:
    """`YYYY-MM` for dated data, or a bare `YYYY` when the axis is a year."""
    if value.isdigit() and len(value) == 4:
        return True
    parts = value.split("-")
    return len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit() and len(parts[0]) == 4


def _analysis_facts(analysis: ChangeAnalysis, filters: str = "") -> str:
    """Render the computed result for the narrator.

    Ordering and labelling matter as much as the numbers. A flat ranked list
    that mixes dimensions reads as additive, and a model handed one will write
    that three segments "each contributed" — double-counting the same money.
    So the single best answer leads, and the overlapping views are explicitly
    fenced off as alternative slicings rather than separate causes.
    """
    direction = "fell" if analysis.delta < 0 else "rose"
    headline = (
        f"{analysis.metric} {direction} from {analysis.total_before:,.2f} to "
        f"{analysis.total_after:,.2f} between {analysis.period_before} and "
        f"{analysis.period_after}: a change of {analysis.delta:,.2f} "
        f"({analysis.pct_change:+.1f}%)."
    )
    lines = [f"TOTAL CHANGE\n{headline}"]
    if filters:
        lines += [
            "",
            "SCOPE — the figures above cover only these rows. Say so in your answer:",
            f"  {filters}",
        ]

    headline = analysis.headline()
    source, best = headline if headline else (None, None)

    if best is not None:
        lines += [
            "",
            "THE ANSWER — the single largest contributor (report this one):",
            f"  {best.segment} ({source})",
            f"  contributed {best.contribution_pp:+.2f} percentage points of the "
            f"{analysis.pct_change:+.1f}% total change",
            f"  this segment's own change was {best.own_pct_change:+.1f}%, and it was "
            f"{best.share_of_total_before:.1f}% of the starting total",
        ]

    if analysis.intersection is not None:
        lines += ["", "SUPPORTING FACTS — the next largest, same slicing:"]
        for s in analysis.intersection.ranked(analysis.direction)[1:3]:
            lines.append(f"  {s.segment}: {s.contribution_pp:+.2f} pp (own change {s.own_pct_change:+.1f}%)")

    lines += [
        "",
        "ALTERNATIVE SLICINGS — do NOT add these together and do NOT list them as",
        "separate contributors. Each row below re-describes the SAME decline from a",
        "different angle, so the figures overlap with THE ANSWER above:",
    ]
    for dimension, s in analysis.top_contributors(3):
        lines.append(f"  by {dimension}: {s.segment} {s.contribution_pp:+.2f} pp")

    decoys = [
        s
        for s in (analysis.intersection.ranked(analysis.direction) if analysis.intersection else [])
        if s.is_decoy
    ]
    if decoys:
        s = decoys[0]
        lines += [
            "",
            "A LARGE MOVE THAT IS NOT THE STORY — warn about this one:",
            f"  {s.segment} moved {s.own_pct_change:+.1f}% on its own, the largest "
            f"percentage move in the data, but explains only {s.contribution_pp:+.2f} pp "
            f"of the {analysis.pct_change:+.1f}% total because it was just "
            f"{s.share_of_total_before:.1f}% of revenue.",
        ]

    failed = [c for c in analysis.checks if not c.passed]
    if failed:
        lines += ["", "VERIFICATION PROBLEMS — state these as caveats:"]
        lines += [f"  {c.name}: {c.detail}" for c in failed]

    return "\n".join(lines)


def answer_question(
    provider: Provider,
    frame: pd.DataFrame,
    profile: DatasetProfile,
    question: str,
) -> Answer:
    """Full free path: interpret -> compute exactly -> narrate."""
    dates = _date_candidates(profile)
    date_range = ""
    if dates:
        series = frame[dates[0]].dropna()
        if not series.empty:
            if pd.api.types.is_datetime64_any_dtype(series):
                date_range = f"{series.min():%Y-%m-%d} to {series.max():%Y-%m-%d}"
            else:
                date_range = f"{series.min()} to {series.max()} (annual; periods are YYYY)"

    plan, plan_completion = resolve_question(
        provider, profile, question, date_range=date_range, frame=frame
    )

    scoped = apply_filters(frame, plan.filters)
    if plan.filters and scoped.empty:
        raise ValueError(
            f"No rows left after filtering on {plan.describe_filters()}."
        )

    analysis = analyze_change(
        scoped,
        metric=plan.metric,
        date_column=plan.date_column,
        period_after=plan.period_after,
        dimensions=plan.dimensions,
    )

    narration = provider.complete(
        system=NARRATE_SYSTEM,
        prompt=(
            f"Question: {question}\n\nVerified results:\n"
            f"{_analysis_facts(analysis, plan.describe_filters())}"
        ),
        max_tokens=400,
        temperature=0.2,
    )

    return Answer(
        question=question,
        narrative=narration.text,
        analysis=analysis,
        plan=plan,
        completions=[plan_completion, narration],
    )
