"""Contribution analysis: what accounts for a change between two periods.

The arithmetic lives here rather than in a model. It is exact, reproducible,
free to run, and testable against a fixture with a known answer — none of which
is true of asking a language model to add things up.

Two ideas do the work:

  contribution_pp   how many percentage points of the *total* change a slice
                    explains. This is the ranking that answers "why".
  own_pct_change    how much the slice moved relative to itself. Attention-
                    grabbing and usually irrelevant: a slice that fell 60% but
                    is 1% of the business explains almost nothing.

Reporting only the second is the classic wrong answer, so both travel together.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from analyst.analysis.normalize import NormalizationReport, normalize_categorical

RECONCILE_TOLERANCE = 0.01  # currency units
DIFFUSE_THRESHOLD_PP = 0.35  # top slice explaining less than this share => diffuse
DECOY_MIN_OWN_CHANGE = 25.0  # percent, on the slice's own base
DECOY_MAX_SHARE = 0.10  # fraction of the total change the slice may explain


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class Slice:
    segment: str
    before: float
    after: float
    change: float
    contribution_pp: float
    own_pct_change: float
    share_of_total_before: float
    share_of_change: float = 0.0  # this slice's change / the total change

    @property
    def is_decoy(self) -> bool:
        """A dramatic percentage move that explains almost nothing.

        Measured as a share of the total change rather than in absolute
        percentage points. An absolute threshold does not survive a change of
        dataset: 1pp is a sixteenth of a -16% move but a fiftieth of a +48% one,
        so the same rule flags a real contributor in one file and misses a decoy
        in the next.
        """
        return (
            abs(self.own_pct_change) >= DECOY_MIN_OWN_CHANGE
            and abs(self.share_of_change) < DECOY_MAX_SHARE
        )


@dataclass
class Breakdown:
    dimension: str
    slices: list[Slice]
    residual: float
    normalization: NormalizationReport | None = None

    @property
    def reconciles(self) -> bool:
        return abs(self.residual) <= RECONCILE_TOLERANCE

    @property
    def concentration(self) -> float:
        """Absolute contribution of the single largest mover."""
        return max((abs(s.contribution_pp) for s in self.slices), default=0.0)

    def ranked(self, direction: int) -> list[Slice]:
        """Slices moving *with* the overall change, largest contribution first.

        `slices` is stored ascending by change, so its first element is the most
        negative one. That happens to be the top contributor when the metric
        falls and is exactly wrong when it rises — on a +48% move it returns the
        largest counter-movement. Always rank through here, never by index.
        """
        with_change = [s for s in self.slices if s.change * direction > 0]
        return sorted(with_change, key=lambda s: abs(s.contribution_pp), reverse=True)

    def leading(self, direction: int) -> Slice | None:
        ranked = self.ranked(direction)
        return ranked[0] if ranked else None


@dataclass
class ChangeAnalysis:
    metric: str
    date_column: str
    period_before: str
    period_after: str
    total_before: float
    total_after: float
    delta: float
    pct_change: float
    breakdowns: dict[str, Breakdown] = field(default_factory=dict)
    intersection: Breakdown | None = None
    checks: list[Check] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def direction(self) -> int:
        """+1 when the metric rose, -1 when it fell."""
        return -1 if self.delta < 0 else 1

    def top_contributors(self, n: int = 5) -> list[tuple[str, Slice]]:
        """Largest movers in the direction of the overall change, across dimensions."""
        rows = [
            (dim, s)
            for dim, b in self.breakdowns.items()
            for s in b.ranked(self.direction)
        ]
        rows.sort(key=lambda r: abs(r[1].contribution_pp), reverse=True)
        return rows[:n]

    def headline(self) -> tuple[str, Slice] | None:
        """The single largest contributor, preferring the intersection view."""
        if self.intersection is not None:
            leading = self.intersection.leading(self.direction)
            if leading is not None:
                return self.intersection.dimension, leading
        top = self.top_contributors(1)
        return (top[0][0], top[0][1]) if top else None


def _slices(
    before: pd.Series, after: pd.Series, total_before: float, total_delta: float = 0.0
) -> list[Slice]:
    """Align two period aggregates over the union of segments — never the intersection.

    Reindexing on the union is what stops segments that exist in only one period
    from vanishing. A dropped segment makes the contributions fail to reconcile,
    which is precisely the bug this guards against.
    """
    segments = before.index.union(after.index)
    b = before.reindex(segments, fill_value=0.0)
    a = after.reindex(segments, fill_value=0.0)

    out = []
    for segment in segments:
        bv, av = float(b[segment]), float(a[segment])
        out.append(
            Slice(
                segment=str(segment),
                before=bv,
                after=av,
                change=av - bv,
                contribution_pp=100 * (av - bv) / total_before if total_before else 0.0,
                own_pct_change=100 * (av - bv) / bv if bv else float("inf") if av else 0.0,
                share_of_total_before=100 * bv / total_before if total_before else 0.0,
                share_of_change=(av - bv) / total_delta if total_delta else 0.0,
            )
        )
    out.sort(key=lambda s: s.change)
    return out


def _breakdown(
    frame: pd.DataFrame,
    dimension: str,
    metric: str,
    period_col: str,
    period_before: str,
    period_after: str,
    total_before: float,
    delta: float,
) -> Breakdown:
    normalized, report = normalize_categorical(frame[dimension])
    work = frame.assign(**{"__seg": normalized})

    grouped = work.groupby(["__seg", period_col], observed=True)[metric].sum()
    before = grouped.xs(period_before, level=period_col, drop_level=True)
    after = grouped.xs(period_after, level=period_col, drop_level=True)

    slices = _slices(before, after, total_before, delta)
    residual = delta - sum(s.change for s in slices)
    return Breakdown(dimension=dimension, slices=slices, residual=residual, normalization=report)


def _period_series(frame: pd.DataFrame, column: str, freq: str) -> pd.Series:
    """Bucket a time column into comparable period labels.

    Handles two shapes: a real datetime, and an integer year. Annual data often
    has no datetime at all, and forcing `2025` through `to_datetime` yields
    1970-01-01 plus 2025 nanoseconds — a silently wrong answer rather than an
    error, which is the worst kind.
    """
    series = frame[column]

    if pd.api.types.is_datetime64_any_dtype(series):
        if getattr(series.dtype, "tz", None) is not None:
            # Periods are timezone-naive. Dropping the offset explicitly avoids a
            # UserWarning and makes the choice visible: bucketing happens in the
            # column's own timezone, not the reader's local one.
            series = series.dt.tz_localize(None)
        return series.dt.to_period(freq).astype(str)

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any() and (numeric.dropna() % 1 == 0).all():
        return numeric.astype("Int64").astype(str)

    raise ValueError(
        f"Column '{column}' is neither a date nor an integer year, so it cannot "
        "be used as a time axis."
    )


def _candidate_dimensions(frame: pd.DataFrame, exclude: set[str], max_unique: int = 50) -> list[str]:
    out = []
    for column in frame.columns:
        if str(column) in exclude or pd.api.types.is_numeric_dtype(frame[column]):
            continue
        if pd.api.types.is_datetime64_any_dtype(frame[column]):
            continue
        # Compare on normalised cardinality: a column that is 4 real regions
        # spelled 16 ways is a dimension, not a free-text field.
        canonical = frame[column].astype("string").str.strip().str.lower()
        if 2 <= canonical.nunique(dropna=True) <= max_unique:
            out.append(str(column))
    return out


def analyze_change(
    frame: pd.DataFrame,
    *,
    metric: str,
    date_column: str,
    period_after: str | None = None,
    period_before: str | None = None,
    dimensions: list[str] | None = None,
    freq: str = "M",
) -> ChangeAnalysis:
    """Decompose the change in `metric` between two periods across dimensions."""
    work = frame.dropna(subset=[date_column, metric]).copy()
    if work.empty:
        raise ValueError(f"No rows with both '{date_column}' and '{metric}' present.")

    work["__period"] = _period_series(work, date_column, freq)
    totals = work.groupby("__period")[metric].sum().sort_index()
    if len(totals) < 2:
        raise ValueError(f"Need at least two {freq}-periods; found {len(totals)}.")

    after = period_after or str(totals.index[-1])
    if after not in totals.index:
        raise ValueError(f"Period {after!r} not in data. Available: {list(totals.index)}")
    earlier = [p for p in totals.index if p < after]
    if not earlier:
        raise ValueError(f"No period before {after!r} to compare against.")
    before = period_before or earlier[-1]

    total_before, total_after = float(totals[before]), float(totals[after])
    delta = total_after - total_before

    analysis = ChangeAnalysis(
        metric=metric,
        date_column=date_column,
        period_before=before,
        period_after=after,
        total_before=total_before,
        total_after=total_after,
        delta=delta,
        pct_change=100 * delta / total_before if total_before else float("nan"),
    )

    dims = dimensions or _candidate_dimensions(work, {metric, date_column, "__period"})
    pair = work[work["__period"].isin([before, after])]

    for dimension in dims:
        breakdown = _breakdown(
            pair, dimension, metric, "__period", before, after, total_before, delta
        )
        analysis.breakdowns[dimension] = breakdown
        if breakdown.normalization and breakdown.normalization.changed:
            analysis.notes.append(breakdown.normalization.describe())

    _add_intersection(analysis, pair, metric, before, after, total_before, delta)
    analysis.checks = _run_checks(analysis, pair, metric, before, after)
    return analysis


def _add_intersection(
    analysis: ChangeAnalysis,
    pair: pd.DataFrame,
    metric: str,
    before: str,
    after: str,
    total_before: float,
    delta: float,
) -> None:
    """Cross the two most concentrated dimensions.

    Single-dimension views each partition the same money, so "Laptop" and "West"
    are overlapping descriptions of one decline. Crossing them is what turns
    two partial views into a specific answer.
    """
    ranked = sorted(analysis.breakdowns.items(), key=lambda kv: kv[1].concentration, reverse=True)
    if len(ranked) < 2:
        return
    (dim_a, _), (dim_b, _) = ranked[0], ranked[1]

    norm_a, _ = normalize_categorical(pair[dim_a])
    norm_b, _ = normalize_categorical(pair[dim_b])
    work = pair.assign(__a=norm_a, __b=norm_b)
    work["__seg"] = work["__a"].astype(str) + " x " + work["__b"].astype(str)

    grouped = work.groupby(["__seg", "__period"], observed=True)[metric].sum()
    slices = _slices(
        grouped.xs(before, level="__period", drop_level=True),
        grouped.xs(after, level="__period", drop_level=True),
        total_before,
        delta,
    )
    analysis.intersection = Breakdown(
        dimension=f"{dim_a} x {dim_b}",
        slices=slices,
        residual=delta - sum(s.change for s in slices),
    )


def _run_checks(
    analysis: ChangeAnalysis, pair: pd.DataFrame, metric: str, before: str, after: str
) -> list[Check]:
    """Checks that can actually fail.

    A check whose condition is true by construction is decoration. Each of these
    compares two independently computed quantities.
    """
    checks: list[Check] = []

    for name, breakdown in analysis.breakdowns.items():
        checks.append(
            Check(
                name=f"'{name}' contributions reconcile to the total change",
                passed=breakdown.reconciles,
                detail=(
                    f"residual {breakdown.residual:,.2f} of {analysis.delta:,.2f}"
                    if not breakdown.reconciles
                    else f"sums to {analysis.delta:,.2f}"
                ),
            )
        )

    if analysis.intersection is not None:
        checks.append(
            Check(
                name=f"'{analysis.intersection.dimension}' contributions reconcile",
                passed=analysis.intersection.reconciles,
                detail=f"residual {analysis.intersection.residual:,.2f}",
            )
        )

    # Period totals recomputed from the untouched rows, independent of any groupby.
    recomputed_before = float(pair.loc[pair["__period"] == before, metric].sum())
    recomputed_after = float(pair.loc[pair["__period"] == after, metric].sum())
    drift = abs(recomputed_before - analysis.total_before) + abs(
        recomputed_after - analysis.total_after
    )
    checks.append(
        Check(
            name="Period totals match a direct recomputation",
            passed=drift <= RECONCILE_TOLERANCE,
            detail=f"drift {drift:,.4f}",
        )
    )

    top = analysis.top_contributors(1)
    if top:
        share = abs(top[0][1].contribution_pp / analysis.pct_change) if analysis.pct_change else 0
        checks.append(
            Check(
                name="Change is concentrated enough to attribute",
                passed=share >= DIFFUSE_THRESHOLD_PP,
                detail=(
                    f"largest single contributor explains {share:.0%} of the move"
                    + ("" if share >= DIFFUSE_THRESHOLD_PP else " — the change is broad, not driven by one segment")
                ),
            )
        )

    return checks
