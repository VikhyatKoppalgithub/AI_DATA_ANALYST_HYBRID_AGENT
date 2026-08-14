"""Categorical normalisation for grouping.

Grouping on a raw column is where messy data silently produces wrong answers:
`West`, `west`, and `" West "` become three segments, each holding part of the
truth. Nothing errors — the totals just quietly disagree.

Normalisation here is reported, never silent. Every function returns what it
changed so the caller can surface it.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from analyst.profiler import NULL_TOKENS

MISSING = "(missing)"


@dataclass
class NormalizationReport:
    column: str
    raw_segments: int
    normalized_segments: int
    merged: dict[str, list[str]]  # canonical -> raw variants folded into it
    missing_rows: int

    @property
    def changed(self) -> bool:
        return bool(self.merged) or self.missing_rows > 0

    def describe(self) -> str:
        parts = []
        if self.merged:
            examples = "; ".join(
                f"{canon} <- {', '.join(repr(v) for v in variants)}"
                for canon, variants in list(self.merged.items())[:3]
            )
            more = "" if len(self.merged) <= 3 else f" (+{len(self.merged) - 3} more)"
            parts.append(
                f"'{self.column}': merged {self.raw_segments} raw values into "
                f"{self.normalized_segments} segments — {examples}{more}"
            )
        if self.missing_rows:
            parts.append(
                f"'{self.column}': {self.missing_rows:,} rows had no usable value "
                f"and were grouped as {MISSING} rather than dropped"
            )
        return "; ".join(parts)


def normalize_categorical(series: pd.Series) -> tuple[pd.Series, NormalizationReport]:
    """Fold case and whitespace variants together; make missing values explicit.

    Missing becomes a real segment rather than a dropped row. A dimension that
    silently drops rows cannot reconcile against the total, and reconciliation
    is the check that catches everything else.
    """
    raw = series.astype("string")
    stripped = raw.str.strip()

    blank = stripped.isna() | stripped.str.lower().isin(NULL_TOKENS)
    missing_rows = int(blank.sum())

    canonical = stripped.str.lower()
    # Recover a readable label: the most common raw spelling of each canonical form.
    labels: dict[str, str] = {}
    merged: dict[str, list[str]] = {}
    for key, group in stripped[~blank].groupby(canonical[~blank]):
        variants = group.value_counts()
        label = str(variants.index[0])
        labels[str(key)] = label
        if len(variants) > 1:
            merged[label] = [str(v) for v in variants.index[1:]]

    result = canonical.map(labels).astype("string")
    result[blank] = MISSING

    return result.fillna(MISSING), NormalizationReport(
        column=str(series.name),
        raw_segments=int(stripped.nunique(dropna=True)),
        normalized_segments=int(result.nunique(dropna=True)),
        merged=merged,
        missing_rows=missing_rows,
    )


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Parse currency/percentage text into floats. Percentages become fractions."""
    if pd.api.types.is_numeric_dtype(series):
        return series
    text = series.astype("string").str.strip()
    is_pct = text.str.contains("%", regex=False, na=False)
    # Parenthesised negatives: "(1,234.00)" is accounting notation for -1234.00
    negated = text.str.match(r"^\(.*\)$", na=False)
    cleaned = text.str.replace(r"[$€£¥,()%\s]", "", regex=True)
    numeric = pd.to_numeric(cleaned, errors="coerce")
    numeric = numeric.where(~negated, -numeric)
    return numeric.where(~is_pct, numeric / 100)
