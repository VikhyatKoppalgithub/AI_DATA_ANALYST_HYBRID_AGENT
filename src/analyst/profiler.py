"""Dataset profiling.

The profile is what the model sees instead of the raw data. A 2M-row file
becomes roughly 800 tokens, so the profile has to carry every fact the model
needs to write correct code on the first attempt:

  - real dtypes, and the *semantic* type when it disagrees with the dtype
    (dates stored as strings, currency stored as "$1,234.00")
  - null and duplicate counts, so it knows what cleaning is warranted
  - cardinality and sample values, so it can guess which columns are
    dimensions worth decomposing by
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pandas.api import types as ptypes

SemanticType = Literal[
    "identifier",
    "date",
    "date_as_string",
    "year",
    "numeric",
    "numeric_as_string",
    "boolean",
    "categorical",
    "free_text",
    "constant",
    "empty",
]

# Values that mean "missing" but arrive as text. Real exports are full of these.
NULL_TOKENS = {"", "na", "n/a", "null", "none", "nan", "-", "--", "?", "unknown", "#n/a"}

_CURRENCY_RE = re.compile(r"^\s*[-+(]?\s*[$€£¥]?\s*[\d,]+(\.\d+)?\s*%?\s*\)?\s*$")
_ID_NAME_RE = re.compile(r"(^|_)(id|key|code|no|num|number|uuid|guid|sku|ref)($|_)", re.I)
_YEAR_NAME_RE = re.compile(r"(^|_)(year|yr|fy|fiscal_?year|period)($|_)", re.I)
# Values that mean "this row is a rollup of other rows", not a peer category.
_ROLLUP_VALUE_RE = re.compile(r"^\s*(level\s*\d+|total|all\b.*|subtotal|aggregate)\s*$", re.I)
YEAR_MIN, YEAR_MAX = 1900, 2100

MAX_SAMPLE_VALUES = 4
MAX_TOP_CATEGORIES = 6
CATEGORICAL_MAX_UNIQUE = 25
CATEGORICAL_MAX_RATIO = 0.5
FREE_TEXT_MIN_AVG_LEN = 40


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    semantic_type: SemanticType
    null_count: int
    null_pct: float
    unique_count: int
    samples: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    note: str | None = None


@dataclass
class DatasetProfile:
    source: str
    sheet: str | None
    n_rows: int
    n_cols: int
    memory_mb: float
    duplicate_rows: int
    columns: list[ColumnProfile]
    candidate_keys: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        """Compact text rendering for the model context."""
        return render_profile(self)


def _clean_str_series(s: pd.Series) -> pd.Series:
    """String values with null-ish tokens removed, for type sniffing."""
    text = s.dropna().astype(str).str.strip()
    return text[~text.str.lower().isin(NULL_TOKENS)]


def _looks_like_dates(sample: pd.Series) -> bool:
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() >= 0.9


def _looks_like_numbers(sample: pd.Series) -> bool:
    if sample.empty:
        return False
    return sample.str.match(_CURRENCY_RE).mean() >= 0.9


def _looks_like_years(s: pd.Series, name: str, unique_count: int) -> bool:
    """Detect a year stored as an integer, e.g. `Year` = 2025.

    Annual data very often has no date column at all — a plain integer year is
    the time axis. Without this it reads as an ordinary numeric column and any
    change-over-time question is refused.

    Kept conservative: a value in 1900-2100 is not enough on its own (a count
    could land there), so a name signal is required, or a shape that no measure
    plausibly has — few distinct values, all whole numbers, all in range.
    """
    values = s.dropna()
    if values.empty or unique_count < 2 or unique_count > 200:
        return False
    try:
        numeric = pd.to_numeric(values, errors="coerce").dropna()
    except (TypeError, ValueError):
        return False
    if len(numeric) < len(values) * 0.99:
        return False
    if not (numeric % 1 == 0).all():
        return False
    if not numeric.between(YEAR_MIN, YEAR_MAX).all():
        return False
    if _YEAR_NAME_RE.search(name):
        return True

    # No name signal, so the shape has to carry it. A year axis is a dense run
    # of consecutive years: 2013-2025 is 13 distinct values spanning 13. A
    # measure that merely lands in range — quantities of 1950, 2000, 2050 — is
    # sparse across its span, so density rejects it.
    span = int(numeric.max() - numeric.min()) + 1
    return unique_count >= 3 and span <= unique_count * 2


def _infer_semantic_type(
    s: pd.Series, name: str, non_null: int, unique_count: int
) -> tuple[SemanticType, str | None]:
    """Classify a column, preferring the semantic reading over the stored dtype."""
    if non_null == 0:
        return "empty", "column is entirely null"
    if unique_count == 1:
        return "constant", f"single value: {s.dropna().iloc[0]!r}"

    if ptypes.is_datetime64_any_dtype(s):
        return "date", None
    if ptypes.is_bool_dtype(s):
        return "boolean", None
    if _looks_like_years(s, name, unique_count):
        return "year", "an integer year — usable as the time axis, not a measure"
    if ptypes.is_numeric_dtype(s):
        # Integer-ish, near-unique, and named like a key -> an ID, not a measure.
        if _ID_NAME_RE.search(name) and unique_count / non_null > 0.95:
            return "identifier", "numeric but reads as an identifier — do not aggregate"
        return "numeric", None

    text = _clean_str_series(s)
    sample = text.head(500)

    if _looks_like_dates(sample):
        return "date_as_string", "stored as text — parse with pd.to_datetime before use"
    if _looks_like_numbers(sample):
        return "numeric_as_string", "stored as text — strip $ , % and cast before aggregating"

    lowered = set(text.str.lower().unique()[:10])
    if lowered and lowered <= {"true", "false", "yes", "no", "y", "n", "0", "1", "t", "f"}:
        return "boolean", "text booleans — normalize before filtering"

    if unique_count / non_null > 0.95 and (_ID_NAME_RE.search(name) or unique_count > 100):
        return "identifier", None

    avg_len = text.str.len().mean() if not text.empty else 0
    if avg_len >= FREE_TEXT_MIN_AVG_LEN:
        return "free_text", None

    if unique_count <= CATEGORICAL_MAX_UNIQUE or unique_count / non_null <= CATEGORICAL_MAX_RATIO:
        return "categorical", None

    return "free_text", None


def _column_stats(s: pd.Series, semantic: SemanticType) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    if semantic == "numeric":
        desc = s.describe()
        for key in ("min", "max", "mean", "50%", "std"):
            if key in desc:
                stats["median" if key == "50%" else key] = round(float(desc[key]), 4)
        if (s.dropna() < 0).any():
            stats["has_negatives"] = True
        if (s.dropna() == 0).any():
            stats["zero_count"] = int((s == 0).sum())
    elif semantic in ("date", "year"):
        valid = s.dropna()
        if not valid.empty:
            stats["min"] = str(valid.min())
            stats["max"] = str(valid.max())
    elif semantic in ("categorical", "boolean"):
        counts = s.value_counts().head(MAX_TOP_CATEGORIES)
        stats["top_values"] = {str(k): int(v) for k, v in counts.items()}
    return stats


def profile_column(s: pd.Series) -> ColumnProfile:
    n = len(s)
    null_count = int(s.isna().sum())
    non_null = n - null_count
    unique_count = int(s.nunique(dropna=True))

    semantic, note = _infer_semantic_type(s, str(s.name), non_null, unique_count)
    samples = [str(v) for v in s.dropna().unique()[:MAX_SAMPLE_VALUES]]

    return ColumnProfile(
        name=str(s.name),
        dtype=str(s.dtype),
        semantic_type=semantic,
        null_count=null_count,
        null_pct=round(100 * null_count / n, 2) if n else 0.0,
        unique_count=unique_count,
        samples=samples,
        stats=_column_stats(s, semantic),
        note=note,
    )


def profile_dataframe(
    df: pd.DataFrame, source: str = "<dataframe>", sheet: str | None = None
) -> DatasetProfile:
    columns = [profile_column(df[c]) for c in df.columns]
    n_rows = len(df)

    warnings: list[str] = []
    if n_rows == 0:
        warnings.append("dataset is empty")

    dup_rows = int(df.duplicated().sum())
    if dup_rows:
        warnings.append(f"{dup_rows} fully duplicated rows ({100 * dup_rows / n_rows:.1f}%)")

    for col in columns:
        if col.semantic_type == "empty":
            warnings.append(f"'{col.name}' is entirely null")
        elif col.null_pct > 50:
            warnings.append(f"'{col.name}' is {col.null_pct}% null")
        if col.semantic_type in ("date_as_string", "numeric_as_string"):
            warnings.append(f"'{col.name}' is {col.semantic_type.replace('_', ' ')}")

    for col in columns:
        if col.semantic_type != "categorical":
            continue
        values = [str(v) for v in col.stats.get("top_values", {})]
        if values and sum(bool(_ROLLUP_VALUE_RE.match(v)) for v in values) >= max(1, len(values) // 2):
            warnings.append(
                f"'{col.name}' looks like an aggregation level ({', '.join(values[:3])}); "
                "rows are probably nested, so summing across all of them double-counts"
            )

    unnamed = [c.name for c in columns if c.name.lower().startswith("unnamed:")]
    if unnamed:
        warnings.append(f"{len(unnamed)} unnamed column(s) — likely spreadsheet artifacts")

    candidate_keys = [
        c.name
        for c in columns
        if n_rows and c.null_count == 0 and c.unique_count == n_rows
    ]

    return DatasetProfile(
        source=source,
        sheet=sheet,
        n_rows=n_rows,
        n_cols=len(df.columns),
        memory_mb=round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        duplicate_rows=dup_rows,
        columns=columns,
        candidate_keys=candidate_keys,
        warnings=warnings,
    )


def load_and_profile(
    path: str | Path, sheet: str | int | None = None
) -> tuple[pd.DataFrame, DatasetProfile]:
    """Read a CSV/Excel file and profile it. Returns the frame and its profile."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".csv", ".tsv", ".txt"):
        sep = "\t" if suffix == ".tsv" else None
        # engine="python" lets pandas sniff the delimiter when sep is None.
        df = pd.read_csv(path, sep=sep, engine="python" if sep is None else "c")
        sheet_name = None
    elif suffix in (".xlsx", ".xls", ".xlsm"):
        target = 0 if sheet is None else sheet
        df = pd.read_excel(path, sheet_name=target)
        sheet_name = str(pd.ExcelFile(path).sheet_names[target] if isinstance(target, int) else target)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
        sheet_name = None
    elif suffix in (".json", ".jsonl"):
        df = pd.read_json(path, lines=suffix == ".jsonl")
        sheet_name = None
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    return df, profile_dataframe(df, source=path.name, sheet=sheet_name)


def render_profile(p: DatasetProfile) -> str:
    """Render a profile as compact text for the model context."""
    header = f"DATASET: {p.source}"
    if p.sheet:
        header += f" (sheet: {p.sheet})"

    lines = [
        header,
        f"{p.n_rows:,} rows x {p.n_cols} columns, {p.memory_mb} MB in memory",
        f"Fully duplicated rows: {p.duplicate_rows:,}",
        "",
        "COLUMNS",
    ]

    for c in p.columns:
        parts = [f"  {c.name}  [{c.semantic_type}, dtype={c.dtype}]"]
        meta = [f"nulls={c.null_count} ({c.null_pct}%)", f"unique={c.unique_count:,}"]
        parts.append("    " + ", ".join(meta))

        if c.stats:
            if "top_values" in c.stats:
                top = ", ".join(f"{k} ({v:,})" for k, v in c.stats["top_values"].items())
                parts.append(f"    top: {top}")
            else:
                parts.append(
                    "    " + ", ".join(f"{k}={v}" for k, v in c.stats.items())
                )
        if c.samples:
            parts.append("    e.g. " + " | ".join(c.samples))
        if c.note:
            parts.append(f"    NOTE: {c.note}")
        lines.append("\n".join(parts))

    if p.candidate_keys:
        lines += ["", f"CANDIDATE KEYS (unique, non-null): {', '.join(p.candidate_keys)}"]

    if p.warnings:
        lines += ["", "WARNINGS"]
        lines += [f"  - {w}" for w in p.warnings]

    return "\n".join(lines)
