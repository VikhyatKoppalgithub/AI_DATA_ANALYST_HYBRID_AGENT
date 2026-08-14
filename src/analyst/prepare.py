"""Turn a raw uploaded frame into one that can be analysed.

Every transformation is recorded and returned. Silent cleaning is how an
analysis becomes untrustworthy: a user who cannot see that 188 rows were
dropped has no way to judge whether the answer is about their data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from analyst.analysis.normalize import coerce_numeric
from analyst.profiler import DatasetProfile, profile_dataframe


@dataclass
class CleaningAction:
    description: str
    rows_affected: int = 0
    reversible: bool = True


@dataclass
class PreparedData:
    frame: pd.DataFrame
    profile: DatasetProfile
    actions: list[CleaningAction] = field(default_factory=list)
    rows_before: int = 0
    rows_after: int = 0

    @property
    def rows_removed(self) -> int:
        return self.rows_before - self.rows_after

    @property
    def quality_issues(self) -> list[str]:
        """Concrete problems in this data, whether or not cleaning fixed them.

        Distinct from `profile.warnings`, which only fires above a 50% null
        threshold — tuned to keep the model's prompt short, and far too blunt to
        show a person. A file with 3,000 missing ratings and 350 duplicate rows
        reported zero warnings under that rule.
        """
        issues = [
            f"'{c.name}' has {c.null_count:,} missing values "
            f"({c.null_pct}%)"
            for c in self.profile.columns
            if c.null_count
        ]
        if self.rows_removed:
            issues.append(f"{self.rows_removed:,} duplicate rows were removed")
        issues += [
            f"'{c.name}' is {c.semantic_type.replace('_', ' ')}"
            for c in self.profile.columns
            if c.semantic_type in ("numeric_as_string", "date_as_string", "empty")
        ]
        return issues


def prepare_frame(
    raw: pd.DataFrame, profile: DatasetProfile, *, drop_duplicates: bool = True
) -> PreparedData:
    """Coerce types the profiler flagged as mis-stored, then optionally dedupe."""
    df = raw.copy()
    actions: list[CleaningAction] = []
    rows_before = len(df)

    for column in profile.columns:
        if column.name not in df.columns:
            continue

        if column.semantic_type == "numeric_as_string":
            converted = coerce_numeric(df[column.name])
            failed = int(converted.isna().sum() - df[column.name].isna().sum())
            df[column.name] = converted
            note = f"Parsed '{column.name}' from text to numbers"
            if failed > 0:
                note += f"; {failed:,} value(s) could not be parsed and became missing"
            actions.append(CleaningAction(note, rows_affected=len(df)))

        elif column.semantic_type == "date_as_string":
            converted = pd.to_datetime(df[column.name], errors="coerce", format="mixed")
            failed = int(converted.isna().sum() - df[column.name].isna().sum())
            df[column.name] = converted
            note = f"Parsed '{column.name}' from text to dates"
            if failed > 0:
                note += f"; {failed:,} value(s) could not be parsed and became missing"
            actions.append(CleaningAction(note, rows_affected=len(df)))

        elif column.semantic_type == "empty":
            df = df.drop(columns=[column.name])
            actions.append(
                CleaningAction(f"Dropped '{column.name}' — every value was missing")
            )

    if drop_duplicates:
        duplicates = int(df.duplicated().sum())
        if duplicates:
            df = df.drop_duplicates().reset_index(drop=True)
            actions.append(
                CleaningAction(
                    f"Removed {duplicates:,} fully duplicated row(s)",
                    rows_affected=duplicates,
                    reversible=False,
                )
            )
        else:
            actions.append(CleaningAction("No duplicated rows found"))
    elif int(raw.duplicated().sum()):
        actions.append(
            CleaningAction(
                f"Kept {int(raw.duplicated().sum()):,} duplicated row(s) — totals will "
                "count them more than once",
            )
        )

    return PreparedData(
        frame=df,
        profile=profile_dataframe(df, source=profile.source, sheet=profile.sheet),
        actions=actions,
        rows_before=rows_before,
        rows_after=len(df),
    )
