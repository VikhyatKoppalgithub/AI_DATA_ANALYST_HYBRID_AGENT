"""Support for data shapes the original fixtures did not have.

Both bundled fixtures are wide (one column per metric) with a real date column.
A real government-statistics export — NZ's Annual Enterprise Survey — had
neither, and failed three separate ways:

  1. its time axis is an integer `Year`, not a date
  2. it is long format: one `Value` column holding 41 different measures,
     selected by `Variable_name`, in three different `Units`
  3. its rows are hierarchical, so summing every row triple-counts the totals

These tests cover each, using small frames with the same shape.
"""

from __future__ import annotations

import pandas as pd
import pytest

from analyst.analysis import analyze_change
from analyst.analysis.change import _period_series
from analyst.interpret import QuestionPlan, _validate_plan, apply_filters
from analyst.profiler import profile_dataframe


# ------------------------------------------------------- integer year axis


@pytest.mark.parametrize(
    "name,values,expected",
    [
        ("Year", list(range(2013, 2026)) * 3, "year"),
        ("fiscal_year", [2023, 2024, 2025] * 8, "year"),
        ("birth_year", [1990, 1985, 2001] * 9, "year"),
        ("period", list(range(2018, 2026)) * 4, "year"),
        # A dense run of years with no name signal is still a year axis.
        ("bucket", list(range(2019, 2026)) * 5, "year"),
        # Values that merely land in year range are not.
        ("quantity", [1950, 2000, 2050] * 9, "numeric"),
        ("price", [1999, 2049, 1899, 2099] * 7, "numeric"),
        ("units_sold", [3, 7, 2, 9] * 12, "numeric"),
    ],
)
def test_year_detection(name, values, expected):
    profile = profile_dataframe(pd.DataFrame({name: values}))
    assert profile.columns[0].semantic_type == expected


def test_an_integer_year_can_drive_change_analysis():
    frame = pd.DataFrame(
        {
            "Year": [2023] * 3 + [2024] * 3,
            "seg": ["a", "b", "c"] * 2,
            "v": [100.0, 50.0, 50.0, 60.0, 50.0, 50.0],
        }
    )
    analysis = analyze_change(frame, metric="v", date_column="Year", dimensions=["seg"])
    assert (analysis.period_before, analysis.period_after) == ("2023", "2024")
    assert analysis.pct_change == pytest.approx(-20.0)
    assert analysis.headline()[1].segment == "a"


def test_year_periods_are_not_silently_epoch_dates():
    """to_datetime(2025) is 1970 plus 2025 nanoseconds — wrong, not an error."""
    frame = pd.DataFrame({"Year": [2024, 2025]})
    assert _period_series(frame, "Year", "M").tolist() == ["2024", "2025"]


def test_a_non_temporal_column_is_rejected_rather_than_coerced():
    frame = pd.DataFrame({"label": ["a", "b"]})
    with pytest.raises(ValueError, match="neither a date nor an integer year"):
        _period_series(frame, "label", "M")


# --------------------------------------------------------- long format


@pytest.fixture
def long_frame() -> pd.DataFrame:
    """One value column holding two measures in two units, as tidy exports do."""
    rows = []
    for year in (2024, 2025):
        for measure, unit, base in [
            ("Total income", "Dollars (millions)", 1000.0),
            ("Total expenditure", "Dollars (millions)", 800.0),
            ("Margin", "Percentage", 20.0),
        ]:
            for industry, share in [("Retail", 0.6), ("Mining", 0.4)]:
                delta = 0.9 if (year == 2025 and industry == "Retail") else 1.0
                rows.append(
                    {
                        "Year": year,
                        "Variable_name": measure,
                        "Units": unit,
                        "Industry": industry,
                        "Value": base * share * delta,
                    }
                )
    return pd.DataFrame(rows)


def test_filters_narrow_to_one_measure_and_unit(long_frame):
    scoped = apply_filters(
        long_frame,
        [("Variable_name", "Total income"), ("Units", "Dollars (millions)")],
    )
    assert len(scoped) == 4
    assert set(scoped["Variable_name"]) == {"Total income"}

    analysis = analyze_change(
        scoped, metric="Value", date_column="Year", dimensions=["Industry"]
    )
    # 1000 -> 940: only Retail moved, and percentages were excluded.
    assert analysis.total_before == pytest.approx(1000.0)
    assert analysis.pct_change == pytest.approx(-6.0)
    assert analysis.headline()[1].segment == "Retail"


def test_without_filters_a_long_frame_sums_incompatible_units(long_frame):
    """The wrong answer this feature exists to prevent."""
    analysis = analyze_change(
        long_frame, metric="Value", date_column="Year", dimensions=["Industry"]
    )
    assert analysis.total_before == pytest.approx(1820.0)  # dollars + percentages


def test_filters_are_matched_case_and_whitespace_insensitively(long_frame):
    scoped = apply_filters(long_frame, [("Variable_name", "  total income ")])
    assert len(scoped) == 4


# -------------------------------------------------- plan validation


def _profile_of(frame: pd.DataFrame):
    return profile_dataframe(frame, source="t.csv")


def test_a_filter_on_an_absent_value_is_dropped_not_applied(long_frame):
    """Filtering to zero rows would yield an empty analysis rather than an error."""
    plan = _validate_plan(
        {
            "metric": "Value",
            "date_column": "Year",
            "period_after": "2025",
            "dimensions": ["Industry"],
            "filters": [{"column": "Variable_name", "value": "Nonexistent measure"}],
        },
        _profile_of(long_frame),
        frame=long_frame,
    )
    assert plan.filters == []
    assert any("no such value" in w for w in plan.warnings)


def test_a_filter_on_an_unknown_column_is_dropped(long_frame):
    plan = _validate_plan(
        {
            "metric": "Value",
            "date_column": "Year",
            "period_after": "2025",
            "dimensions": [],
            "filters": [{"column": "not_a_column", "value": "x"}],
        },
        _profile_of(long_frame),
        frame=long_frame,
    )
    assert plan.filters == []
    assert any("unknown column" in w for w in plan.warnings)


def test_a_filtered_column_is_not_also_used_as_a_dimension(long_frame):
    """Grouping by a column pinned to one value produces a single useless slice."""
    plan = _validate_plan(
        {
            "metric": "Value",
            "date_column": "Year",
            "period_after": "2025",
            "dimensions": ["Industry", "Units"],
            "filters": [{"column": "Units", "value": "Dollars (millions)"}],
        },
        _profile_of(long_frame),
        frame=long_frame,
    )
    assert plan.dimensions == ["Industry"]
    assert plan.filters == [("Units", "Dollars (millions)")]


def test_a_bare_year_is_a_valid_period(long_frame):
    plan = _validate_plan(
        {
            "metric": "Value",
            "date_column": "Year",
            "period_after": "2025",
            "dimensions": [],
            "filters": [],
        },
        _profile_of(long_frame),
        frame=long_frame,
    )
    assert plan.period_after == "2025"
    assert not any("malformed" in w for w in plan.warnings)


# ------------------------------------------------ hierarchical rows


def test_nested_aggregation_levels_are_flagged_to_the_planner():
    """Summing across levels double-counts; the planner must be told."""
    frame = pd.DataFrame(
        {
            "Industry_aggregation": ["Level 1", "Level 2", "Level 3"] * 10,
            "Value": [1.0] * 30,
        }
    )
    warnings = profile_dataframe(frame).warnings
    assert any("aggregation level" in w and "double-count" in w for w in warnings)


def test_ordinary_categories_are_not_flagged_as_aggregation_levels():
    frame = pd.DataFrame({"region": ["West", "East", "South"] * 10, "v": [1.0] * 30})
    assert not any("aggregation level" in w for w in profile_dataframe(frame).warnings)


# ------------------------------------------------ snapshotting for the sandbox


def test_a_mixed_type_column_does_not_break_the_sandbox_snapshot(tmp_path):
    """Parquet requires one type per column; spreadsheets do not oblige.

    A worksheet with a title row read as a header produces an object column
    holding both str and int, which raises ArrowTypeError on to_parquet and
    previously crashed before the sandbox ever started.
    """
    from analyst.llm.base import Completion, Provider, ProviderInfo
    from analyst.prepare import prepare_frame
    from analyst.session import AnalystSession

    class Stub(Provider):
        info = ProviderInfo(name="stub", model="stub", local=True)

        def complete(self, **kwargs):
            raise NotImplementedError

        def chat(self, **kwargs):
            raise NotImplementedError

        def health(self):
            return True, "stub"

    frame = pd.DataFrame(
        {"notes": ["instructions", 1, "more text", 2, "x"], "value": [1.0, 2.0, 3.0, 4.0, 5.0]}
    )
    with pytest.raises(Exception):  # the failure this guards against
        frame.to_parquet(tmp_path / "probe.parquet", index=False)

    data = prepare_frame(frame, profile_dataframe(frame, source="worksheet.xlsx"))
    session = AnalystSession(Stub(), data, workdir=tmp_path)
    try:
        snapshot, reader = session._write_snapshot()
        assert snapshot.exists()
        assert reader == "read_pickle"
        restored = pd.read_pickle(snapshot)
        assert list(restored.columns) == ["notes", "value"]
        assert str(restored["value"].dtype) == "float64"
    finally:
        session.close()


def test_a_clean_frame_still_uses_parquet(tmp_path):
    from analyst.llm.base import Provider, ProviderInfo
    from analyst.prepare import prepare_frame
    from analyst.session import AnalystSession

    class Stub(Provider):
        info = ProviderInfo(name="stub", model="stub", local=True)

        def complete(self, **kwargs):
            raise NotImplementedError

        def chat(self, **kwargs):
            raise NotImplementedError

        def health(self):
            return True, "stub"

    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    data = prepare_frame(frame, profile_dataframe(frame, source="clean.csv"))
    session = AnalystSession(Stub(), data, workdir=tmp_path)
    try:
        _, reader = session._write_snapshot()
        assert reader == "read_parquet"
    finally:
        session.close()


# ------------------------------------------------- data-quality reporting


def test_quality_issues_catch_what_the_prompt_threshold_misses():
    """profile.warnings only fires above 50% null — useless as a user signal."""
    from analyst.prepare import prepare_frame

    frame = pd.DataFrame(
        {
            "a": [1.0, 2.0, None] * 10,          # 33% null: under the warn threshold
            "b": ["x", "y", "z"] * 10,
            "amount": ["$1.00", "$2.00", "$3.00"] * 10,   # numeric stored as text
        }
    )
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)  # a duplicate row

    data = prepare_frame(frame, profile_dataframe(frame, source="t.csv"))
    issues = data.quality_issues

    assert any("missing values" in i for i in issues)
    assert any("duplicate rows" in i for i in issues)
    assert len(issues) > len(data.profile.warnings)


def test_the_cleaning_log_reaches_the_codegen_prompt():
    """Problems fixed before the model looks are invisible unless it is told."""
    from analyst.codegen import CodegenSession
    from analyst.prepare import prepare_frame

    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    frame = pd.concat([frame, frame], ignore_index=True)  # every row duplicated
    data = prepare_frame(frame, profile_dataframe(frame, source="t.csv"))

    session = CodegenSession(
        None, None, data.profile,
        preparation_log=[a.description for a in data.actions],
    )
    assert "Already applied" in session.system
    assert "duplicated row" in session.system
    assert "cannot rediscover them" in session.system


def test_no_preparation_log_leaves_the_prompt_unchanged():
    from analyst.codegen import CodegenSession

    frame = pd.DataFrame({"a": [1, 2, 3]})
    session = CodegenSession(None, None, profile_dataframe(frame), preparation_log=[])
    assert "Already applied" not in session.system
