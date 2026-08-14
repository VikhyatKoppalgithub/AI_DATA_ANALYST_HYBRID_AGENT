"""Failures found by running a real file, not by reading the code.

The bundled fixtures are synthetic, so they contain the mess I thought to add.
NYC's Citywide Payroll export (1,113,117 rows, FY2024 vs FY2025) contained mess
I did not, and broke five separate ways:

  1. the decoy warning hardcoded the word "revenue", so a payroll analysis was
     told a segment was "1.1% of revenue"
  2. the concentration check is one-sided, so a top contributor explaining 126%
     of the net change passed as "concentrated" — that figure is the signature
     of large offsetting movements, not of concentration
  3. `last_name` (125,431 distinct, 11% of rows) profiles as categorical and was
     accepted as a dimension, because the engine's 50-value cap only applies
     when the plan names no dimensions at all
  4. `payroll_number` holds 159 agency codes; it escapes the near-uniqueness
     identifier rule and so became the *first* numeric metric fallback
  5. `agency_start_date` (hire dates from 1901) sorts ahead of `fiscal_year`, so
     the planner — told to infer the year from the range it is shown — was shown
     the hire dates rather than the two reporting years

Each is covered below, in both directions where a one-sided test would pass for
the wrong reason.
"""

from __future__ import annotations

import pandas as pd
import pytest

from analyst.analysis import analyze_change
from analyst.interpret import (
    _analysis_facts,
    _describe_time_axes,
    _numeric_candidates,
    _too_granular,
    _validate_plan,
)
from analyst.profiler import profile_dataframe


def _two_period_frame(rows: list[tuple[str, int, float]], metric: str = "value") -> pd.DataFrame:
    """(segment, year, value) triples -> a frame the change engine can analyse."""
    return pd.DataFrame(
        {"seg": [r[0] for r in rows], "Year": [r[1] for r in rows], metric: [r[2] for r in rows]}
    )


def _offsetting(name: str = "seg") -> pd.DataFrame:
    """A -5% net that is really -20pp of falls against +15pp of rises."""
    return _two_period_frame(
        [("A", 2023, 100.0), ("B", 2023, 100.0), ("A", 2024, 60.0), ("B", 2024, 130.0)]
    ).rename(columns={"seg": name})


# ------------------------------------------------------ offsetting movements


def test_offsetting_movements_fail_the_check():
    analysis = analyze_change(
        _offsetting(), metric="value", date_column="Year", dimensions=["seg"]
    )
    assert analysis.pct_change == pytest.approx(-5.0)

    check = next(c for c in analysis.checks if "offsetting" in c.name)
    assert not check.passed
    assert not analysis.passed, "an offsetting-dominated change must not read as verified"
    # The detail has to carry the counter-mover, or the narrator cannot report it.
    assert "B" in check.detail


def test_a_one_directional_change_passes_the_offsetting_check():
    """The shape both bundled fixtures have: everything moves one way."""
    frame = _two_period_frame(
        [("A", 2023, 100.0), ("B", 2023, 100.0), ("A", 2024, 60.0), ("B", 2024, 100.0)]
    )
    analysis = analyze_change(frame, metric="value", date_column="Year", dimensions=["seg"])

    check = next(c for c in analysis.checks if "offsetting" in c.name)
    assert check.passed
    assert analysis.passed


def test_offsetting_is_detected_when_the_metric_rises():
    """Direction bugs only appear on a rising metric — the same trap as `ranked`."""
    frame = _two_period_frame(
        [("A", 2023, 100.0), ("B", 2023, 100.0), ("A", 2024, 160.0), ("B", 2024, 70.0)]
    )
    analysis = analyze_change(frame, metric="value", date_column="Year", dimensions=["seg"])
    assert analysis.pct_change == pytest.approx(15.0)

    check = next(c for c in analysis.checks if "offsetting" in c.name)
    assert not check.passed
    assert "B" in check.detail


def test_counter_movers_are_exactly_what_ranked_discards():
    analysis = analyze_change(
        _offsetting(), metric="value", date_column="Year", dimensions=["seg"]
    )
    breakdown = analysis.breakdowns["seg"]

    with_change = {s.segment for s in breakdown.ranked(analysis.direction)}
    against = {s.segment for s in breakdown.counter_movers(analysis.direction)}
    assert with_change == {"A"}
    assert against == {"B"}
    assert not (with_change & against), "a slice cannot both move with and against"
    assert breakdown.counter_movement(analysis.direction) == pytest.approx(15.0)


# --------------------------------------------------- high-cardinality dimensions


def _wide_frame(n_rows: int, n_unique: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label": [f"v{i % n_unique}" for i in range(n_rows)],
            "Year": [2023 + (i % 2) for i in range(n_rows)],
            "value": [1.0] * n_rows,
        }
    )


def test_a_per_entity_column_is_rejected_as_a_dimension():
    """`last_name`: 2,000 distinct over 20,000 rows, the shape of a name column."""
    frame = _wide_frame(20_000, 2_000)
    profile = profile_dataframe(frame)
    assert _too_granular(profile, "label")

    plan = _validate_plan(
        {
            "metric": "value",
            "date_column": "Year",
            "period_after": "2024",
            "dimensions": ["label"],
            "filters": [],
        },
        profile,
        frame=frame,
    )
    assert "label" not in plan.dimensions
    assert any("too many distinct values" in w for w in plan.warnings)


def test_a_legitimately_wide_dimension_is_kept():
    """`title_description`: 1,100 job titles over 30,000 rows is a real dimension.

    Without the ratio condition this would be rejected too, and the payroll
    breakdown that actually produced the answer would have been thrown away.
    """
    frame = _wide_frame(30_000, 1_100)
    profile = profile_dataframe(frame)
    assert not _too_granular(profile, "label")

    plan = _validate_plan(
        {
            "metric": "value",
            "date_column": "Year",
            "period_after": "2024",
            "dimensions": ["label"],
            "filters": [],
        },
        profile,
        frame=frame,
    )
    assert plan.dimensions == ["label"]
    assert not any("too many distinct values" in w for w in plan.warnings)


def test_a_small_column_is_never_rejected_for_cardinality():
    frame = _wide_frame(500, 4)
    assert not _too_granular(profile_dataframe(frame), "label")


# ------------------------------------------------------- coded ID columns


def test_a_coded_id_column_is_not_the_first_metric_fallback():
    frame = pd.DataFrame(
        {
            "payroll_number": [67, 3, 15] * 40,
            "total_ot_paid": [100.0, 200.0, 300.0] * 40,
        }
    )
    profile = profile_dataframe(frame)

    candidates = _numeric_candidates(profile)
    assert candidates[0] == "total_ot_paid"
    assert candidates[-1] == "payroll_number", "a coded ID must sort last, not vanish"


def test_a_low_cardinality_coded_column_is_flagged_but_stays_numeric():
    frame = pd.DataFrame({"payroll_number": [67, 3, 15] * 40})
    column = profile_dataframe(frame).columns[0]

    assert column.semantic_type == "numeric"
    assert column.note is not None and "not a quantity to sum" in column.note


def test_a_near_unique_id_is_still_an_identifier():
    """The original rule must keep working: `ticket_id` is not a measure."""
    frame = pd.DataFrame({"ticket_id": list(range(500))})
    column = profile_dataframe(frame).columns[0]
    assert column.semantic_type == "identifier"


def test_an_ordinary_measure_is_not_flagged():
    frame = pd.DataFrame({"revenue": [10.0, 20.0, 30.0] * 40})
    column = profile_dataframe(frame).columns[0]
    assert column.semantic_type == "numeric"
    assert column.note is None


# ----------------------------------------------------------- time axes


def test_every_time_axis_is_described_to_the_planner():
    """Both axes, or the planner infers a period from the wrong column's range."""
    frame = pd.DataFrame(
        {
            "fiscal_year": [2024, 2025] * 10,
            "agency_start_date": pd.to_datetime(["1990-01-01", "2020-06-01"] * 10),
            "value": [1.0] * 20,
        }
    )
    described = _describe_time_axes(profile_dataframe(frame), frame)

    assert "fiscal_year" in described
    assert "agency_start_date" in described
    assert "2024" in described and "2025" in described


# -------------------------------------------------------- decoy wording


def test_the_decoy_warning_names_the_actual_metric():
    """It said "of revenue" on a payroll file, which is simply a false statement."""
    frame = pd.DataFrame(
        {
            "dim_a": ["big", "decoy", "big", "decoy"],
            "dim_b": ["x", "y", "x", "y"],
            "Year": [2023, 2023, 2024, 2024],
            "total_ot_paid": [1000.0, 10.0, 800.0, 3.0],
        }
    )
    analysis = analyze_change(
        frame,
        metric="total_ot_paid",
        date_column="Year",
        dimensions=["dim_a", "dim_b"],
    )
    decoys = [s for s in analysis.intersection.slices if s.is_decoy]
    assert decoys, "fixture must actually contain a decoy for this test to mean anything"

    facts = _analysis_facts(analysis)
    assert "total_ot_paid" in facts
    assert "of revenue" not in facts
