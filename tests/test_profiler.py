"""Profiler behaviour — the profile is the model's only view of the data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analyst.profiler import profile_dataframe


def _col(profile, name):
    return next(c for c in profile.columns if c.name == name)


def test_detects_dates_stored_as_strings():
    df = pd.DataFrame({"d": ["01/15/2026", "02/20/2026", "03/01/2026"] * 5})
    assert _col(profile_dataframe(df), "d").semantic_type == "date_as_string"


def test_detects_currency_stored_as_strings():
    df = pd.DataFrame({"revenue": ["$1,234.56", "$99.00", "$12,000.10"] * 5})
    col = _col(profile_dataframe(df), "revenue")
    assert col.semantic_type == "numeric_as_string"
    assert "strip" in col.note


def test_numeric_id_column_is_not_treated_as_a_measure():
    """Summing an order_id is a classic silent wrong answer."""
    df = pd.DataFrame({"order_id": range(1000, 1100), "amount": np.arange(100.0)})
    assert _col(profile_dataframe(df), "order_id").semantic_type == "identifier"
    assert _col(profile_dataframe(df), "amount").semantic_type == "numeric"


def test_low_cardinality_strings_are_categorical_with_top_values():
    df = pd.DataFrame({"region": ["West"] * 60 + ["East"] * 40})
    col = _col(profile_dataframe(df), "region")
    assert col.semantic_type == "categorical"
    assert col.stats["top_values"] == {"West": 60, "East": 40}


def test_case_variants_surface_as_distinct_categories():
    """The model must be able to see that 'west' and 'West' are split."""
    df = pd.DataFrame({"region": ["West"] * 50 + ["west"] * 10 + ["East"] * 40})
    col = _col(profile_dataframe(df), "region")
    assert col.unique_count == 3
    assert "west" in col.stats["top_values"]


def test_flags_empty_and_constant_columns():
    df = pd.DataFrame({"a": [1, 1, 1], "empty": [np.nan] * 3, "b": ["x", "y", "z"]})
    profile = profile_dataframe(df)
    assert _col(profile, "a").semantic_type == "constant"
    assert _col(profile, "empty").semantic_type == "empty"
    assert profile.duplicate_rows == 0


def test_counts_fully_duplicated_rows():
    df = pd.DataFrame({"a": [1, 2, 1, 2], "b": ["x", "y", "x", "y"]})
    profile = profile_dataframe(df)
    assert profile.duplicate_rows == 2
    assert any("duplicated" in w for w in profile.warnings)


def test_empty_column_produces_exactly_one_warning():
    df = pd.DataFrame({"gone": [np.nan] * 10})
    warnings = [w for w in profile_dataframe(df).warnings if "'gone'" in w]
    assert warnings == ["'gone' is entirely null"]


def test_identifies_candidate_keys():
    df = pd.DataFrame({"id": ["a", "b", "c"], "val": [1, 1, 2]})
    assert profile_dataframe(df).candidate_keys == ["id"]


def test_handles_an_empty_frame_without_crashing():
    profile = profile_dataframe(pd.DataFrame({"a": []}))
    assert profile.n_rows == 0
    assert "dataset is empty" in profile.warnings


def test_prompt_rendering_stays_compact_on_a_wide_frame():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({f"c{i}": rng.normal(size=5_000) for i in range(30)})
    text = profile_dataframe(df).to_prompt()
    assert len(text) < 12_000  # ~3k tokens for a 30-column frame
    assert "c29" in text
