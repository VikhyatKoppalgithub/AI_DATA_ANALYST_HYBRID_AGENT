"""Tests for the graders.

A grader that cannot fail is worse than no grader — it reports success while
measuring nothing. Every assertion here is exercised in both directions.
"""

from __future__ import annotations

import pandas as pd
import pytest

from analyst.analysis import analyze_change, coerce_numeric
from analyst.evals import grade as g
from analyst.evals.cases import SUITE
from analyst.codegen import Attempt, CodegenRun
from analyst.interpret import Answer, QuestionPlan
from analyst.sandbox import ExecResult
from analyst.session import RoutedAnswer


@pytest.fixture(scope="module")
def analysis():
    df = pd.read_csv("data/sales_2026.csv")
    df["revenue"] = coerce_numeric(df["revenue"])
    df["order_date"] = pd.to_datetime(df["order_date"], format="%m/%d/%Y")
    return analyze_change(
        df.drop_duplicates(),
        metric="revenue",
        date_column="order_date",
        period_after="2026-03",
    )


def make_answer(analysis, narrative: str, **plan_kwargs) -> RoutedAnswer:
    """A change-route answer, as the session would return it."""
    plan = QuestionPlan(
        metric=plan_kwargs.get("metric", "revenue"),
        date_column="order_date",
        period_after="2026-03",
        dimensions=plan_kwargs.get("dimensions", ["region", "product", "channel"]),
    )
    change = Answer(question="q", narrative=narrative, analysis=analysis, plan=plan)
    return RoutedAnswer(
        question="q", route="change", narrative=narrative, verified=True, change=change
    )


def make_code_answer(narrative: str, *, stdout: str = "", ok: bool = True) -> RoutedAnswer:
    """A code-route answer whose only computed figures are in `stdout`."""
    run = CodegenRun(question="q", answer=narrative)
    run.attempts.append(
        Attempt(code="print(1)", result=ExecResult(ok=ok, stdout=stdout))
    )
    return RoutedAnswer(
        question="q", route="code", narrative=narrative, verified=False, code=run
    )


# ------------------------------------------------------------- both directions


def test_no_invented_numbers_catches_a_fabricated_figure(analysis):
    answer = make_answer(analysis, "Laptop x West contributed -99.87 percentage points.")
    passed, detail = g.no_invented_numbers()(answer)
    assert not passed and "99.87" in detail


def test_no_invented_numbers_accepts_traceable_figures(analysis):
    answer = make_answer(analysis, "Laptop x West contributed -11.08 points of the -16.1% change.")
    passed, _ = g.no_invented_numbers()(answer)
    assert passed


def test_no_causal_claim_catches_causal_language(analysis):
    answer = make_answer(analysis, "Revenue fell due to weak laptop sales in the West.")
    passed, detail = g.no_causal_claim()(answer)
    assert not passed and "due to" in detail


def test_no_causal_claim_allows_arithmetic_explanation(analysis):
    """'explains little because it is a small share' is arithmetic, not causation."""
    answer = make_answer(
        analysis,
        "Docking Station x South explains only -0.66 pp because it was just 1.1% of revenue.",
    )
    passed, _ = g.no_causal_claim()(answer)
    assert passed


def test_no_summed_contributions_catches_double_counting(analysis):
    answer = make_answer(
        analysis, "Laptop, West and Direct each contributed over 10 percentage points."
    )
    passed, detail = g.no_summed_contributions()(answer)
    assert not passed and "each contribut" in detail


def test_no_summed_contributions_accepts_a_single_attribution(analysis):
    answer = make_answer(analysis, "Laptop x West was the largest contributor at -11.08 pp.")
    passed, _ = g.no_summed_contributions()(answer)
    assert passed


def test_top_slice_assertion_both_ways(analysis):
    answer = make_answer(analysis, "")
    assert g.top_slice_is("Laptop", "West")(answer)[0]
    assert not g.top_slice_is("Keyboard", "South")(answer)[0]


def test_contribution_near_is_tolerant_but_not_blind(analysis):
    answer = make_answer(analysis, "")
    assert g.contribution_near(("Laptop", "West"), -11.08)(answer)[0]
    assert not g.contribution_near(("Laptop", "West"), -5.00)(answer)[0]


def test_plan_metric_both_ways(analysis):
    answer = make_answer(analysis, "", metric="revenue")
    assert g.plan_metric("revenue")(answer)[0]
    assert not g.plan_metric("units")(answer)[0]


def test_plan_dimensions_reports_what_is_missing(analysis):
    answer = make_answer(analysis, "", dimensions=["region"])
    passed, detail = g.plan_includes_dimensions("region", "product")(answer)
    assert not passed and "product" in detail


def test_leads_with_requires_the_first_sentence(analysis):
    buried = make_answer(
        analysis, "Several segments moved. Laptop x West was the largest contributor."
    )
    assert not g.narrative_leads_with("Laptop", "West")(buried)[0]

    upfront = make_answer(analysis, "Laptop x West was the largest contributor. Others moved less.")
    assert g.narrative_leads_with("Laptop", "West")(upfront)[0]


def test_decoy_assertion_identifies_the_planted_decoy(analysis):
    answer = make_answer(analysis, "")
    assert g.decoy_flagged("Docking Station", "South")(answer)[0]
    assert not g.decoy_flagged("Laptop", "West")(answer)[0]


# ------------------------------------------------------------------ harness


def test_grade_reports_every_assertion_with_its_family(analysis):
    answer = make_answer(analysis, "Laptop x West contributed -11.08 pp.")
    results = g.grade(
        answer,
        {
            "interpretation": [g.plan_metric("revenue")],
            "correctness": [g.top_slice_is("Laptop", "West")],
            "communication": [g.no_invented_numbers()],
        },
    )
    assert len(results) == 3
    assert {r.family for r in results} == {"interpretation", "correctness", "communication"}
    assert all(r.passed for r in results)


def test_a_raising_assertion_is_a_failure_not_a_crash(analysis):
    def broken(_answer):
        raise RuntimeError("boom")

    results = g.grade(make_answer(analysis, ""), {"correctness": [broken]})
    assert not results[0].passed
    assert "RuntimeError" in results[0].detail


# --------------------------------------------------------- code-route graders


def test_code_figures_must_trace_to_executed_output():
    answer = make_code_answer("The correlation is -0.1468.", stdout="-0.0033")
    passed, detail = g.no_invented_numbers()(answer)
    assert not passed and "0.1468" in detail


def test_code_figures_present_in_output_are_accepted():
    answer = make_code_answer("The correlation is -0.0033.", stdout="-0.0033")
    assert g.no_invented_numbers()(answer)[0]


def test_executed_code_requires_a_successful_run():
    assert g.executed_code()(make_code_answer("x", stdout="1", ok=True))[0]
    assert not g.executed_code()(make_code_answer("x", stdout="", ok=False))[0]


def test_reports_value_both_ways():
    answer = make_code_answer("The average is 2.6961 units.", stdout="2.6961")
    assert g.reports_value(2.6961, tolerance=0.01)(answer)[0]
    assert not g.reports_value(99.0, tolerance=0.01)(answer)[0]


def test_route_and_verified_assertions(analysis):
    change = make_answer(analysis, "x")
    code = make_code_answer("x", stdout="1")
    assert g.route_is("change")(change)[0] and not g.route_is("code")(change)[0]
    assert g.is_verified(True)(change)[0]
    assert g.is_verified(False)(code)[0]


def test_change_graders_fail_clearly_on_a_code_route_answer():
    """A change assertion must not crash when the question was routed elsewhere."""
    code = make_code_answer("x", stdout="1")
    passed, detail = g.all_verification_passes()(code)
    assert not passed and "routed to 'code'" in detail


def test_suite_cases_have_unique_ids_and_real_assertions():
    ids = [c.id for c in SUITE]
    assert len(ids) == len(set(ids))
    for case in SUITE:
        assert case.question.strip()
        assert sum(len(v) for v in case.assertions.values()) >= 3, case.id
