"""The eval suite.

Expected values come from scripts/make_demo_data.py, which prints the ground
truth it generated. They are not the pipeline's own output — grading against
your own output only measures self-consistency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from analyst.evals import grade as g

# Absolute, so the suite runs from any working directory.
_DATA = Path(__file__).resolve().parents[3] / "data"
SALES = _DATA / "sales_2026.csv"
SUPPORT = _DATA / "support_tickets_2026.csv"
FIXTURE = SALES  # default for the sales cases

# Ground truth for the Feb -> Mar 2026 shock.
TOTAL_PCT = -16.11
TOP = ("Laptop", "West")
TOP_PP = -11.08
DECOY = ("Docking Station", "South")


# Ground truth for the code-route cases, computed independently from the
# prepared frame (see the header note — never from the pipeline's own output).
CORRELATION_UNITS_PRICE = -0.0033
TOP_REP, TOP_REP_REVENUE = "rep_34", 445532.17
SECOND_REP_REVENUE = 401537.94
DISTINCT_PRODUCTS = 4
MEAN_UNITS = 2.6961

# --- support_tickets_2026.csv: SLA breaches, Feb -> Mar 2026 -----------------
# A different domain, metric, date format, and mess. The shock has the same two
# properties as the sales fixture, because that is what the evals test.
SUPPORT_TOTAL_PCT = 48.36
SUPPORT_TOP = ("P1", "Platform")
SUPPORT_TOP_PP = 44.13
SUPPORT_DECOY = ("P1", "Onboarding")   # +1100% on its own, 5% of the change
SUPPORT_MEDIAN_RESOLUTION = 25.18
SUPPORT_OPEN_TICKETS = 2210
SUPPORT_REOPENED = 2688

# The averaging trap, measured directly from the prepared frame. Summing is not
# an approximation of averaging here — it is the opposite sign:
#
#   mean  resolution_hours  32.5769 -> 32.6414   = +0.20%
#   total resolution_hours   225,660 -> 222,712  = -1.31%   <- what summing gives
#
# The engine can only produce the second, so `NON_SUMMABLE` routes the question
# to generated code instead. That routing IS the mitigation, so it is what this
# case asserts.
SUPPORT_MEAN_RESOLUTION_MAR = 32.64


@dataclass
class EvalCase:
    id: str
    question: str
    dataset: Path = FIXTURE
    routing: list[g.Assertion] = field(default_factory=list)
    interpretation: list[g.Assertion] = field(default_factory=list)
    correctness: list[g.Assertion] = field(default_factory=list)
    communication: list[g.Assertion] = field(default_factory=list)
    note: str = ""

    @property
    def assertions(self) -> dict[str, list[g.Assertion]]:
        return {
            "routing": self.routing,
            "interpretation": self.interpretation,
            "correctness": self.correctness,
            "communication": self.communication,
        }


SUITE: list[EvalCase] = [
    EvalCase(
        id="march_drop",
        question="Why did revenue drop in March?",
        note="The canonical case: find the real driver, resist the decoy.",
        routing=[g.route_is("change"), g.is_verified(True)],
        interpretation=[
            g.plan_metric("revenue"),
            g.plan_period("2026-03"),
            g.plan_includes_dimensions("region", "product"),
        ],
        correctness=[
            g.total_change_near(TOTAL_PCT),
            g.top_slice_is(*TOP),
            g.contribution_near(TOP, TOP_PP),
            g.decoy_flagged(*DECOY),
            g.all_verification_passes(),
        ],
        communication=[
            g.narrative_leads_with("Laptop", "West"),
            g.no_invented_numbers(),
            g.no_causal_claim(),
            g.no_summed_contributions(),
        ],
    ),
    EvalCase(
        id="march_drop_paraphrase",
        question="Revenue took a hit in March 2026. What accounts for most of it?",
        note="Same question, different words — interpretation must be robust.",
        interpretation=[g.plan_metric("revenue"), g.plan_period("2026-03")],
        correctness=[g.top_slice_is(*TOP), g.all_verification_passes()],
        communication=[g.narrative_mentions("Laptop", "West"), g.no_invented_numbers()],
    ),
    EvalCase(
        id="units_not_revenue",
        question="How did units sold change in March 2026?",
        note="Metric trap: 'units' must not be silently answered with 'revenue'.",
        interpretation=[g.plan_metric("units"), g.plan_period("2026-03")],
        correctness=[g.all_verification_passes()],
        communication=[g.no_invented_numbers(), g.no_causal_claim()],
    ),
    EvalCase(
        id="latest_period",
        question="What happened to revenue in the most recent month?",
        note="No month named — must default to the latest period in the data.",
        interpretation=[g.plan_metric("revenue"), g.plan_period("2026-05")],
        correctness=[g.all_verification_passes()],
        communication=[g.no_invented_numbers(), g.no_summed_contributions()],
    ),
    EvalCase(
        id="decoy_pressure",
        question=(
            "Which product and region combination had the biggest percentage "
            "collapse in March 2026, and did it actually matter?"
        ),
        note=(
            "Asks directly for the decoy. Naming it is correct here; the test is "
            "whether the narrative also says it explains almost nothing."
        ),
        interpretation=[g.plan_metric("revenue"), g.plan_period("2026-03")],
        correctness=[g.decoy_flagged(*DECOY), g.all_verification_passes()],
        communication=[
            g.narrative_mentions("Docking Station"),
            g.no_invented_numbers(),
            g.no_summed_contributions(),
        ],
    ),
    EvalCase(
        id="channel_question",
        question="Did the Direct or Retail channel drive more of the March 2026 revenue decline?",
        note="Names a dimension explicitly; must include it in the breakdown.",
        interpretation=[g.plan_metric("revenue"), g.plan_includes_dimensions("channel")],
        correctness=[g.total_change_near(TOTAL_PCT), g.all_verification_passes()],
        communication=[g.no_invented_numbers(), g.no_summed_contributions()],
    ),
    # ----------------------------------------------------- generated-code route
    EvalCase(
        id="correlation",
        question="What is the correlation between units and unit_price?",
        note=(
            "The engine cannot express this. Also the case that exposed the "
            "fabrication bug: the model once answered -0.1468 with no code run."
        ),
        routing=[g.route_is("code"), g.is_verified(False)],
        correctness=[
            g.executed_code(),
            g.reports_value(CORRELATION_UNITS_PRICE, tolerance=0.002),
            g.repairs_at_most(2),
        ],
        communication=[g.no_invented_numbers(), g.no_causal_claim()],
    ),
    EvalCase(
        id="top_sales_rep",
        question="Which sales rep had the highest total revenue, and how much?",
        note="A ranking. Claiming a comparison requires printing the ranking.",
        routing=[g.route_is("code")],
        correctness=[
            g.executed_code(),
            g.reports_value(TOP_REP_REVENUE, tolerance=1.0),
            g.repairs_at_most(2),
        ],
        communication=[g.narrative_mentions(TOP_REP), g.no_invented_numbers()],
    ),
    EvalCase(
        id="distinct_products",
        question="How many distinct products are in this dataset?",
        note="Trivially checkable; catches a model answering from the profile alone.",
        routing=[g.route_is("code")],
        correctness=[g.executed_code(), g.reports_value(DISTINCT_PRODUCTS, tolerance=0.01)],
        communication=[g.no_invented_numbers()],
    ),
    EvalCase(
        id="mean_units",
        question="What is the average number of units per order?",
        note="A mean the model could plausibly guess. It must compute it.",
        routing=[g.route_is("code")],
        correctness=[g.executed_code(), g.reports_value(MEAN_UNITS, tolerance=0.05)],
        communication=[g.no_invented_numbers()],
    ),
]


SUITE += [
    # ============================ support tickets ============================
    EvalCase(
        id="support_breach_spike",
        question="Why did SLA breaches go up in March 2026?",
        dataset=SUPPORT,
        note=(
            "The canonical case on the second fixture: a count metric, ISO 8601 "
            "timestamps, and a numeric ticket_id that must not be summed."
        ),
        routing=[g.route_is("change"), g.is_verified(True)],
        interpretation=[
            g.plan_metric("sla_breach"),
            g.plan_period("2026-03"),
            g.plan_includes_dimensions("team", "priority"),
        ],
        correctness=[
            g.total_change_near(SUPPORT_TOTAL_PCT, tolerance=0.5),
            g.top_slice_is(*SUPPORT_TOP),
            g.contribution_near(SUPPORT_TOP, SUPPORT_TOP_PP, tolerance=0.5),
            g.decoy_flagged(*SUPPORT_DECOY),
            g.all_verification_passes(),
        ],
        communication=[
            g.narrative_leads_with("Platform"),
            g.no_invented_numbers(),
            g.no_causal_claim(),
            g.no_summed_contributions(),
        ],
    ),
    EvalCase(
        id="support_decoy_pressure",
        question=(
            "Which team and priority had the sharpest percentage rise in SLA "
            "breaches in March 2026, and how much did it actually matter?"
        ),
        dataset=SUPPORT,
        note=(
            "Onboarding P1 rose 1100% — by far the largest percentage move — but "
            "explains only 5% of the increase.\n\n"
            "Phrased as a ranking ('which had the sharpest rise'), so ROUTE_SYSTEM "
            "sends it to generated code — it lists 'a ranking' among the things "
            "that belong on the `other` route. The change engine would answer it "
            "better, since decoy detection is exactly what it does, so this is a "
            "real routing weakness rather than a quirk of phrasing; see the "
            "README's limitations. The case asserts what the router does, not what "
            "would serve the question best. Decoy detection itself stays covered "
            "by support_breach_spike."
        ),
        routing=[g.route_is("code"), g.is_verified(False)],
        correctness=[g.executed_code()],
        communication=[g.no_invented_numbers(), g.narrative_mentions("Onboarding")],
    ),
    EvalCase(
        id="support_metric_trap",
        question="How did average resolution time change in March 2026?",
        dataset=SUPPORT,
        note=(
            "The averaging trap. The change engine only sums, and on this metric "
            "summing gives -1.31% where the mean moved +0.20% — the opposite sign. "
            "So the correct behaviour is to refuse the engine and generate code. "
            "This case previously asserted change-route behaviour, which "
            "contradicted the NON_SUMMABLE regex added to mitigate exactly this; "
            "it scored 1/5 while the committed eval report still showed 5/5."
        ),
        routing=[g.route_is("code"), g.is_verified(False)],
        correctness=[
            g.executed_code(),
            g.reports_value(SUPPORT_MEAN_RESOLUTION_MAR, tolerance=0.05),
        ],
        communication=[g.no_invented_numbers(), g.no_causal_claim()],
    ),
    EvalCase(
        id="support_median_resolution",
        question="What is the median resolution time in hours across all tickets?",
        dataset=SUPPORT,
        note="A median, which the change engine cannot express — must generate code.",
        routing=[g.route_is("code")],
        correctness=[
            g.executed_code(),
            g.reports_value(SUPPORT_MEDIAN_RESOLUTION, tolerance=0.05),
            g.repairs_at_most(2),
        ],
        communication=[g.no_invented_numbers()],
    ),
    EvalCase(
        id="support_open_tickets",
        question="How many tickets are still open, meaning they have no closed_at value?",
        dataset=SUPPORT,
        note="Null-counting on a genuinely nullable datetime — no equivalent in the sales fixture.",
        routing=[g.route_is("code")],
        correctness=[g.executed_code(), g.reports_value(SUPPORT_OPEN_TICKETS, tolerance=1.0)],
        communication=[g.no_invented_numbers()],
    ),
    EvalCase(
        id="support_reopened_count",
        question="How many tickets were reopened?",
        dataset=SUPPORT,
        note="A text boolean ('Yes'/'No') that has to be filtered, not summed.",
        routing=[g.route_is("code")],
        correctness=[g.executed_code(), g.reports_value(SUPPORT_REOPENED, tolerance=1.0)],
        communication=[g.no_invented_numbers()],
    ),
]


def by_id(case_id: str) -> EvalCase:
    for case in SUITE:
        if case.id == case_id:
            return case
    raise KeyError(f"no eval case {case_id!r}; have {[c.id for c in SUITE]}")
