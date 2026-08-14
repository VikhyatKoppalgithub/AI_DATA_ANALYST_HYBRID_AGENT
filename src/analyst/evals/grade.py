"""Assertions for grading a RoutedAnswer.

Grading is deterministic — no model judges another model. Every assertion is a
plain function over the answer object, so a score is reproducible, free, and
means the same thing next week.

Four families:

  interpretation   did it map the question onto the right columns and period
  routing          did it pick the path that can actually answer the question
  correctness      does the computed result match known ground truth
  communication    does the narrative report that result honestly

The change and code routes produce numbers with different guarantees, so
`communication` is checked differently for each: on the change route every
figure must trace to the verified analysis, and on the code route every figure
must trace to something a sandbox execution actually printed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from analyst.interpret import _analysis_facts
from analyst.session import RoutedAnswer

Assertion = Callable[[RoutedAnswer], tuple[bool, str]]

NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")
# Causal phrasings about the metric itself. "explains little because it is a
# small share" is arithmetic, not causation, so bare "because" is not banned.
CAUSAL = re.compile(
    r"\b(fell|dropped|declined|rose|increased|decreased|change[d]?)\b[^.]{0,40}?"
    r"\b(due to|because of|driven by|caused by|resulted from)\b",
    re.I,
)


@dataclass
class Result:
    name: str
    family: str
    passed: bool
    detail: str


def _numbers(text: str) -> set[float]:
    out = set()
    for token in NUMBER.findall(text):
        try:
            out.add(abs(float(token.replace(",", ""))))
        except ValueError:
            continue
    return out


def _named(fn: Assertion, name: str) -> Assertion:
    fn.__name__ = name  # type: ignore[attr-defined]
    return fn


def _needs_change(answer: RoutedAnswer) -> tuple[bool, str] | None:
    if answer.change is None:
        return False, f"routed to '{answer.route}', so there is no change analysis"
    return None


def _needs_code(answer: RoutedAnswer) -> tuple[bool, str] | None:
    if answer.code is None:
        return False, f"routed to '{answer.route}', so no code was generated"
    return None


# ------------------------------------------------------------------- routing


def route_is(expected: str) -> Assertion:
    def check(answer: RoutedAnswer) -> tuple[bool, str]:
        return answer.route == expected, f"routed to {answer.route!r}"

    return _named(check, f"routed to {expected}")


def is_verified(expected: bool = True) -> Assertion:
    def check(answer: RoutedAnswer) -> tuple[bool, str]:
        return answer.verified is expected, f"verified={answer.verified}"

    return _named(check, f"verified is {expected}")


# ------------------------------------------------------------- interpretation


def plan_metric(expected: str) -> Assertion:
    def check(answer: RoutedAnswer) -> tuple[bool, str]:
        if (bail := _needs_change(answer)) is not None:
            return bail
        got = answer.change.plan.metric
        return got == expected, f"chose {got!r}, expected {expected!r}"

    return _named(check, f"metric is {expected}")


def plan_period(expected: str) -> Assertion:
    def check(answer: RoutedAnswer) -> tuple[bool, str]:
        if (bail := _needs_change(answer)) is not None:
            return bail
        got = answer.change.analysis.period_after
        return got == expected, f"compared into {got!r}, expected {expected!r}"

    return _named(check, f"period is {expected}")


def plan_includes_dimensions(*required: str) -> Assertion:
    def check(answer: RoutedAnswer) -> tuple[bool, str]:
        if (bail := _needs_change(answer)) is not None:
            return bail
        dimensions = answer.change.plan.dimensions
        missing = [d for d in required if d not in dimensions]
        return not missing, f"missing {missing}" if missing else f"has {sorted(dimensions)}"

    return _named(check, f"dimensions include {', '.join(required)}")


# ---------------------------------------------------------------- correctness


def total_change_near(expected_pct: float, tolerance: float = 0.2) -> Assertion:
    def check(answer: RoutedAnswer) -> tuple[bool, str]:
        if (bail := _needs_change(answer)) is not None:
            return bail
        got = answer.change.analysis.pct_change
        return abs(got - expected_pct) <= tolerance, f"{got:+.2f}% vs {expected_pct:+.2f}%"

    return _named(check, f"total change is {expected_pct:+.1f}%")


def top_slice_is(*segments: str) -> Assertion:
    def check(answer: RoutedAnswer) -> tuple[bool, str]:
        if (bail := _needs_change(answer)) is not None:
            return bail
        analysis = answer.change.analysis
        headline = analysis.headline()
        if headline is None:
            return False, "no contributor found"
        top = headline[1]
        return {p.strip() for p in top.segment.split(" x ")} == set(segments), (
            f"top slice was {top.segment!r} ({top.contribution_pp:+.2f} pp)"
        )

    return _named(check, f"top contributor is {' x '.join(segments)}")


def contribution_near(
    segment_parts: tuple[str, ...], expected_pp: float, tolerance: float = 0.1
) -> Assertion:
    def check(answer: RoutedAnswer) -> tuple[bool, str]:
        if (bail := _needs_change(answer)) is not None:
            return bail
        breakdown = answer.change.analysis.intersection
        if not breakdown:
            return False, "no intersection computed"
        for s in breakdown.slices:
            if {p.strip() for p in s.segment.split(" x ")} == set(segment_parts):
                return abs(s.contribution_pp - expected_pp) <= tolerance, (
                    f"{s.contribution_pp:+.2f} pp vs {expected_pp:+.2f} pp"
                )
        return False, f"segment {segment_parts} not found"

    return _named(check, f"{' x '.join(segment_parts)} contributes {expected_pp:+.2f} pp")


def all_verification_passes() -> Assertion:
    def check(answer: RoutedAnswer) -> tuple[bool, str]:
        if (bail := _needs_change(answer)) is not None:
            return bail
        checks = answer.change.analysis.checks
        failed = [c.name for c in checks if not c.passed]
        return not failed, (
            f"{len(checks) - len(failed)}/{len(checks)} passed"
            + (f"; failed: {failed}" if failed else "")
        )

    return _named(check, "all verification checks pass")


def decoy_flagged(*segment_parts: str) -> Assertion:
    def check(answer: RoutedAnswer) -> tuple[bool, str]:
        if (bail := _needs_change(answer)) is not None:
            return bail
        breakdown = answer.change.analysis.intersection
        if not breakdown:
            return False, "no intersection computed"
        for s in breakdown.slices:
            if {p.strip() for p in s.segment.split(" x ")} == set(segment_parts):
                return s.is_decoy, (
                    f"own {s.own_pct_change:+.1f}%, contributes {s.contribution_pp:+.2f} pp"
                )
        return False, f"segment {segment_parts} not found"

    return _named(check, f"{' x '.join(segment_parts)} flagged as a decoy")


# ------------------------------------------------------- correctness (code)


def executed_code(minimum: int = 1) -> Assertion:
    """A code-route answer with no successful execution is a fabrication."""

    def check(answer: RoutedAnswer) -> tuple[bool, str]:
        if (bail := _needs_code(answer)) is not None:
            return bail
        ok = sum(1 for a in answer.code.attempts if a.ok)
        return ok >= minimum, f"{ok} successful of {len(answer.code.attempts)} attempts"

    return _named(check, f"ran at least {minimum} successful code block(s)")


def reports_value(expected: float, tolerance: float = 0.01) -> Assertion:
    """The narrative must contain the true answer, computed independently."""

    def check(answer: RoutedAnswer) -> tuple[bool, str]:
        found = _numbers(answer.narrative)
        hit = [n for n in found if abs(n - abs(expected)) <= tolerance]
        return bool(hit), (
            f"found {sorted(found)[:6]}, expected ~{abs(expected)}"
            if not hit
            else f"reported {hit[0]}"
        )

    return _named(check, f"reports the value {expected}")


def repairs_at_most(limit: int) -> Assertion:
    def check(answer: RoutedAnswer) -> tuple[bool, str]:
        if (bail := _needs_code(answer)) is not None:
            return bail
        return answer.code.repairs <= limit, f"{answer.code.repairs} failed attempt(s)"

    return _named(check, f"at most {limit} repair(s)")


# -------------------------------------------------------------- communication


def narrative_mentions(*terms: str) -> Assertion:
    def check(answer: RoutedAnswer) -> tuple[bool, str]:
        low = answer.narrative.lower()
        missing = [t for t in terms if t.lower() not in low]
        return not missing, f"missing {missing}" if missing else "all present"

    return _named(check, f"narrative mentions {', '.join(terms)}")


def narrative_leads_with(*terms: str) -> Assertion:
    def check(answer: RoutedAnswer) -> tuple[bool, str]:
        first = answer.narrative.split(".")[0].lower()
        missing = [t for t in terms if t.lower() not in first]
        return not missing, f"first sentence: {answer.narrative.split('.')[0][:90]!r}"

    return _named(check, f"leads with {', '.join(terms)}")


def no_invented_numbers() -> Assertion:
    """Every figure in the prose must trace to something actually computed.

    The source differs by route: the verified analysis on one, the sandbox's
    own stdout on the other. Both answer the same question — did this number
    come from a computation, or from the model?
    """

    def check(answer: RoutedAnswer) -> tuple[bool, str]:
        if answer.change is not None:
            source = _analysis_facts(
                answer.change.analysis, answer.change.plan.describe_filters()
            )
            label = "the verified analysis"
        elif answer.code is not None:
            source = "\n".join(
                a.result.to_model_text() for a in answer.code.attempts if a.ok
            )
            label = "executed output"
        else:
            return False, "no computation to trace figures to"

        computed = _numbers(source)
        invented = [n for n in _numbers(answer.narrative) if not any(
            abs(n - c) < 0.051 for c in computed
        )]
        return not invented, (
            f"invented {sorted(invented)} (not in {label})" if invented else "all traceable"
        )

    return _named(check, "no invented numbers")


def no_causal_claim() -> Assertion:
    def check(answer: RoutedAnswer) -> tuple[bool, str]:
        match = CAUSAL.search(answer.narrative)
        return match is None, f"found {match.group(0)!r}" if match else "none"

    return _named(check, "no causal claim about the metric")


def no_summed_contributions() -> Assertion:
    """Adding overlapping partitions is the failure this project exists to avoid."""

    patterns = [
        r"\beach contribut",
        r"\btogether (they )?(account|contribut|explain)",
        r"\bcombined,? (they )?(account|contribut|explain)",
        r"\bsum(med)? to\b",
    ]

    def check(answer: RoutedAnswer) -> tuple[bool, str]:
        for pattern in patterns:
            match = re.search(pattern, answer.narrative, re.I)
            if match:
                return False, f"found {match.group(0)!r}"
        return True, "none"

    return _named(check, "does not sum overlapping segments")


def grade(answer: RoutedAnswer, assertions: dict[str, list[Assertion]]) -> list[Result]:
    results: list[Result] = []
    for family, checks in assertions.items():
        for assertion in checks:
            try:
                passed, detail = assertion(answer)
            except Exception as exc:  # a broken assertion is a failure, not a crash
                passed, detail = False, f"assertion raised {type(exc).__name__}: {exc}"
            results.append(
                Result(
                    name=getattr(assertion, "__name__", "assertion"),
                    family=family,
                    passed=passed,
                    detail=detail,
                )
            )
    return results
