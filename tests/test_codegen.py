"""Codegen loop mechanics, driven by a scripted provider (no model required)."""

from __future__ import annotations

import pandas as pd
import pytest

from analyst.codegen import UNVERIFIABLE, CodegenSession, extract_code, strip_code
from analyst.llm.base import Completion, Message, Provider, ProviderInfo
from analyst.profiler import profile_dataframe
from analyst.sandbox import Kernel


class ScriptedProvider(Provider):
    """Replays queued replies; repeats the last one if the loop keeps going."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.seen: list[list[Message]] = []
        self.info = ProviderInfo(name="scripted", model="stub", local=True)

    def chat(self, *, system, messages, max_tokens=1024, temperature=0.0) -> Completion:
        self.seen.append(list(messages))
        text = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return Completion(text=text, model="stub", output_tokens=len(text.split()))

    def complete(self, *, system, prompt, schema=None, max_tokens=1024, temperature=0.0):
        raise NotImplementedError

    def health(self):
        return True, "stub"


@pytest.fixture
def session_factory(tmp_path):
    made: list[Kernel] = []

    def make(replies: list[str], **kwargs):
        kernel = Kernel(tmp_path, timeout=15)
        kernel.set_bootstrap(
            "import pandas as pd\ndf = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 4, 6]})"
        )
        made.append(kernel)
        profile = profile_dataframe(pd.DataFrame({"a": [1, 2, 3], "b": [2, 4, 6]}))
        return CodegenSession(ScriptedProvider(replies), kernel, profile, **kwargs)

    yield make
    for kernel in made:
        kernel.shutdown()


# ------------------------------------------------------------------ parsing


@pytest.mark.parametrize(
    "text,expected",
    [
        ("```python\nprint(1)\n```", "print(1)"),
        ("```py\nprint(1)\n```", "print(1)"),
        ("```\nprint(1)\n```", "print(1)"),
        ("prose only", None),
        ("```python\n\n```", None),
    ],
)
def test_extract_code(text, expected):
    assert extract_code(text) == expected


def test_strip_code_leaves_prose():
    assert strip_code("before\n```python\nx=1\n```\nafter") == "before\n\nafter"


# ------------------------------------------------------------- the loop


def test_executes_then_answers(session_factory):
    session = session_factory(
        ["```python\nprint(df['a'].sum())\n```", "The sum of column a is 6."]
    )
    run = session.ask("what is the sum of a?")
    assert run.stop_reason == "answered"
    assert len(run.attempts) == 1
    assert run.attempts[0].result.stdout.strip() == "6"
    assert run.answer == "The sum of column a is 6."


def test_repairs_after_a_failure(session_factory):
    session = session_factory(
        [
            "```python\nprint(df['nope'].sum())\n```",
            "```python\nprint(df['a'].sum())\n```",
            "6, after fixing the column name.",
        ]
    )
    run = session.ask("sum a")
    assert run.repairs == 1
    assert len(run.attempts) == 2
    assert run.attempts[1].ok


def test_traceback_is_fed_back_to_the_model(session_factory):
    session = session_factory(
        ["```python\nprint(df['nope'])\n```", "could not compute it"]
    )
    session.ask("sum nope")
    # Not necessarily the *last* message: a failed run leaves nothing verified,
    # so the fabrication nudge follows the traceback.
    fed_back = [
        m.content
        for turn in session.provider.seen
        for m in turn
        if m.role == "user" and "KeyError" in m.content
    ]
    assert fed_back, "traceback was never shown to the model"
    assert "df['nope']" in fed_back[0]


# ------------------------------------------------- the fabrication guard


def test_prose_before_any_execution_is_nudged_not_accepted(session_factory):
    """A figure stated before running code is invented, however confident."""
    session = session_factory(
        [
            "The correlation is -0.1468.",  # fabricated, nothing has run
            "```python\nprint(df['a'].corr(df['b']))\n```",
            "The correlation is 1.0.",
        ]
    )
    run = session.ask("correlation of a and b?")

    assert run.stop_reason == "answered"
    assert run.answer == "The correlation is 1.0."
    assert "-0.1468" not in run.answer
    assert len(run.attempts) == 1

    nudged = any(
        "not run any code" in m.content
        for turn in session.provider.seen
        for m in turn
        if m.role == "user"
    )
    assert nudged


def test_persistent_refusal_to_run_code_yields_no_answer(session_factory):
    session = session_factory(["The answer is 42."])  # never emits a code block
    run = session.ask("what is it?")
    assert run.stop_reason == "no_execution"
    assert run.answer == UNVERIFIABLE
    assert "42" not in run.answer


def test_a_failed_execution_does_not_count_as_verification(session_factory):
    """Only a *successful* run licenses a prose answer."""
    session = session_factory(
        ["```python\nraise ValueError('boom')\n```", "The answer is 7."]
    )
    run = session.ask("what is it?")
    assert run.stop_reason == "no_execution"
    assert "7" not in run.answer


# ------------------------------------------------------------------ limits


def test_max_turns_is_enforced(session_factory):
    """Distinct code each turn, or the stuck-loop guard fires first."""
    session = session_factory(
        [f"```python\nx = {i}\n```" for i in range(1, 9)], max_turns=3
    )
    run = session.ask("loop")
    assert run.stop_reason == "max_turns"
    assert len(run.attempts) == 3


def test_identical_failing_code_aborts_instead_of_burning_turns(session_factory):
    """A model re-sending the same broken block is stuck, not repairing."""
    session = session_factory(["```python\nraise ValueError('boom')\n```"], max_turns=8)
    run = session.ask("go")

    assert run.stop_reason == "stuck"
    assert len(run.attempts) == 3  # first submission, then MAX_REPEATS identical ones
    assert "same failing code" in run.answer
    assert "ValueError" in run.answer


def test_repeated_code_that_starts_producing_output_is_not_aborted(session_factory):
    """Only silent repeats are a stuck loop; a repeat that prints is progress."""
    session = session_factory(
        ["```python\nprint(df['a'].sum())\n```"] * 3 + ["The sum is 6."], max_turns=8
    )
    run = session.ask("sum a")
    assert run.stop_reason == "answered"
    assert run.informative_attempts >= 1


def test_repeated_failures_stop_the_loop(session_factory):
    session = session_factory(
        ["```python\nraise ValueError('boom')\n```"], max_consecutive_failures=2, max_turns=5
    )
    run = session.ask("break it")
    assert run.repairs >= 2
    nudged = any(
        "Stop debugging" in m.content
        for turn in session.provider.seen
        for m in turn
        if m.role == "user"
    )
    assert nudged


def test_charts_are_collected(session_factory):
    session = session_factory(
        ["```python\nopen('c.png','w').write('x')\n```", "chart written to c.png"]
    )
    run = session.ask("chart it")
    assert run.charts == ["c.png"]


def test_codegen_results_are_never_marked_verified(session_factory):
    """Generated-code numbers carry no reconciliation, unlike the engine's."""
    session = session_factory(["```python\nprint(1)\n```", "It is 1."])
    assert session.ask("what?").verified is False
