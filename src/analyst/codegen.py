"""Sandboxed code generation, for questions the deterministic engine cannot express.

The engine answers "why did X change" exactly. It cannot answer "what is the
correlation between units and discount" — that needs arbitrary code. This loop
writes it, runs it in the sandbox, and iterates on real tracebacks.

The model signals intent with a fenced ```python block rather than a native
tool call. Tool-calling support varies across local models and providers; a code
fence works with every one of them, which is the point of a free-first design.

Numbers here come from executed output rather than from the deterministic
engine, so they cannot be reconciled the same way. That difference is carried
through to the caller as `verified=False` — the UI labels it, rather than
presenting both paths as equally trustworthy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from analyst.llm.base import Completion, Message, Provider
from analyst.profiler import DatasetProfile
from analyst.sandbox import ExecResult, Kernel

CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
MAX_TURNS = 6
MAX_CONSECUTIVE_FAILURES = 3

CODEGEN_SYSTEM = """\
You answer questions about a dataset by writing Python and reading its output.

A persistent Python process holds `df`, already loaded and cleaned, with \
`pandas as pd`, `numpy as np`, `duckdb`, and `matplotlib.pyplot as plt` \
available. Variables persist between your messages, so build on earlier steps.

To run code, write exactly one fenced block:

```python
print(df['units'].corr(df['unit_price']))
```

You will be shown its stdout, the value of a trailing expression, and the full \
traceback if it raises. Anything you do not print, you do not see.

Rules:
- One code block per message. Keep each step small enough to check.
- Print the numbers you intend to use. Never state a figure you have not seen \
in output.
- The same applies to comparisons and rankings. "The highest" needs the ranking \
printed; "much larger than the next" needs the next one printed too. If you did \
not compute it, do not claim it.
- Do not derive new figures from printed ones. A percentage, ratio, difference, \
or average worked out in your head is a guess, even when the inputs are real. \
Compute it in code and print it.
- If code raises, read the traceback and fix the actual cause. Do not repeat \
the same call unchanged.
- Charts work. Save with `fig.savefig('name.png', dpi=110, bbox_inches='tight')` \
and say the filename. Never claim plotting is unavailable, and never describe a \
chart you did not save — a described chart is a chart nobody can check.
- When asked to quantify a relationship, group means are not a quantification. \
Compute a statistic: a correlation, a slope, an effect size. Then judge it \
against the spread of the data. A gap of 0.03 between group means, where the \
values themselves have a standard deviation of 1.0, is noise — reporting it as \
a trend is wrong even though the means are correct. Say plainly when a \
relationship is negligible; "no meaningful relationship (r = -0.003)" is a real \
answer and often the true one.
- Check whether a pattern actually holds before naming it. Group means that rise, \
fall, and rise again are not a trend, and a bin holding 54 of 250,000 rows \
carries no weight.

Answer every part of the question. "How did revenue and profit change \
month-over-month and year-over-year, and were there unusual periods?" is five \
things to compute, not one. Work out what the question is asking for before you \
start, and make sure your final reply covers each part with its own figure.

Never describe a pattern you have not computed. Printing a column of monthly \
totals and forming an impression of it is not computing a trend — "rose \
steadily", "increased consistently", "spiked in December" are all claims that \
need a calculation behind them. If you want to say revenue grew, compute the \
change and quote it. If you want to name an unusual period, rank the periods and \
quote the outlier. An impression of a table is a guess wearing the clothes of an \
answer.

`df` is a working copy inside a sandbox. Nothing you do touches the person's \
file, and no change you make is saved anywhere. So never report a fix as done: \
write "these can be imputed with the mode, 'Debit Card'", never "these have been \
imputed". Describing a remedy is answering the question; claiming to have \
applied it is a false statement about their data.

For the same reason, do not modify `df` in place. Your variables persist into \
the next question, so an imputation or a dropped column silently changes what \
every later answer is computed from. Work on a copy — `clean = df.copy()` — and \
leave `df` as loaded.

Answer the question, do not narrate how you got there. No "Step 1", "Step 2", \
no markdown headings, no "the output shows that". The person asked a question, \
not for a transcript of your exploration. If you ran five blocks to find one \
number, report the number.

When you have the answer, reply with prose and NO code block. That is how you \
finish. Give the figures, not just the direction: "-9.6% between the first and \
last month" says something, "revenue declined" does not. This is observational \
data: describe what the numbers show, not what caused it.
"""

FINISH_NUDGE = (
    "You have run {n} steps. Stop running code and reply now with your answer in "
    "prose, using only figures you have already printed. If something could not "
    "be computed, say so plainly."
)

REPAIR_EXHAUSTED = (
    "The last {n} code blocks all failed. Stop debugging and reply in prose with "
    "whatever you established before the failures, stating what you could not "
    "compute and why."
)

# A prose reply before anything has run is a fabrication, not an answer. Asked
# for the correlation between two columns, a 14B model replied "-0.1468" on the
# first turn with an invented interpretation attached; the real value was
# -0.0033. The loop must not accept a figure that no execution produced.
NO_CODE_NUDGE = (
    "You have not run any code yet, so you cannot know the answer. Do not state "
    "figures from memory or estimate them. Reply with a single ```python block "
    "that computes and prints what the question asks for."
)

UNVERIFIABLE = (
    "No answer: the model replied without running any code, so every figure it "
    "offered would have been invented. Rephrase the question, or ask something "
    "the verified change-analysis path can answer."
)

REPEATED_CODE = (
    "That is character-for-character the code you already ran, and it behaved "
    "identically. Re-running it cannot produce a different result. Write "
    "something different: fix the specific line the traceback names, or take a "
    "simpler route to the same figure."
)

STUCK = (
    "No answer: the model submitted the same failing code repeatedly and could "
    "not make progress. The last error was:\n{error}"
)

# Two identical submissions is a stuck loop, not a repair attempt. Spending the
# remaining turns on it only delays an answer that will not arrive — a 14B model
# observed re-sending the same broken block four times against the same
# traceback, unmoved by being told so.
MAX_REPEATS = 2

# A figure built with plt/pandas plotting exists only in memory until savefig is
# called. The model reliably forgets, then describes the chart it "made" — the
# same fabrication as describing an unrun computation, so it is caught in code
# rather than asked for in the prompt.
PLOTTING_CALL = re.compile(r"\b(plt\.|\.plot\(|sns\.|pyplot|subplots\()", re.I)
CHART_MENTION = re.compile(r"\b(chart|graph|plot|figure|visuali[sz])", re.I)
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".svg")

UNSAVED_FIGURE = (
    "You built a figure but never saved it, so no image file exists and the "
    "person will see nothing. Add "
    "fig.savefig('name.png', dpi=110, bbox_inches='tight') and run it again."
)

CHART_CLAIMED_NOT_SAVED = (
    "Your answer refers to a chart, but no image file was written during this "
    "session. Either save the chart with "
    "fig.savefig('name.png', dpi=110, bbox_inches='tight'), or rewrite the "
    "answer without referring to a chart that does not exist."
)

SILENT_CODE_NUDGE = (
    "Nothing you have run has printed anything yet, so you still have no figures "
    "to answer with. Do not answer from general knowledge or typical patterns. "
    "Run one small block that print()s the specific numbers the question asks for."
)

MAX_NUDGES = 2


@dataclass
class Attempt:
    code: str
    result: ExecResult

    @property
    def ok(self) -> bool:
        return self.result.ok


@dataclass
class CodegenRun:
    question: str
    answer: str
    attempts: list[Attempt] = field(default_factory=list)
    charts: list[str] = field(default_factory=list)
    completions: list[Completion] = field(default_factory=list)
    stop_reason: str = "answered"

    verified: bool = False  # numbers come from generated code, not the checked engine

    @property
    def repairs(self) -> int:
        return sum(1 for a in self.attempts if not a.ok)

    @property
    def informative_attempts(self) -> int:
        """Runs that both succeeded and surfaced something."""
        return sum(1 for a in self.attempts if a.ok and a.result.produced_output)

    @property
    def seconds(self) -> float:
        return sum(c.duration_s for c in self.completions)

    @property
    def output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.completions)


def extract_code(text: str) -> str | None:
    """First fenced Python block, or None when the model is answering in prose."""
    match = CODE_FENCE.search(text)
    if not match:
        return None
    code = match.group(1).strip()
    return code or None


def strip_code(text: str) -> str:
    return CODE_FENCE.sub("", text).strip()


class CodegenSession:
    def __init__(
        self,
        provider: Provider,
        kernel: Kernel,
        profile: DatasetProfile,
        *,
        preparation_log: list[str] | None = None,
        max_turns: int = MAX_TURNS,
        max_consecutive_failures: int = MAX_CONSECUTIVE_FAILURES,
    ) -> None:
        self.provider = provider
        self.kernel = kernel
        self.profile = profile
        self.max_turns = max_turns
        self.max_consecutive_failures = max_consecutive_failures

        sections = [CODEGEN_SYSTEM, f"## Dataset profile\n\n{profile.to_prompt()}"]
        if preparation_log:
            # `df` is the cleaned frame, so problems already fixed are invisible
            # in it. Asked what was wrong with a file, a model inspected the
            # cleaned copy and reported two issues while a third — 350 duplicate
            # rows — had been silently removed before it ever looked.
            sections.append(
                "## Already applied to `df` before you saw it\n\n"
                + "\n".join(f"- {a}" for a in preparation_log)
                + "\n\nThese are facts about the source data. `df` no longer "
                "shows them, so you cannot rediscover them by inspecting it — "
                "but they are real, and a question about data quality must "
                "include them."
            )
        self.system = "\n\n".join(sections)

    def ask(self, question: str) -> CodegenRun:
        run = CodegenRun(question=question, answer="")
        messages = [Message(role="user", content=question)]
        consecutive_failures = 0
        nudges = 0

        for turn in range(1, self.max_turns + 1):
            completion = self.provider.chat(
                system=self.system, messages=messages, max_tokens=1200, temperature=0.1
            )
            run.completions.append(completion)
            messages.append(Message(role="assistant", content=completion.text))

            code = extract_code(completion.text)
            if code is None:
                if run.informative_attempts:
                    text = completion.text.strip()
                    if (
                        CHART_MENTION.search(text)
                        and not run.charts
                        and nudges < MAX_NUDGES
                    ):
                        nudges += 1
                        messages.append(
                            Message(role="user", content=CHART_CLAIMED_NOT_SAVED)
                        )
                        continue
                    run.answer = text
                    run.stop_reason = "answered"
                    break
                # Nothing has run, so any figure in this reply is invented.
                if nudges < MAX_NUDGES:
                    nudges += 1
                    nudge = (
                        SILENT_CODE_NUDGE
                        if any(a.ok for a in run.attempts)
                        else NO_CODE_NUDGE
                    )
                    messages.append(Message(role="user", content=nudge))
                    continue
                run.answer = UNVERIFIABLE
                run.stop_reason = "no_execution"
                break

            repeats = sum(1 for a in run.attempts if a.code.strip() == code.strip())
            result = self.kernel.execute(code)
            run.attempts.append(Attempt(code=code, result=result))

            if repeats >= MAX_REPEATS and not result.produced_output:
                last = (result.error or {}).get("traceback", "no output produced")
                run.answer = STUCK.format(error=last.strip()[:400])
                run.stop_reason = "stuck"
                break
            run.charts += [
                f for f in result.new_files if f.lower().endswith((".png", ".jpg", ".svg"))
            ]

            consecutive_failures = 0 if result.ok else consecutive_failures + 1

            feedback = result.to_model_text()
            if PLOTTING_CALL.search(code) and not any(
                f.lower().endswith(IMAGE_SUFFIXES) for f in result.new_files
            ):
                feedback += "\n\n" + UNSAVED_FIGURE
            if repeats:
                feedback += "\n\n" + REPEATED_CODE
            if consecutive_failures >= self.max_consecutive_failures:
                feedback += "\n\n" + REPAIR_EXHAUSTED.format(n=consecutive_failures)
                run.stop_reason = "repair_exhausted"
                consecutive_failures = 0
            elif turn == self.max_turns - 1:
                feedback += "\n\n" + FINISH_NUDGE.format(n=turn)

            messages.append(Message(role="user", content=feedback))
        else:
            # Loop ran to the cap without the model dropping the code fence.
            run.stop_reason = "max_turns"
            run.answer = strip_code(messages[-2].content) or (
                "Ran out of steps before reaching an answer."
            )

        run.charts = sorted(set(run.charts))
        return run
