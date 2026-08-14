# AI Data Analyst

Upload a messy spreadsheet, ask a business question, get an answer whose every
number is computed and checked rather than generated.

Runs entirely on a local model. No API key, no per-query cost.

```
Q: Why did revenue drop in March?

Laptop x West was the largest contributor, accounting for -11.08 percentage
points of the -16.1% total change. This segment's own change was -36.7%, and it
was 30.2% of the starting total. Docking Station x South moved -61.4% on its
own, but explains only -0.66 pp of the total because it was just 1.1% of revenue.

  PASS  'region' contributions reconcile to the total change    sums to -465,306.31
  PASS  'product x region' contributions reconcile              residual -0.00
  PASS  Period totals match a direct recomputation              drift 0.0000
  PASS  Change is concentrated enough to attribute              top contributor explains 94%

route=change, 3 local model calls, 188 output tokens, 32.0s, $0.00 — verified
```

---

## Quick start

Requires Python 3.11+ and [Ollama](https://ollama.com).

```bash
git clone <this repo> && cd ai-data-analyst
pip install -e ".[dev]"

brew services start ollama          # or: ollama serve
ollama pull qwen2.5-coder:14b       # ~9GB

python scripts/make_demo_data.py        # sales fixture
python scripts/make_support_data.py     # support fixture
pytest                                  # 109 tests, no model needed
```

Then either interface:

```bash
streamlit run app.py

python -m analyst.cli data/sales_2026.csv "Why did revenue drop in March?"
python -m analyst.cli data/sales_2026.csv --profile-only    # no model needed
python -m analyst.cli data/sales_2026.csv                   # interactive
```

Smaller machine? `--model qwen2.5-coder:1.5b` runs in ~1GB and about 5x faster,
at a real cost in quality — quantified in [Results](#results).

---

## Architecture

```
                        ┌──────────────┐
  upload ──▶ profile ──▶│    router    │
                        └──────┬───────┘
                   ┌───────────┴───────────┐
                   ▼                       ▼
        ┌────────────────────┐   ┌────────────────────┐
        │  change analysis   │   │  generated code    │
        │  deterministic     │   │  sandboxed Python  │
        │  reconciled        │   │  iterates on       │
        │  ✅ verified       │   │  tracebacks        │
        └─────────┬──────────┘   │  ⚠️  unverified    │
                  │              └─────────┬──────────┘
                  └───────────┬────────────┘
                              ▼
                       narration (model)
```

**Profile, don't dump.** The raw rows never reach the model. The profiler
renders a 37,000-row file into roughly 580 tokens, carrying what is needed to
write correct code first time: real dtypes, the *semantic* type where it
disagrees (dates and currency stored as text), null and duplicate counts,
cardinality, and sample values. It also flags traps — an `int64` `ticket_id` is
reported as *"numeric but reads as an identifier — do not aggregate"*.

**The model does two jobs, neither of them arithmetic.** It reads a question and
picks columns; later it reads finished numbers and writes prose. Column names it
returns are validated against the profile and replaced with a fallback if
hallucinated, rather than failing deep inside pandas.

**Two answering paths with different guarantees**, surfaced rather than blended:

| route | numbers come from | verified |
|---|---|---|
| `change` | the deterministic contribution engine, reconciled | ✅ yes |
| `code` | Python the model wrote, run in a sandbox | ⚠️ no |

Everything the code path produces is labelled unverified in both interfaces, and
the code that produced each figure is shown next to it.

---

## Why hybrid

### The model cannot do the arithmetic

Asked to split a 16% decline between two segments, `qwen2.5-coder:14b` answered
**13pp and 3pp**. The true values are **15.4pp and 0.61pp** — wrong by 5x on the
smaller one, and neatly summing to 16. It back-fits totals.

That is not a prompt problem. It is what the deterministic engine exists to
prevent: contribution arithmetic is exact, reproducible, free to run, and
testable against a fixture with a known answer. None of that is true of asking a
model to add things up.

### But a deterministic engine cannot answer everything

The engine answers *"how did X change between two periods, and what accounts for
it"* exactly. It cannot express a correlation, a median, a null count, or a
ranking. Those need arbitrary code — so the second path generates it, runs it in
a sandbox, and iterates on real tracebacks.

Refusing those questions would be a worse product. Answering them with the same
confidence as verified arithmetic would be dishonest. Labelling them is the
third option.

### Contribution, not the biggest percentage

"Why did revenue drop?" is a decomposition, not a query. For each dimension the
engine computes how many percentage points of the *total* change each slice
explains:

```
contribution_pp = 100 * (slice_after - slice_before) / total_before
```

Those contributions must sum to the total change, which makes them checkable: if
they do not sum, the grouping is dropping rows.

The failure this guards against is reporting the largest *percentage* move
instead. Both fixtures contain a planted decoy for exactly this — a slice that
moves dramatically and explains almost nothing:

| fixture | real answer | decoy |
|---|---|---|
| sales | `Laptop x West` −11.08pp (69% of the move) | `Docking Station x South` **−61.4%** own change, −0.66pp |
| support | `P1 x Platform` +44.13pp (91% of the move) | `P1 x Onboarding` **+1100%** own change, +2.58pp |

### The model does not get to answer without computing

Asked for a correlation, the model replied on its first turn:

> The correlation between units and unit_price is **-0.1468**. This indicates a
> weak negative relationship; as units sold increases, unit price tends to
> decrease slightly.

No code had run. The true value is **-0.0033** — no relationship at all. The
figure was invented and an interpretation built on top of it.

The code loop now refuses a prose answer until at least one execution has
*succeeded*. A model that answers without computing is told to run code; if it
persists, the answer is discarded rather than shown. A *failed* execution does
not count.

---

## Evaluation

**16 cases across 2 datasets, graded by 102 deterministic assertions.** No model
judges another model, so a score is reproducible and costs nothing to produce.

```bash
python -m analyst.evals.runner --models qwen2.5-coder:14b,qwen2.5-coder:1.5b --repeat 3
```

Assertions fall into four families, which is more informative than one number —
they separate *reading the question* from *deciding how to answer it* from
*reporting honestly*:

| family | asks |
|---|---|
| routing | did it pick the path that can actually answer this? |
| interpretation | did it map the question onto the right columns and period? |
| correctness | does the computed result match independently known ground truth? |
| communication | does the narrative report that result honestly? |

Two assertions carry most of the weight:

- **`no invented numbers`** — every figure in the prose must trace to something
  computed. The source differs by route: the verified analysis on one, the
  sandbox's own stdout on the other. Same question either way.
- **`no summed contributions`** — catches the model presenting three overlapping
  partitions of the same money as separate additive causes.

Ground truth comes from the generator scripts, which print the shock they
created. Grading a pipeline against its own output measures self-consistency,
not correctness.

The graders are themselves tested, each in both directions. A grader that cannot
fail is worse than no grader.

### The fixtures

Two datasets that differ on every axis that matters, so the pipeline is not
validated against one shape of problem:

| | `sales_2026.csv` | `support_tickets_2026.csv` |
|---|---|---|
| domain | retail sales | support operations |
| rows | 9,596 | 37,166 |
| metric | currency (`revenue`) | a count (`sla_breach`) |
| direction | **falls** −16.1% | **rises** +48.4% |
| dates | `MM/DD/YYYY` | ISO 8601 with timezone |
| mess | `$1,234.56`, `west`/`West`, `-`/`unknown` | `NULL`/`n/a`/`TBD`, `PLATFORM`/`Platform` |
| traps | numeric-looking IDs, an all-null column | `int64` ticket ID, a nullable datetime, text booleans |

Both are synthetic. That is a limitation (see below) and also the only way to
have ground truth at all.

### Data shapes handled

Added after a real government-statistics export failed on all three counts:

| shape | example | how |
|---|---|---|
| wide, dated | the two fixtures | the default path |
| **annual** | `Year` = `2025`, no date column | a `year` semantic type; `to_datetime(2025)` silently yields 1970 + 2025ns, so year columns bypass it |
| **long / tidy** | one `Value` column, 41 measures in `Variable_name`, 3 units | plans carry `filters` pinning one measure and one unit |
| **hierarchical** | `Level 1` totals containing `Level 4` detail | the profiler flags nested levels and the planner pins one |

---

## Results

Three full passes of the suite per model, on an M4 MacBook Air. 96 runs, 26
minutes, $0.00.

| model | score | routing | interpretation | correctness | communication | median s |
|---|---|---|---|---|---|---|
| `qwen2.5-coder:14b` | **299/306 (98%)** | 100% | 100% | 100% | 93% | 25s |
| `qwen2.5-coder:1.5b` | 266/306 (87%) | 85% | 100% | 87% | 80% | 5s |

### Variance across the three runs

| model | mean | sd | worst | best | unstable cases |
|---|---|---|---|---|---|
| `qwen2.5-coder:14b` | 97.7% | 0.6 pp | 97.1% | 98.0% | `channel_question` |
| `qwen2.5-coder:1.5b` | 86.9% | 1.5 pp | 85.3% | 88.2% | `latest_period`, `mean_units`, `support_breach_spike` |

The 14B is stable enough that a single run is close to its true score. The 1.5b
is not: one case in five moves between runs, and earlier single passes put it
anywhere from 85% to 98%. Reporting one run as *the* number would have been
wrong by up to 11 points.

### Reading the families

**Interpretation is 100% on both models.** Mapping a question onto the right
metric, period, and dimensions is the easy part, and model size does not help.
Everything that separates the two models is elsewhere.

**Correctness is 100% on the 14B** across both fixtures — every computed figure
matches independently derived ground truth, which is what the deterministic
engine is for. On the 1.5b it drops to 87%, but almost entirely as a *consequence
of routing*: send a ranking question to the change engine and there is no
ranking to be correct about.

**Routing is where size shows.** The 1.5b scores 85%, and its single worst case
is total: `top_sales_rep` scored **1/6 in all three runs**, because it routed
"which sales rep had the highest total revenue?" to change analysis instead of
generated code. Five downstream assertions then failed for lack of anything to
check.

**Communication is the 14B's only real weakness (93%)**, and the failures are the
same shape every time:

> There are **2,210** tickets still open, meaning they have no `closed_at` value.
> This represents approximately **6.04%** of the total tickets.

The count is computed and correct. The percentage is not — it was derived in the
model's head from the row count. It is even arithmetically right here, which is
precisely why the rule exists: the same habit produced *13pp and 3pp* against a
true *15.4 and 0.61*. Six of the 14B's seven failures are this, on the two
counting cases, in every run. An explicit prompt rule against deriving figures
did not stop it.

That is a finding, not a grader artifact. The honest summary is that the
fabrication guards catch answers with *no* computation behind them, and do not
stop a model from decorating a real number with an underived one.

---

## What building it found

Every item here was found by running the thing, not by reading the code.

**A reviewer that could not fail.** An earlier deterministic engine shipped four
"verification checks", three of which were true by construction — one compared
`monthly.sum()` against `valid[metric].sum()` where `monthly` *was* a groupby-sum
of `valid`. It showed all-green while the region breakdown was silently wrong.

**Silent segment fragmentation.** `West`, `west`, and `" West "` grouped as three
segments, understating West by $29,223. The reconciliation check that replaced
the tautology catches it immediately: contributions summed to −467,513 against a
total of −465,306.

**Index-aligned subtraction drops segments.** `after - before` on two groupbys
silently loses any segment present in only one period. The engine reindexes on
the union, so a segment that appears or disappears shows up at zero.

**An entire class of bug that only appears when a metric rises.** Slices are
stored ascending by change, so `slices[0]` is the top contributor when a metric
falls — and the largest *counter*-movement when it rises. On the support fixture
the narrator was about to report `P2 x Platform` at **−1.41pp** as the answer to
a **+48%** increase. The sales fixture could never have caught this, because it
only ever falls. Ranking is now direction-aware everywhere.

**A threshold that did not survive a second dataset.** The decoy test used an
absolute cutoff (`contribution < 1.0pp`). On a −16% total that is a sixteenth of
the move; on a +48% total it is a fiftieth. The support fixture's decoy —
explaining 5% of the change — sailed past it. It is now measured as a share of
the total change.

**Fabrication is qualitative as often as numeric.** Three separate cases: a
correlation stated before any code ran; a "significantly higher than the next
rep" claim when only `.head(1)` had been computed; a percentage derived in the
model's head from a real count. All three needed different guards.

**Fact *presentation* moves the answer as much as prompt wording.** Handed a flat
ranked list mixing dimensions, the model wrote that three segments "each
contributed over 10 percentage points" — silently adding overlapping partitions.
Same model, same instructions; leading with the intersection and explicitly
fencing off the alternative slicings fixed it.

---

## Limitations

**The change engine only sums.** Ask *"how did average resolution time change?"*
and it reports the change in the **total**. On the support fixture that is
**−1.3%** where the true change in the mean is **+0.2%** — the opposite sign.
The router sends the question to the engine because it is a change-over-time
question, and the metric is chosen correctly; the aggregation is simply wrong for
the question. **The eval suite does not catch this**: `support_metric_trap`
asserts only that `resolution_hours` was selected. The fix is to route
mean/median/rate questions to generated code, or to teach the engine
mean-difference decomposition.

**The code path is unverified by construction.** Its figures come from code the
model wrote. The guards catch answers with no computation behind them; they
cannot catch code that runs cleanly and computes the wrong thing.

**Derived figures slip past the guards.** The model reliably computes the number
it was asked for, then decorates it with one it worked out in its head — a
percentage of a row count, most often. Six of the 14B's seven eval failures are
this, reproducible in every run, and an explicit prompt rule against it did not
help. The guards enforce *that a computation happened*, not that every figure in
the sentence came from one.

**The sandbox is isolation, not a security boundary.** Separate process,
resource limits, a dedicated result fd, and neutralised sockets — but code
reaching for `ctypes` or spawning a subprocess can get around the in-process
guards. A container with no network interface is the real boundary; the kernel is
structured so that backend drops in behind the same interface.

**Routing is a single classification call.** A wrong route degrades everything
downstream, and the 1.5b model gets it wrong on questions the 14B handles.

**Both fixtures are synthetic.** They contain the mess I thought to add. Real
exports contain mess I did not — the first real file tried (New Zealand's Annual
Enterprise Survey) failed three separate ways, all since fixed and covered by
tests, but the lesson generalises: shapes I have not seen will keep appearing.

**Contribution analysis assumes two periods and a summable metric.** No trend
decomposition, no seasonality, no mix-vs-rate split, no multi-file joins.

**Hierarchy handling is a heuristic, not a guarantee.** Nested rows — where
"Level 1" totals already contain the "Level 4" detail — are detected by looking
at category *values*, and the planner is told to pin one level. A dataset that
nests without saying so will still be summed across its own subtotals. This is
the one failure mode observed that produces a *confidently wrong number rather
than a refusal*: the first attempt on that survey reported 3,540,602 where the
true figure was 976,077, and every verification check passed, because the
arithmetic was internally consistent with the rows it was given.

**Local inference is slow.** 20–35 seconds per question on an M4. Fine for
analysis, wrong for anything interactive.

**Nothing here establishes causation.** The engine identifies what *accounts for*
a change. The narrator is instructed to say so and the graders check the
language, but observational data cannot support more.

---

## Project layout

```
src/analyst/
  profiler.py          semantic column profiling — what the model sees
  prepare.py           type coercion and dedup, every action reported
  analysis/
    change.py          contribution decomposition + verification checks
    normalize.py       categorical folding before grouping
  llm/
    base.py            provider interface, usage ledger
    ollama.py          local provider (num_ctx, JSON-schema decoding)
  sandbox/
    kernel.py          parent-side process handle, timeouts, bootstrap replay
    worker.py          subprocess: persistent namespace, fd-level capture
  codegen.py           the code-fence loop and its fabrication guards
  interpret.py         question -> columns, and results -> prose
  session.py           routing, and the two paths' differing guarantees
  evals/               cases, deterministic graders, runner
app.py                 Streamlit UI
scripts/               fixture generators, each printing its ground truth
tests/                 109 tests, no model required
```

Tests need no model or API key. The eval suite needs Ollama.
