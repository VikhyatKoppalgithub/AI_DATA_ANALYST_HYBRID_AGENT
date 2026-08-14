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
pytest                                  # 122 tests, no model needed
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

Added after two real government exports failed — NZ's Annual Enterprise Survey on
the first three, NYC's Citywide Payroll on the last two:

| shape | example | how |
|---|---|---|
| wide, dated | the two fixtures | the default path |
| **annual** | `Year` = `2025`, no date column | a `year` semantic type; `to_datetime(2025)` silently yields 1970 + 2025ns, so year columns bypass it |
| **long / tidy** | one `Value` column, 41 measures in `Variable_name`, 3 units | plans carry `filters` pinning one measure and one unit |
| **hierarchical** | `Level 1` totals containing `Level 4` detail | the profiler flags nested levels and the planner pins one |
| **per-entity columns** | `last_name` — 125,431 distinct, 11% of rows, profiles as categorical | a dimension is dropped above 1,000 distinct *and* 5% of rows, so 1,616 job titles survive and surnames do not |
| **competing time axes** | `fiscal_year` (2024–25) beside `agency_start_date` (hire dates from 1901) | every candidate axis is described to the planner with its own range, not just the first |

---

## Results

Three full passes of the suite per model, on an M4 MacBook Air. 96 runs, 53
minutes, $0.00.

| model | score | routing | interpretation | correctness | communication | median s |
|---|---|---|---|---|---|---|
| `qwen2.5-coder:14b` | **301/306 (98%)** | 100% | 100% | 98% | 97% | 25s |
| `qwen2.5-coder:1.5b` | 245/306 (80%) | 75% | 96% | 73% | 83% | 9s |

### Variance across the three runs

| model | mean | sd | worst | best | unstable cases |
|---|---|---|---|---|---|
| `qwen2.5-coder:14b` | 98.4% | 0.6 pp | 98.0% | 99.0% | `support_metric_trap` |
| `qwen2.5-coder:1.5b` | 80.1% | 0.6 pp | 79.4% | 80.4% | `march_drop`, `decoy_pressure`, `support_decoy_pressure`, `support_median_resolution` |

The 14B is stable enough that a single run is close to its true score. The 1.5b
is not: a quarter of its cases move between runs, and earlier single passes put
it anywhere from 85% to 98%. Reporting one run as *the* number would have been
wrong by up to 11 points.

> **On an earlier published number.** This table read 299/306 and 266/306 for a
> while, and neither reproduced. Three eval cases were asserting behaviour the
> code had deliberately stopped having — `support_metric_trap` and
> `support_decoy_pressure` both asserted the change route while the router
> correctly sends them to generated code — so the old figures measured a version
> that no longer existed. The 14B is unchanged at 98%; the 1.5b's real score is
> 80%, not 87%, because those cases now demand the code route and routing is
> exactly where a 1.5B model is weakest. Re-running your own headline number is
> the only way to find this, and it is worth doing before anyone else does.

### Reading the families

**Interpretation is near-perfect on both** — 100% on the 14B, 96% on the 1.5b.
Mapping a question onto the right metric, period, and dimensions is the easy
part, and model size barely helps. Everything separating the two models is
elsewhere.

**The 14B's five remaining failures are both on the code route**, and they are
specific:

```
3x  support_decoy_pressure   narrative mentions Onboarding
2x  support_metric_trap      reports the value 32.64
```

The first is the routing weakness measuring itself. That question — *which team
and priority rose most, and did it matter?* — is a contribution question, and
the change engine would name `Onboarding` from its decoy detection. Routed to
generated code instead, the 14B fails to identify it in **all three runs**. The
case asserts current behaviour, and current behaviour is worse than the engine's
would be. That is the argument for fixing the router, stated in data.

**Routing is where size shows.** The 1.5b scores 75%, and its worst case is
total: `top_sales_rep` scored **1/6 in all three runs**, routing "which sales rep
had the highest total revenue?" to change analysis instead of generated code.
Five downstream assertions then failed for lack of anything to check. Its
correctness figure (73%) is mostly this same effect one step downstream — send a
ranking to the change engine and there is no ranking to be correct about.

**The 14B's communication weakness did not reproduce.** Earlier passes put it at
93%, with six of seven failures being a real count decorated with a percentage
worked out in the model's head:

> There are **2,210** tickets still open, meaning they have no `closed_at` value.
> This represents approximately **6.04%** of the total tickets.

In this run both counting cases scored 4/4 across all three passes and
communication came in at 97%. The habit is real — it is the same one that
produced *13pp and 3pp* against a true *15.4 and 0.61* — but it is evidently not
as stable as "every run" implied, which is its own argument for repeating the
measurement rather than quoting it once. The guard analysis stands regardless:
fabrication guards catch answers with *no* computation behind them, and do not
stop a model decorating a real number with an underived one.

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

**A net change can be the residue of movement in both directions.** NYC's payroll
export (1.1M rows, FY2024 vs FY2025) reported overtime pay down **−4.47%**, led by
the Police Department. True, and materially incomplete:

```
gross down  -8.75 pp   (Police -5.63, HRA -1.16, Fire -1.02, …)
gross up    +4.27 pp   (Correction +2.64, Sanitation +1.29, …)
net         -4.47 pp
```

Correction's overtime *rose* by nearly a third of the gross decline, and the
narrative was structurally incapable of saying so — `ranked()` filters to slices
moving *with* the change, by design. The tell was the concentration check
reporting that one contributor explained **126%** of the move, and passing,
because `share >= 0.35` is one-sided. A figure above 100% is not concentration;
it is the signature of offsetting movements. There is now a separate check for
it, and it fails. Both synthetic fixtures peak at 1.5% counter-movement, so
neither could ever have caught this.

**Fact *presentation* moves the answer as much as prompt wording.** Handed a flat
ranked list mixing dimensions, the model wrote that three segments "each
contributed over 10 percentage points" — silently adding overlapping partitions.
Same model, same instructions; leading with the intersection and explicitly
fencing off the alternative slicings fixed it.

---

## Limitations

**The change engine only sums.** Ask *"how did average resolution time change?"*
and summing gives **−1.31%** where the true change in the mean is **+0.20%** —
the opposite sign, not an approximation. A `NON_SUMMABLE` regex now forces
average/median/rate questions onto the code path, and `support_metric_trap`
asserts that routing, so the mitigation is tested. But the mitigation is a word
list: the engine itself is unchanged, and a rate question phrased without any of
those words still reaches it and still gets a total. The real fix is
mean-difference decomposition in the engine.

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

**Phrasing decides the route, and sometimes decides it wrongly.** `ROUTE_SYSTEM`
lists "a ranking" among the things belonging on the `other` route, so *"which
team and priority had the sharpest percentage rise in SLA breaches, and how much
did it actually matter?"* goes to generated code — 5 times out of 5 on the 14B.
That is a contribution question wearing a ranking's clothes, and the change
engine would answer it better, because resisting the biggest-percentage decoy is
precisely what the engine is for. A verified path and its decoy detection are
lost on wording alone. `support_decoy_pressure` asserts the current behaviour
rather than the desired one; fixing it means changing the routing prompt, which
is a change I would want A/B'd across both models before trusting it.

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
  fetch_nyc_payroll.py real 1.1M-row export — downloaded, not committed
tests/                 122 tests, no model required
```

Tests need no model or API key. The eval suite needs Ollama.
