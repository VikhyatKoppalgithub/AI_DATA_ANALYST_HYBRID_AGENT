---
title: AI Data Analyst
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Ask a spreadsheet a question, get an answer whose numbers were computed and checked
---

# AI Data Analyst — live demo

Upload a messy spreadsheet, ask a business question in English, and get an
answer whose every number was **computed and checked** rather than generated.

The design constraint: the language model never does arithmetic. It reads your
question and picks columns; pandas computes the answer; the model then writes
prose over figures it was handed and is not allowed to invent one.

**Source and full write-up:**
<https://github.com/VikhyatKoppalgithub/AI_DATA_ANALYST_HYBRID_AGENT>

---

## Two things to know before you try it

**This Space runs `qwen2.5-coder:1.5b`, not the 14B.** A free CPU Space has
2 vCPU and no GPU, and the 14B the published numbers come from needs ~9GB and
Metal/CUDA to be tolerable. The 1.5b is the same model the eval suite already
scores, so its weaknesses are documented rather than mysterious:

| model | eval score | routing | where it fails |
|---|---|---|---|
| `qwen2.5-coder:14b` | 98% (301/306) | 100% | two code-route cases |
| `qwen2.5-coder:1.5b` *(this Space)* | 80% (245/306) | 75% | mostly routing — it sends ranking questions to the wrong path |

**It is slow.** CPU inference on 2 cores, three model calls per verified answer.
Expect a minute or more per question, and longer on the first one after the
Space wakes up. That is the honest cost of keeping the demo free and
key-less rather than wiring it to a hosted API.

## What to try

Tick **"Use the bundled demo dataset"** and ask:

- *Why did revenue drop in March?* — the verified path. Watch the verification
  panel: those checks can genuinely fail.
- *What is the correlation between units and unit_price?* — the sandboxed code
  path. The generated Python is shown next to the answer, labelled unverified.

The **Data profile** tab works with no model at all, and is the fastest way to
see what the agent actually sends to the LLM: a 9,000-row file rendered into a
few hundred tokens.

## Why the answer carries a label

Two paths, two different guarantees, surfaced rather than blended:

- **Verified** — the deterministic contribution engine. Segment contributions
  must sum to the total change, so a dropped-row bug shows up as a failed check
  rather than a plausible wrong number.
- **Unverified** — model-written Python in a subprocess sandbox. It answers
  things the engine cannot express (correlations, medians, rankings), but
  nothing reconciles its figures, and the UI says so.
