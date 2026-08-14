## Eval results

| model | score | routing | interpretation | correctness | communication | median s | cost |
|---|---|---|---|---|---|---|---|
| `qwen2.5-coder:14b` | **299/306** (98%) | 100% | 100% | 100% | 93% | 25 | free |
| `qwen2.5-coder:1.5b` | **266/306** (87%) | 85% | 100% | 87% | 80% | 5 | free |

### Per case

| case | `qwen2.5-coder:14b` | `qwen2.5-coder:1.5b` |
|---|---|---|
| march_drop | 14–14/14 | 13–13/14 |
| march_drop_paraphrase | 6–6/6 | 5–5/6 |
| units_not_revenue | 5–5/5 | 5–5/5 |
| latest_period | 5–5/5 | 4–5/5 |
| decoy_pressure | 7–7/7 | 7–7/7 |
| channel_question | 5–6/6 | 6–6/6 |
| correlation | 7–7/7 | 7–7/7 |
| top_sales_rep | 6–6/6 | 1–1/6 |
| distinct_products | 4–4/4 | 4–4/4 |
| mean_units | 4–4/4 | 3–4/4 |
| support_breach_spike | 14–14/14 | 12–13/14 |
| support_decoy_pressure | 6–6/6 | 6–6/6 |
| support_metric_trap | 5–5/5 | 5–5/5 |
| support_median_resolution | 5–5/5 | 5–5/5 |
| support_open_tickets | 3–3/4 | 1–1/4 |
| support_reopened_count | 3–3/4 | 3–3/4 |

### Run-to-run variance

| model | mean score | sd | worst run | best run | unstable cases |
|---|---|---|---|---|---|
| `qwen2.5-coder:14b` | 97.7% | 0.6 pp | 97.1% | 98.0% | `channel_question` |
| `qwen2.5-coder:1.5b` | 86.9% | 1.5 pp | 85.3% | 88.2% | `latest_period`, `mean_units`, `support_breach_spike` |

Each model ran the full suite 3 times. A case listed as unstable scored differently across runs — local models are not deterministic even at temperature 0, so a single pass is a sample, not a score.

### Most common failures

- `communication / no invented numbers` — failed 19x
- `routing / routed to code` — failed 6x
- `correctness / ran at least 1 successful code block(s)` — failed 6x
- `correctness / reports the value 445532.17` — failed 3x
- `correctness / at most 2 repair(s)` — failed 3x
- `communication / narrative mentions rep_34` — failed 3x
- `communication / leads with Platform` — failed 3x
- `correctness / reports the value 2210` — failed 3x
