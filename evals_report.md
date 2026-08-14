## Eval results

| model | score | routing | interpretation | correctness | communication | median s | cost |
|---|---|---|---|---|---|---|---|
| `qwen2.5-coder:14b` | **301/306** (98%) | 100% | 100% | 98% | 97% | 25 | free |
| `qwen2.5-coder:1.5b` | **245/306** (80%) | 75% | 96% | 73% | 83% | 9 | free |

### Per case

| case | `qwen2.5-coder:14b` | `qwen2.5-coder:1.5b` |
|---|---|---|
| march_drop | 14–14/14 | 9–13/14 |
| march_drop_paraphrase | 6–6/6 | 6–6/6 |
| units_not_revenue | 5–5/5 | 5–5/5 |
| latest_period | 5–5/5 | 5–5/5 |
| decoy_pressure | 7–7/7 | 6–7/7 |
| channel_question | 6–6/6 | 6–6/6 |
| correlation | 7–7/7 | 7–7/7 |
| top_sales_rep | 6–6/6 | 1–1/6 |
| distinct_products | 4–4/4 | 4–4/4 |
| mean_units | 4–4/4 | 4–4/4 |
| support_breach_spike | 14–14/14 | 13–13/14 |
| support_decoy_pressure | 4–4/5 | 1–2/5 |
| support_metric_trap | 5–6/6 | 5–5/6 |
| support_median_resolution | 5–5/5 | 2–5/5 |
| support_open_tickets | 4–4/4 | 1–1/4 |
| support_reopened_count | 4–4/4 | 2–2/4 |

### Run-to-run variance

| model | mean score | sd | worst run | best run | unstable cases |
|---|---|---|---|---|---|
| `qwen2.5-coder:14b` | 98.4% | 0.6 pp | 98.0% | 99.0% | `support_metric_trap` |
| `qwen2.5-coder:1.5b` | 80.1% | 0.6 pp | 79.4% | 80.4% | `march_drop`, `decoy_pressure`, `support_decoy_pressure`, `support_median_resolution` |

Each model ran the full suite 3 times. A case listed as unstable scored differently across runs — local models are not deterministic even at temperature 0, so a single pass is a sample, not a score.

### Most common failures

- `routing / routed to code` — failed 9x
- `correctness / ran at least 1 successful code block(s)` — failed 9x
- `communication / no invented numbers` — failed 6x
- `communication / narrative mentions Onboarding` — failed 4x
- `correctness / reports the value 32.64` — failed 4x
- `correctness / at most 2 repair(s)` — failed 4x
- `correctness / reports the value 445532.17` — failed 3x
- `communication / narrative mentions rep_34` — failed 3x
