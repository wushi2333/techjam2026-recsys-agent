# Bonus KuaiRand-1K (`run_1k_aug31`)

Optional scale. **Not** the contest primary CSV. User and item ids are re-indexed, so these numbers are not comparable with Pure.

Search stopped on the 6 h wall at **31 / 50** billed iterations. Finalize retrained the confirmed 3-seed `use_time_decay` bag (`039`, `040`, `041`) on train and wrote test scores. A later 1-seed DCNv2 (0.65280) never got a 3-seed because the clock ran out; it is not in the CSV.

| | GAUC | nDCG@5 | primary | vs 1K FM |
|---|---|---|---|---|
| Official FM (1K) | 0.67461 | 0.60944 | 0.64203 | — |
| Finalize 3-seed rank average | 0.67654 | 0.62348 | **0.65001** | **+0.00798** |

Search bag valid was 0.65045; retrain drifted **-0.00044**. Alignment check: **4,132,081** test rows.

| | |
|---|---|
| Billed iterations | 31 / 50 |
| Stop | `wall_clock` (6.50 h) |
| Tokens (in + out) | 496,180 |
| Runtime interventions | 0 |

`report.json` still prints `delta_vs_baseline` against Pure’s FM (0.6016). Use **+0.00798** vs the 1K FM above.

This folder is the public snapshot: journal, progress, metrics. Extra tables and the 4.1M-row CSV: [techjam2026-recsys-agent_data-log](https://github.com/wushi2333/techjam2026-recsys-agent_data-log) (`1k/`).

Pure contest files: [`../pure-v5/`](../pure-v5/).
