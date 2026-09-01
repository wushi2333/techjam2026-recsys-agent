# deliverables/

**Pure is the contest primary.** 1K is an optional bonus with a different id space. Do not mix the two CSVs.

## Open first

| Path | Role |
|---|---|
| [`pure-v6/submission.csv`](pure-v6/submission.csv) | **Contest** scores, 170,588 test rows |
| [`pure-v6/progress.log`](pure-v6/progress.log) | Readable v6 trace (thinking off) |
| [`pure-v6/results.json`](pure-v6/results.json) | Valid 0.60458; hidden test 0.59833 (once after search) |
| [`pure-v6/journal.jsonl`](pure-v6/journal.jsonl) | Per-trial hypothesis and metrics |
| [`pure-v5/`](pure-v5/) | Thinking-on BPR comparison |
| [`../docs/report.md`](../docs/report.md) | Longer write-up |
| [`../docs/DEVPOST.md`](../docs/DEVPOST.md) | Short project description |
| [`1k/`](1k/) | Bonus 1K snapshot (metrics, journal) |
| Extra tables / 1K CSV / dumps | [data-log repo](https://github.com/wushi2333/techjam2026-recsys-agent_data-log) |

## Submitted Pure (`pure-v6/`)

Contest search. DeepSeek thinking off. 3-seed DeepFM + seq-100 pool + l2. 50/50 billed iterations, 1.87 h, 0 runtime interventions.

## Comparison Pure (`pure-v5/`)

Thinking-on BPR bag. Valid 0.60440 / hidden 0.59766. Not the contest CSV.

## Bonus 1K

Public snapshot: [`1k/`](1k/). Finalize bag valid **0.65001** vs 1K FM **0.64203**. Search stopped on the 6 h wall at 31/50.

The ~117 MB 1K `submission.csv` may exist locally as `1k-aug31/` (gitignored). Extra tables, that CSV, and run dumps: [techjam2026-recsys-agent_data-log](https://github.com/wushi2333/techjam2026-recsys-agent_data-log).

27K was not attempted.
