# deliverables/

**Pure is the contest primary.** 1K is an optional bonus with a different id space. Do not mix the two CSVs.

## Open first

| Path | Role |
|---|---|
| [`pure-v5/submission.csv`](pure-v5/submission.csv) | Contest scores, 170,588 test rows |
| [`pure-v5/progress.log`](pure-v5/progress.log) | Readable Pure trace |
| [`pure-v5/results.json`](pure-v5/results.json) | Valid table (0.60440 vs FM 0.6016) |
| [`pure-v5/journal.jsonl`](pure-v5/journal.jsonl) | Per-trial hypothesis and metrics |
| [`../docs/report.md`](../docs/report.md) | Longer write-up |
| [`../docs/DEVPOST.md`](../docs/DEVPOST.md) | Short project description |
| [`1k/`](1k/) | Bonus 1K snapshot (metrics, journal; **no** CSV) |

## Pure (`pure-v5/`)

Designated KuaiRand-Pure run. 3-seed pairwise BPR on the numpy FM. 50/50 billed iterations, 2.91 h, 0 runtime interventions.

## 1K (bonus)

Public snapshot: [`1k/`](1k/). Finalize bag valid **0.65001** vs 1K FM **0.64203**. Search stopped on the 6 h wall at 31/50.

The ~117 MB `submission.csv` may exist locally as `1k-aug31/` and is **gitignored**.

27K was not attempted.
