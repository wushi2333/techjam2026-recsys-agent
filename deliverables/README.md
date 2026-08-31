# deliverables/

Two scales, kept apart. **Pure is the contest primary.** 1K is an optional bonus with a different id space.

## Pure (tracked, this is the submission)

[`pure-v5/`](pure-v5/) — designated KuaiRand-Pure run.

- `submission.csv` — 170,588 test rows
- `results.json` / `summary.json` — valid table, 50/50, 2.91 h
- `progress.log` / `journal.jsonl` — readable trace

Valid primary **0.60440** vs official FM **0.6016**. Write-up: [`../docs/report.md`](../docs/report.md). Devpost notes: [`../docs/DEVPOST.md`](../docs/DEVPOST.md).

## 1K (bonus)

[`1k/`](1k/) — public snapshot (no CSV): metrics, journal, progress. Finalize bag valid **0.65001** vs 1K FM **0.64203**.

`1k-aug31/` may exist on disk with the ~117 MB `submission.csv`. It is **gitignored** and is not mixed into the Pure file.

27K was not attempted.
