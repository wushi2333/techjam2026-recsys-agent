# recsys-agent

Autonomous ML research agent for **TikTok TechJam 2026 Track 2**: within-user ranking on **KuaiRand-Pure**, label `long_view`, scored by the official starter-kit `evaluate.py`.

Primary = mean(GAUC, nDCG@5). The agent reproduces the official Factorization Machine, then runs Draft / Debug / Improve / Ablate on **train + validation only**. Hidden test labels are never used in search, EDA, or feature state. Finalize infers test **scores** once for the designated CSV.

## For judges (about one minute)

| Want | Open |
|---|---|
| Score + CSV | [`deliverables/pure-v5/README.md`](deliverables/pure-v5/README.md) and [`submission.csv`](deliverables/pure-v5/submission.csv) |
| Why 0.604 not 0.640 | [`docs/report.md`](docs/report.md) §2 (withdrawn run) |
| Human-readable 50-iter trace | [`deliverables/pure-v5/progress.log`](deliverables/pure-v5/progress.log) |
| Hypothesis / diff / metrics | [`deliverables/pure-v5/journal.jsonl`](deliverables/pure-v5/journal.jsonl) |
| Paste-ready Devpost text | [`docs/DEVPOST.md`](docs/DEVPOST.md) |
| Do **not** re-run the 2.9 h search | The CSV is the §2.5 output. Optional: `python -m unittest discover -s tests -v` (no Pure dump) |

Designated valid primary **0.60440** (+0.00280 vs official FM 0.6016). Runtime interventions **0**. Hidden is scored by organizers from the CSV.

## Scoring contract (kit, not the brief's NDCG@10 / Recall@50 line)

| | |
|---|---|
| Task | Within-user ranking over logged impressions (not full-catalog retrieval) |
| Label | `long_view` |
| Metrics | GAUC, nDCG@5; **primary = mean of the two** |
| Split | train `20220408–20220421` / valid `20220422–20220428` / test `20220429–20220508` |
| Official FM | valid primary **0.6016** / hidden **0.5946** |
| Stop | ε = 0.002, N = 3 on validation primary, **or** 50 billed iterations, **or** 6 h wall — whichever first |

**Protocol we pin (same metric, honest sequential features):** GAUC / nDCG@5 rank **all** of a user's impressions in that split together (valid = 7 days, test = 10 days). Recency / decay / last-k features therefore update **only from train labels**. Valid and test rows are unlabeled (`LABEL_MISSING`, not `0`). A row never sees its own label.

Do not treat a public 0.66-style number as this contract unless eval-split labels were frozen the same way.

## Designated Pure result

Run `run_pure_v5`. Logs and CSV: [`deliverables/pure-v5/`](deliverables/pure-v5/).

| | GAUC | nDCG@5 | primary | Δ vs FM valid |
|---|---|---|---|---|
| Official FM (valid) | 0.6674 | 0.5357 | 0.6016 | — |
| **Submitted bag (valid, 3-seed rank-average, `bpr_global`)** | **0.67105** | **0.53774** | **0.60440** | **+0.00280** |

Runtime interventions: **0**. Tokens: **862,773**. Agent wall-clock: **2.91 h**. Iterations: **50 / 50**. GPU-hours: **0**.

KuaiRand-1K / 27K are **bonus** (optional). A 1K job may be running separately; it does not replace Pure.

## Layout

```
agent/                 search, sandbox, journal, promote gates, finalize
templates/             trial pipeline (FM / optional GBM / torch)
benchmarks/kuairand/   spec.json + priors (not bans)
deliverables/pure-v5/  designated run slice for judges
tests/                 synthetic harness, no full dump required
```

The parent process re-scores every trial from `scores.npz` with kit `evaluate.py`. Trials cannot patch `evaluate.py`.

## Setup

Python 3.9+ (3.10 used in the designated run). Need the starter kit and KuaiRand-Pure logs from [kuairand.com](https://kuairand.com) / Zenodo.

```bash
export KUAI_KIT_DIR=/path/to/kuairand-starter-kit
export KUAI_DATA_DIR=/path/to/KuaiRand-Pure/data
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m agent run --smoke
```

`--smoke` is 1 epoch and a row cap (minutes, not a contest run).

Full search (needs an OpenAI-compatible key in `.env`; see `.env.example`):

```bash
cp .env.example .env   # set OPENAI_API_KEY + OPENAI_BASE_URL
python -m agent ping-llm
python -m agent run --llm --run-dir run_pure
python -m agent finalize --run-dir run_pure
```

Finalize retrains the selected identity on **train**, writes `run_pure/finalize/submission.csv` in kit order, and does **not** call `evaluate` on test labels. Watch without intervening: `python -m agent status`.

On AutoDL, KuaiRand-1K is a **launch pin** (`--data-scale 1k`), not a mid-run experiment. See `docs/autodl.md`.

## Reproduce

The designated scores are `deliverables/pure-v5/submission.csv` (170,588 rows). That file is the §2.5 model output. The slice next to it has `journal.jsonl` / `summary.json` so the run can be audited without the multi-GB `trials/` tree.

To run a **new** Pure search (not required to accept the CSV):

```bash
python -m agent run --llm --run-dir run_pure
python -m agent finalize --run-dir run_pure
```

Format-check the CSV with the starter kit (`submit.py --check --split test`) when `data.py` is pointed at Pure.

## Limitations (what we would do with more time)

- The 50-iteration cap stopped this Pure run (`stop_reason=cap`). Ensemble / 3-seed children are not billed; that is our accounting, documented in `journal.py`.
- Finalize selected a 3-seed `bpr_global` bag on nested valid, not the search incumbent (DeepFM + seq 50). Both beat official FM valid; the bag is the designated CSV.
- Sequential features that **roll eval-split labels into the same ranking list** inflate valid and test together. We froze those labels after a withdrawn run (`run_pure_v4`) printed valid **0.63975** and then a post-hoc CSV check **0.56790** (below FM). The designated valid **0.60440** is the number after that fix. Details: `docs/report.md` §2.
- Bonus 1K/27K are optional; Pure is 100% of the primary metric.
- A ~3 min video is recommended by the brief and is not in this repo yet.

## Team

Solo unless additional names are listed on the Devpost entry.

## License / data

Code in this repository is for the TechJam submission. KuaiRand data remains under its own terms. Do not commit `.env`.
