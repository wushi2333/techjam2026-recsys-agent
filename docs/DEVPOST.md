# Devpost — Track 2: Autonomous ML research agent (KuaiRand-Pure)

Paste into the written project description. Hidden test is scored by organizers from the CSV; this text reports **validation** and resources.

## How this addresses the problem

Track 2 asks for an agent that reproduces the official FM, iterates on train + public validation, and designates one submission for a single hidden-test evaluation. Ours is a greedy champion–challenger loop (Draft / Debug / Improve / Ablate / ensemble) over a numpy FM pipeline.

Pinned contract (starter kit, not the brief's leftover NDCG@10 / Recall@50 / click line):

- Within-user ranking on logged impressions
- Label `long_view`
- Primary = mean(GAUC, nDCG@5) via unmodified `evaluate.py`
- Search never reads test `long_view`
- Sequential features (decay / last-k) update from **train labels only**; valid/test are missing, not zero

Designated run `run_pure_v5`:

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| Official FM valid | 0.6674 | 0.5357 | 0.6016 |
| Our valid-best (3-seed rank-average, pairwise BPR) | 0.67105 | 0.53774 | **0.60440** (+0.00280) |

Runtime interventions: **0**. 50 / 50 billed iterations, 2.91 h wall, 862,773 tokens, 0 GPU-hours. CSV: `deliverables/pure-v5/submission.csv` (170,588 rows).

The agent tried loss, architecture, sequence length, regularization, and capacity. Pairwise BPR aligned training with a ranking metric; extra depth on top of a confirmed FM did not clear the 3-seed screen. Same-config seed bagging is the designated submit.

## What did not work (kept on purpose)

An earlier full Pure run (`run_pure_v4`) reached **valid primary 0.63975** (search bag 0.63931) by stacking recency-decay features and blending same-family siblings. That is **not** our submission. After the run we scored that CSV once as a diagnostic: **0.56790**, below official FM hidden 0.5946. Valid had been inflated by (1) rolling valid `long_view` into decay/last-k while GAUC/nDCG rank the whole 7-day list, and (2) filling test labels with 0, which poisoned decay on test.

We then froze eval labels (`LABEL_MISSING`, train-only sequential state), screened against the bag with CI and temporal halves, and forbade same-leak blends. The designated v5 valid is **0.60440**. A lower valid number after a protocol fix is the result we trust.

## Development tools

- VS Code / terminal; Python 3.10
- Git
- AutoDL (optional bonus KuaiRand-1K job; not required for Pure)

## APIs

- DeepSeek Chat Completions, OpenAI-compatible (`OPENAI_BASE_URL=https://api.deepseek.com`, model `deepseek-v4-flash`)
- No organizer-hosted model. Dummy planner if no key (not the designated run)

## Libraries and frameworks

- numpy (FM / BPR, designated path)
- LightGBM (optional `model_family=gbm`, not in the designated CSV)
- PyTorch (optional `model_family=torch`, not in the designated CSV)
- Standard library + `tomli` on Python < 3.11

Starter-kit `evaluate.py` / `submit.py` are used as given.

## Datasets and assets

- **KuaiRand-Pure** only for the primary score (official splits in the starter kit)
- KuaiRand-1K is a bonus instance (different ID space; do not compare its primary to Pure 0.6016)
- No extra training data, no weights trained on these benchmarks' test labels
- `log_random_*` is scored once at finalize as an off-policy check and is **not** used to choose the model

## Robustness (official: recover / retry / route around)

The designated Pure run did not crash (0 buggy, 0 Debug, 0 timeouts). It **did** route around 10 skip nodes (duplicate fingerprints, a v4 leak graveyard, an empty sequence arm) so search did not restuck. Crash, timeout-partial, hidden-label block, `evaluate.py` patch block, journal/wall restart, and Debug routing are covered by `python scripts/fault_matrix.py` (10/10 injects). We do not claim the 6 h run cannot fail; we claim a failed *step* is not supposed to kill the *run*. v4’s 0.64 valid was metric divergence, not a process crash — freeze-eval is the fix for that class.

## Autonomy and logs

Per-iteration hypothesis, diff, metrics, and recovery: `deliverables/pure-v5/journal.jsonl` and `changelog.jsonl`.  
Runtime intervention count: **0**. Five **build-time** lines (pre-run scripts / one template fix) are in `interventions.jsonl` and are not mid-run direction changes.

## Limitations

See the GitHub README. We report the kit valid number and let hidden be scored once from the CSV. We do not treat rolling eval-split labels in sequential features as a valid way to raise primary.
