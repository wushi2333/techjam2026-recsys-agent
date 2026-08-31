# Submission packing (official §2.5)

Designated Pure run: **`run_pure_v5`** → slice `deliverables/pure-v5/`.  
KuaiRand-1K is bonus and optional; do not wait for it to file Pure.

## 1. Devpost written description

- [ ] Paste `docs/DEVPOST.md`
- [ ] Fill team names if not solo
- [ ] Tools / APIs / libraries / datasets already listed
- [ ] Link the public GitHub
- [ ] Attach or link `deliverables/pure-v5/submission.csv` if Devpost allows files
- [ ] ~3 min walkthrough for Devpost (**do this last**; then link it from README)
- [x] Extra result tables / 1K CSV / run dumps: [techjam2026-recsys-agent_data-log](https://github.com/wushi2333/techjam2026-recsys-agent_data-log)

## 2. Public GitHub

Allowlist (do **not** upload the working tree):

```
agent/ templates/ tests/ config/ benchmarks/ scripts/
README.md requirements.txt pytest.ini conftest.py .env.example .gitignore
docs/DEVPOST.md docs/report.md docs/autodl.md docs/checklist.md
docs/figures/
deliverables/
```

Keep out:

- `.env`
- `run/` `run_*/` (full trees)
- `finalize/hidden_test.json`, `infer_scores.npz`, `*.npz`
- `docs/题目.txt`, webinar PNGs with other names
- `docs/第一轮提示词.txt` `docs/第二轮提示词.txt`
- `_ab_budget/`, `.encode_cache/`

```bash
git status   # no .env, no hidden_test.json, no run_pure_v5/trials
```

README already has overview, setup, reproduce, limitations, team.

## 3. Run logs + interventions

Already in `deliverables/pure-v5/`:

| Required | File |
|---|---|
| Hypothesis / diff / metrics / recovery | `journal.jsonl`, `changelog.jsonl` |
| Intervention summary | runtime **0**; file `interventions.jsonl` |

## 4. Final CSV + valid table + resources

| Required | Value |
|---|---|
| CSV | `deliverables/pure-v5/submission.csv` |
| Valid GAUC / nDCG@5 / primary | 0.67105 / 0.53774 / **0.60440** |
| Δ vs FM valid primary | **+0.00280** |
| Hidden test GAUC / nDCG@5 / primary | 0.66486 / 0.53046 / **0.59766** |
| Δ vs FM hidden primary | **+0.00306** (scored once after search; not used to pick the model) |
| Tokens in+out | 862,773 |
| Agent wall-clock | 2.91 h |
| Iterations | 50 / 50 |
| GPU-hours | 0.0 |

Hidden numbers above are a one-time diagnostic after search. Organizers still score the CSV once.

## Same-day (deadline)

- [ ] Public repo URL on Devpost
- [ ] CSV uploaded wherever the form asks for model output
- [ ] 1K is a finished bonus (optional, different id space); do not treat it as the Pure CSV
- [ ] ~3 min video last
- [x] Extra tables / 1K CSV: https://github.com/wushi2333/techjam2026-recsys-agent_data-log
- [ ] Rotate the AutoDL password (it was used in chat)

## Robustness smoke (optional, for the report table)

```bash
python -m unittest tests.test_fault_matrix -v
python scripts/fault_matrix.py
```

Ten injected faults; all must print `"ok": true`. Not a contest iteration.

## 1K bonus (does not replace Pure)

Search + finalize already finished (`run_1k_aug31`). Public snapshot: `deliverables/1k/`. The 1K CSV and extra tables: https://github.com/wushi2333/techjam2026-recsys-agent_data-log

```bash
# logs only, if the instance is still up
tail -f /root/autodl-tmp/recsys-agent/run_1k_aug31/progress.log
```
