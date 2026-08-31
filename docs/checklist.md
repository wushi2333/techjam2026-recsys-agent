# Submission packing (official §2.5)

Designated Pure run: **`run_pure_v5`** → slice `deliverables/pure-v5/`.  
KuaiRand-1K is bonus and optional; do not wait for it to file Pure.

## 1. Devpost written description

- [ ] Paste `docs/DEVPOST.md`
- [ ] Fill team names if not solo
- [ ] Tools / APIs / libraries / datasets already listed
- [ ] Link the public GitHub
- [ ] Attach or link `deliverables/pure-v5/submission.csv` if Devpost allows files
- [ ] Optional ~3 min video (recommended, not required). If skipped, the long report is `docs/report.md`

## 2. Public GitHub

Allowlist (do **not** upload the working tree):

```
agent/ templates/ tests/ config/ benchmarks/ scripts/
README.md requirements.txt pytest.ini conftest.py .env.example .gitignore
docs/DEVPOST.md docs/report.md docs/autodl.md docs/checklist.md
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
| Tokens in+out | 862,773 |
| Agent wall-clock | 2.91 h |
| Iterations | 50 / 50 |
| GPU-hours | 0.0 |

Do **not** put a self-scored hidden primary in the public table. Organizers score the CSV once.

## Same-day (deadline)

- [ ] Public repo URL on Devpost
- [ ] CSV uploaded wherever the form asks for model output
- [ ] 1K still running: mention “bonus in progress / optional” only; Pure is complete
- [ ] Rotate the AutoDL password (it was used in chat)

## Robustness smoke (optional, for the report table)

```bash
python -m unittest tests.test_fault_matrix -v
python scripts/fault_matrix.py
```

Ten injected faults; all must print `"ok": true`. Not a contest iteration.

## Watch 1K (does not block Pure)

```bash
tail -f /root/autodl-tmp/recsys-agent/run_1k_aug31/progress.log
```
