# Six-layer autonomous ranking agent

This repo is an autonomous ML research agent for the KuaiRand-Pure ranking
task. It searches in **code + config space**. The official `evaluate.py` is
the only objective function.

## Kit pins (do not override)

The hackathon brief mentions click / NDCG@10 / Recall@50 in places. The
starter kit is the scoring contract:

| Field | Value |
|---|---|
| Task | Within-user ranking over logged impressions |
| Label | `long_view` |
| Metrics | GAUC, nDCG@5; **primary = mean** |
| FM valid | primary 0.6016 |
| FM hidden test | primary 0.5946 |
| Convergence | ε = 0.002, N = 3 on **validation** primary |
| Split | train 20220408–20220421 / valid 20220422–20220428 / test 20220429–20220508 |

Search never reads test labels. Test is scored once at finalize.

Organizer dead ends (do not spend iterations here): extra static features,
larger embedding `k`. User-side first-order terms cannot change within-user
order. Headroom is loss (BPR / listwise), sequences, multi-task, watch-time
(CWM), then DeepFM / DCN.

## Layers

```
L6  Deliverable   valid-submission gate, top-k hook, ablation table, dashboard
L5  Recsys prior  Thompson arms, local → jump, paper module registry
L4  Memory        Journal, Error Memory, Experiment Skill, Paper KB
L3  Search        greedy Draft/Debug/Improve; UCT + fan-out reserved
L2  Operators     Draft, Debug, Improve, Crossover, PaperImpl, Ablate
L1  Environment   isolated trial, timeout, forbidden paths, kit evaluator
L0  Contract      official FM = s0, ranking only, interventions = 0
```

## Reserved switches (wired, not fully executed)

| Switch | Config | Behaviour today |
|---|---|---|
| Error Memory | `[error_memory]` | Records signatures; retrieves by token overlap |
| Jump DeepFM/DCNv2 | `[jump]` | Unlocks after stagnation; PaperImpl still reserved |
| Multi-task aux heads | `[multitask]` | Spec + arm exist; trainer not switched on |
| 2–4 parallel trials | `[parallel]` | Fan-out helper present; default `n_workers=1` |

UCT and island evolution are **not** the default. Greedy + atomic Improve is.

LLM: `provider=auto` uses an OpenAI-compatible Chat Completions endpoint when
`OPENAI_API_KEY` or `XAI_API_KEY` is set. Draft 0 never calls the LLM.
Empty / invalid patches are skipped so a no-op arm does not retrain.

## Observability (watch-only)

Humans may read `run/status.json`, `run/journal.jsonl`, `run/events.jsonl`,
and `run/trials/*/train.log` at any time. That is not an intervention.
Interventions are writes to code, arms, or promotion.
