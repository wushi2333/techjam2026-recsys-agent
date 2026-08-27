# recsys-agent

Autonomous ranking research agent for KuaiRand-Pure. Six-layer tree search
over an official Factorization Machine, scored only by the starter-kit
`evaluate.py`.

## What it does

1. Reproduces the official numpy FM (validation primary ≈ 0.6016).
2. Iterates with Draft / Debug / Improve. Each step logs hypothesis, diff,
   metrics, and recovery.
3. Promotes a candidate only if **validation** primary improves.
4. Stops at ε = 0.002 over N = 3 rounds, or when the iteration cap hits.
5. Writes a `row_id,user_id,video_id,score` CSV for the designated split.

LLM provider is `auto`: if `OPENAI_API_KEY` or `XAI_API_KEY` is set, Improve
uses that OpenAI-compatible model; otherwise Dummy mutates `trial_config.json`.
Draft 0 always reproduces the official FM and does not call the LLM.

## Layout

```
agent/          L0–L6 implementation
templates/      runnable FM pipeline copied into each trial
config/         default.toml
docs/           architecture
run/            live logs (gitignored)
```

## Setup

Python 3.9+ and numpy. Data and kit already live next to this repo:

```
D:\tictokJam\kuairand-starter-kit
D:\tictokJam\Kuairand\KuaiRand-Pure\data
```

```powershell
cd D:\tictokJam\recsys-agent
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m agent run --smoke
python -m agent status
```

`--smoke` uses 1 epoch and a row cap so a laptop can finish in minutes.

Full search:

```powershell
copy .env.example .env   # then fill OPENAI_API_KEY or XAI_API_KEY
python -m agent ping-llm
python -m agent run
```

Without a key, `provider=auto` stays on Dummy.

Watch without intervening:

```powershell
python -m agent status
Get-Content run\journal.jsonl -Tail 5
Get-Content run\status.json
```

## Scoring contract

Label is `long_view`, not click. Primary is mean(GAUC, nDCG@5). Hidden test
is not used during search. See `docs/architecture.md`.
