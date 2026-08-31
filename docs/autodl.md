# AutoDL 1K job

Bonus 1K is a **KuaiRand-1K task instance**, pinned at launch (`--data-scale 1k` or `config/autodl.toml`). That is the job spec, not a mid-search intervention. The agent still chooses `fm|gbm|torch` keys, loss, seq, and architecture. Contest hidden test remains Pure and is a different ID space.

## Upload (do not upload `video_features_statistic_1k.csv`)

On the instance, typical layout:

```
/root/autodl-tmp/
  recsys-agent/
  kuairand-starter-kit/     # needs evaluate.py
  KuaiRand-1K/data/
    log_standard_4_08_to_4_21_1k.csv
    log_standard_4_22_to_5_08_1k.csv
    log_random_4_22_to_5_08_1k.csv
    video_features_basic_1k.csv
    user_features_1k.csv        # unused
```

Copy `.env` (DeepSeek key). Install:

```bash
cd /root/autodl-tmp/recsys-agent
python -m pip install -r requirements.txt
# image usually already has CUDA torch; if `python -c "import torch; print(torch.cuda.is_available())"` is False, install the matching wheel
```

## Launch (no further human choices)

```bash
export KUAI_CONFIG=/root/autodl-tmp/recsys-agent/config/autodl.toml
python -m agent env-check --config config/autodl.toml --data-scale 1k
python -m agent run --llm --config config/autodl.toml --data-scale 1k --run-dir run_1k
```

`env-check` must print `ok=1`. Draft 0 uses official FM hyperparameters (`k=16`, `lr=0.001`, `batch=8192`, `logloss`) on 1K; backend is torch when PyTorch imports. One GPU worker. Trial timeout floor 3600s.

Watch:

```bash
tail -f run_1k/progress.log
cat run_1k/env_probe.json
```

## What you do not need to send

- AutoDL password / SSH key
- A named next trial (DIN, DeepFM, …)

Useful if you have it: GPU SKU / VRAM (24GB is comfortable for 1K FM; 8GB is tight), and the directory you actually mounted if it is not `/root/autodl-tmp`.
