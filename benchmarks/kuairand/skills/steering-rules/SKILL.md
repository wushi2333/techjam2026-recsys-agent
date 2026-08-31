---
name: steering-rules
description: Short ranking-search steering. Use on plateau or when tempted to add a deep architecture. Do not ingest RecBole or a 200-item generic ML dump.
arm: any
keys: ""
status: wired
---

# Steering (scoped)

Compressed from Google Rules of ML and this task's measurements. Not a to-do list.

1. Change one thing; the metric is within-user order of logged impressions, not train loss.
2. Keep 2–5 ensemble members that disagree on the list head. More clones do not help.
3. Plateau: new **information** (causal time aggregation, a different family encoding), not larger k or AutoInt.
4. Incumbent errors are a feature mine — `diagnose` `user_mixed` / `sparse_counts` (train+valid only).
5. Wired families already cover FM / DeepFM / DCNv2 / DIN / GBM / listwise. Do not import RecBole, Qlib, or tsfresh into the loop.
