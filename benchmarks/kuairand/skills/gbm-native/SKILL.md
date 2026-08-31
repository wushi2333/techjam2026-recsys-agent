---
name: gbm-native
description: LightGBM on un-bucketed numeric columns with small trees. Use when model_family=gbm is already on and gbm_leaves is still default 31. Do not treat a default ID-only GBM as a family ban.
arm: architecture
keys: model_family, gbm_leaves
status: wired
---

# GBM native encoding

Trees need continuous splits. Forcing GBM through FM quantile buckets / high-card IDs is a different experiment from `model_family=gbm` plus `enc["num"]` (beh_cross / time_decay floats).

When `model_family=gbm`, `legal_untried` includes `gbm_leaves` in {2, 7}. Stumps on native floats are the prior. Default `gbm_leaves=31` is not a falsification of the family.
