---
name: time-decay
description: Causal recency-decay and session momentum versus static ID fields. Use when the features arm still has use_time_decay untried. Do not retry organizer static-ID ablation as this flag.
arm: features
keys: use_time_decay, bpr_decay_sample
status: wired
---

# Time decay

`use_time_decay` (default false, low prior) attaches decay_rate / decay_act (halflife 2.5d), decay_tab (3d), last1 / lastk_rate / gap. Same-calendar-day rows do not see each other for decay. A row never reads its own label. Test `long_view` is 0, so test days add impressions not positives.

`bpr_decay_sample` weights BPR/listwise user draws by decayed train positives^0.75. No-op on logloss.

Organizer "static IDs no gain" is **scoped to static IDs**. It does not ban temporal aggregation. Do not add tsfresh's 76 calculators; this flag is the family.
