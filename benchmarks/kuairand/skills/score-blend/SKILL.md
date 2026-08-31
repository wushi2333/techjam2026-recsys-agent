---
name: score-blend
description: Combine diverse ranking scores on valid only. Use when two identities have ≥2 seeds and are not head clones. Do not use ARIMA or time-series forecasts of blend weights.
arm: ensemble
keys: ""
status: wired
---

# Score blend

Same-config seed bag (rank-average) is first. Complementary first takes identities whose 3-seed means sit within ε=0.002 of the best (so a weaker FM is not pulled into a DeepFM–DeepFM pair). Only if that set has fewer than two fingerprints does the window widen to 0.03 (GBM + weaker FM). Those pairs get a **valid-only** minmax linear α plus score-product γ grid. Then near-top rank-average (ε=0.002).

- A weaker partner is legal if within-user top-1 agree is low. Clones (top-1 >0.98) add nothing.
- Two members: the α grid already searches correlation-aware weights. Do not import Qlib / Ledoit-Wolf.
- Product γ is the second-order term. It is not ARIMA.
- Submit the blend only if kit primary beats the **best same-config bag** by **2 paired SE**. A grid-boundary γ win inside noise is valid-overfit; keep the bag. Finalize only blends partners whose bag is within ε of the best bag (widen to 0.03 only if no near-top pair). Do not fuse a 3-seed loser onto the bag.
- Do not pick members on hidden test.
