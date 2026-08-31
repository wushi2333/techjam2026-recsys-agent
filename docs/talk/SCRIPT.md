# 3-minute walkthrough — script

Slides: [`index.html`](index.html) (8 slides, 1920×1080).  
Target: **2:50–3:00**. Speak slightly slower than conversation. Do not ad-lib numbers.

Keys: `→` / space next · `←` back · `F` fullscreen · `N` speaker notes · `C` elapsed clock.

---

## How to record

1. Chrome, open `docs/talk/index.html` (or serve the `docs/` folder). Press `F`.
2. OBS / Clipchamp: 1920×1080, 30 fps, capture the browser window.
3. Put this script on a second screen or a phone. Do **not** press `N` in the recording window.
4. Press `C` on the recording window if you want a clock only you can see — skip it if it sits in the capture.
5. Click or tap `→` at the timestamps below. Leave ~0.4 s of silence after each advance.

---

## 中文讲稿（建议用来录）

### 1 · 封面 · 0:00–0:12

大家好。这是 TikTok TechJam 2026 赛道二：KuaiRand-Pure 上的用户内排序。我做了一个自主搜索循环——从官方 FM 出发，每次只改一件事，用官方 `evaluate.py` 打分，最后写出测试 CSV。全程 **0** 次人工干预。

### 2 · 任务 · 0:12–0:32

任务不是全库检索。对每个用户，只给他自己的曝光排序。标签是 `long_view`。主分是 GAUC 和 nDCG@5 的平均。官方 FM 验证集是 **0.6016**，公开测试是 0.5946。Kit 的打分器我们不改。

### 3 · 数字 · 0:32–0:55

交上去的是 **Submitted Pure**：三个随机种子上 pairwise BPR 的秩平均。验证主分 **0.60440**，比 FM 高 **0.00280**。搜索结束后，我们只把 CSV 打了一次测试：**0.59766**。这个测试分没有用来选模型。预算打满 50 次，2.91 小时。

### 4 · 架构图 · 0:55–1:25

架构分三层。上面是搜索：LLM 提出一个补丁，隔离的 trial 去训练，**父进程**用 kit 的 `evaluate.py` 打分——trial 改不了打分器。中间是决策：journal 当记忆，晋级门看当前 bag，而不是一个幸运种子。下面是输出：停下来之后，`finalize` 只在 train 上重训，写出 `submission.csv`。搜索阶段永远不读测试标签。

### 5 · 流程图 · 1:25–1:52

循环是 champion–challenger。每次一个原子改动。一颗种子先过四道门，看起来像样再三颗种子 ablate。三颗同号才晋级。停的条件是 ε=0.002 连续三次、五十次迭代、或六小时。Submitted Pure 打满了五十次。

### 6 · 泄漏 · 1:52–2:20

最有用的结果是一次失败。更早的 **Leaky Pure** 让 recency 看到了验证标签，并把缺失的测试标签存成 0。验证到了 **0.64**，同一份 CSV 测试却是 **0.568**——比官方 FM 还差。因为 kit 把整个 split 当成一张列表来排，滚动 `long_view` 就是组泄漏。

### 7 · 锁定 · 2:20–2:40

所以现在：没见过的标签是 **-1**，不是 0；decay 只吃 train 的 0 和 1；父进程持有打分器；finalize 选稳定的 bag，不选 valid 最大。泄漏那次的指纹也不会再被叠上去。

### 8 · 收束 · 2:40–3:00

Submitted Pure：50/50，2.91 小时，大约 86 万 token，**0** 次 runtime intervention。Bonus 1K 是可选尺度、另一套 ID：官方 FM 0.642 到 **0.650**，六小时墙上停在 31 次。代码在 GitHub，日志和 1K CSV 在独立的 data-log 仓库。谢谢。

---

## English script (if the video is in English)

### 1 · Cover · 0:00–0:12

This is TikTok TechJam 2026, Track 2: within-user ranking on KuaiRand-Pure. An autonomous loop. Start from the official factorization machine. Change one thing. Score with kit `evaluate.py`. Write the test CSV. Zero runtime interventions.

### 2 · Task · 0:12–0:32

This is not full-catalog retrieval. For each user, rank that user’s own impressions. Label is `long_view`. Primary is the mean of GAUC and nDCG@5. Official FM is 0.6016 on valid, 0.5946 on the published test. We do not edit the kit scorer.

### 3 · Numbers · 0:32–0:55

**Submitted Pure** is a three-seed rank average of pairwise BPR. Valid primary 0.60440, plus 0.00280 versus the FM. After search we scored the CSV once: 0.59766. That test number did not pick the model. Fifty of fifty billed steps, 2.91 hours.

### 4 · Architecture · 0:55–1:25

Three rows. Search: the LLM proposes a patch, an isolated trial trains, the **parent** runs kit `evaluate.py` — a trial cannot grade itself. Decide: the journal is memory; promotion is against the current bag, not a lucky seed. Emit: after stop, `finalize` retrains on train only and writes `submission.csv`. Search never reads test labels.

### 5 · Flowchart · 1:25–1:52

Champion–challenger. One atomic change. One seed through four gates; if it looks real, a three-seed ablate. Same-sign three of three promotes. Stop at epsilon 0.002 for three billed steps, fifty iterations, or six hours. Submitted Pure hit the iteration cap.

### 6 · Leak · 1:52–2:20

The useful result was a failure. **Leaky Pure** let recency see valid labels, and stored missing test labels as zero. Valid reached 0.64. The same CSV on test was 0.568 — worse than the official FM. The kit ranks a whole split as one list, so rolling `long_view` is group leakage.

### 7 · Locks · 2:20–2:40

So: unseen labels are minus one, not zero. Decay updates from train 0/1 only. The parent owns the scorer. Finalize picks a stable bag, not max valid. The leaky fingerprints are not restacked.

### 8 · Close · 2:40–3:00

Submitted Pure: fifty of fifty, 2.91 hours, about 863 thousand tokens, zero interventions. Bonus 1K is optional and a different id space: official FM 0.642 to 0.650, stopped on the six-hour wall at 31. Code on GitHub; logs and the 1K CSV in the data-log repo. Thank you.

---

## Numbers — say only these

| Line | Say |
|---|---|
| Official FM valid | 0.6016 |
| Submitted Pure valid | 0.60440 · plus 0.00280 |
| Contest CSV, once | 0.59766 |
| Leaky valid / test | 0.64 / 0.568 |
| Official FM test | 0.5946 |
| Submitted Pure budget | 50 of 50 · 2.91 hours · 0 interventions |
| Bonus 1K | 0.650 versus 0.642 · 31 of 50 · 6 hours |

Do not say the 1K `report.json` delta 0.048. Do not say nigelyeap. Do not say hidden labels were used to pick anything.
