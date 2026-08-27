# 阶段总结 01：调研定调 + 六层框架落地 + 全量自检

**日期：** 2026-08-27  
**仓库：** `D:\tictokJam\recsys-agent`  
**Git：** `0881fab` 脚手架 → `8f99a82` 本报告初稿 → 本文件为全量实验后的补充  
**目的：** 给后续 Devpost / 书面报告 / 答辩提供可引用事实，不代替最终 hidden-test 成绩表。

---

## 1. 本阶段完成了什么

四件事，按顺序：

1. **读题并调研**公开的 Autonomous ML Research Agent 与推荐系统自迭代文献，确定可获奖路线，而不是去复现某篇 SOTA 表。
2. **把路线收成六层架构**，并明确哪些优点互相冲突、不能同时打开。
3. **在本地新建 git 工作区并搭出可跑的脚手架**：Draft 0 复现官方 FM，Dummy LLM 就能闭环，人只读观测、零干预。
4. **关掉 smoke，全量复现官方 FM，并在同一份 FM 上做了几轮 Dummy 局部搜索**（见 §6.1 / §6.2）。

**官方 FM 是评分参照，也是 Task 1 必须打到的分数；不是比赛的全部。** Agent 应当自己写后续阶段的代码。本阶段 Dummy 只改了 `trial_config`（损失 / 学习率），还没有让模型改 `pipeline.py`。Hidden test 未评。

---

## 2. 题目与 starter kit：报告里必须写清的口径

黑客松题目正文（2026-08-26 更新）与 `kuairand-starter-kit` 有不一致。**评分以 kit 为准。**

| 项目 | 题目正文里出现过 | **Starter kit 钉死（采用）** |
|---|---|---|
| 任务 | 推荐漏斗、偏 ranking | 用户内排序（within-user ranking over logged impressions），不做全库召回 |
| 标签 | `is_click` / click | **`long_view`** |
| 指标 | NDCG@10 / Recall@50 | **GAUC、nDCG@5；primary = 二者平均** |
| 官方 FM（valid） | GAUC 0.6674 / nDCG@5 0.5357 / primary **0.6016** | 同左 |
| 官方 FM（hidden test） | GAUC 0.6610 / nDCG@5 0.5282 / primary **0.5946**（5 seed，std 0.0008） | 同左 |
| 对照 | 随机 0.4753；热门 0.5715 | 同左 |
| 收敛 | ε = 0.002，N = 3 | 同左，看 **validation primary** |
| 划分 | train 20220408–20220421 / valid 20220422–20220428 / test 20220429–20220508 | 同左 |
| 提交 | `row_id,user_id,video_id,score` | 同左；`(user_id, video_id)` 非主键（test 约 3.06% 重复） |

Oracle 上限（用真标签当分数）：test primary **0.8645**（nDCG@5 上限 0.7289，因 27.1% 用户全负、nDCG 恒为 0）。评估进展应以 oracle 为分母，不要拿 1.0 当满分。

**组委会已测死、禁止再烧迭代：**

- 把 CWM 的 13 个静态特征域接进 FM：无收益甚至略降。
- embedding `k = 8/16/32`：几乎不动。
- 纯用户侧一阶项对 within-user 排序贡献恒为 0。

**组委会标明未测、建议优先级：**

1. 换损失（pairwise BPR / listwise softmax）——与 GAUC/nDCG 对齐  
2. 用户历史序列（DIN / SIM）  
3. 多任务辅助 `long_view`（click / like / follow 等）  
4. 观看时长（CWM 删失回归）  
5. 再换 DeepFM / DCN / xDeepFM  
6. 时间特征与分布漂移  
7. `log_random_*` 无偏验证  

KuaiRand-Pure 是 **100% primary**。1K / 27K 仅为加分，本阶段不碰。

---

## 3. 文献结论：评委在评什么，公开系统贡献了什么

### 3.1 打分轴（题目 2.6）

| 轴 | 权重 | 本阶段对应设计 |
|---|---|---|
| Technical Execution | 35% | 相对官方 FM 的 hidden-test 绝对提升；失败可恢复 |
| Innovation & Insight | 20% | 全栈改动 + 引用公开方法；不限模型结构 |
| Impact（Autonomy） | 20% | 人工干预次数，目标 0 |
| Feasibility | 15% | 总 token + GPU-hours |
| Presentation | 10% | 决赛；Journal 可直接生成表格 |

交付物要求每轮：**hypothesis、code diff、metrics、error/recovery**。这决定了不能抄无逐步日志的岛屿进化。

### 3.2 题目点名的三套

| 系统 | 真正优点 | 本阶段吸收 |
|---|---|---|
| **MLE-bench** | 考场纪律：合法提交、debug/恢复、hidden test 只评一次；通用 agent 会早停 | L0/L1：`evaluate.py` 唯一 \(h(s)\)；超时记 buggy；search 禁止 test 标签 |
| **AIDE** | 代码空间树搜索；Draft / Debug / Improve；greedy + debug 深度帽；原子 Improve；\(\Sigma(T)\) 摘要 | L2/L3 默认策略。**不吸收** LLM 从 stdout 读 metric |
| **AI-Scientist-v2** | 四阶段实验经理、多 draft、阶段最优播种、消融、成文 | 阶段机写进架构（复现 → 局部 → jump → 消融 → 成文）；本阶段只跑通「复现 + 局部」 |

AIDE 在 RE-Bench 上承认 greedy 会陷入局部最优。因此后续允许**停滞后再升级**，而不是一上来 MCTS。

### 3.3 后续两年真正拉开奖牌率的机制

| 系统 | 采用 | 明确不采用 |
|---|---|---|
| **AIRA-dojo** | Agent = 算子集 × 搜索策略；**算子质量 > 搜索炫技** | 不把 MCTS 当 v1 主搜索 |
| **ML-Master** | \(R=-1\) 无 metric；修好 bug 可记 recovery | 完整 UCT 仅预留 |
| **MLEvolve** | Planner–Coder；Base / Stepwise / Diff；错误检索；跨枝融合作预留 | 不默认整页重写 |
| **RecHarness** | **Bandit 选方向、LLM 只在方向内写代码**；local → jump；Experiment Skill | 不让 LLM 自由选方向（其消融：LLM 选方向成功率 21.7% vs Thompson 47.9%） |
| **AgentX** | 论文 → 可插模块 → 接到统一 backbone | PaperImpl 接口已留，DeepFM/DCN 仍 reserved |
| **ARTS / Arbor** | 先判「实现错 vs 假设错」；假设–产物–证据–洞见 | Journal 四字段已对齐；诊断字段已留 |
| **FM-Agent 岛屿进化** | — | **不采用**：缺逐步 hypothesis/diff，对不上本题日志 |

公开 KuaiRand 论文**没有**给出本日期切分 + 5 域 numpy FM 上的 DeepFM/DCN/DIN/ESMM 对照表。Agent 必须自己搜，不能抄榜。

---

## 4. 架构决策（写报告时的「为什么这样设计」）

一句话：**硬编码的树搜索管「试哪条、何时停、失败怎么收」；LLM 只管「在指定方向上写一条可回滚的改动」。**

```
L6  Deliverable   提交门、top-k 钩子、消融表、dashboard
L5  Recsys prior  Thompson arms、local→jump、论文模块表
L4  Memory        Journal、Error Memory、Experiment Skill、Paper KB
L3  Search        默认 greedy；UCT / 2–4 并行预留
L2  Operators     Draft / Debug / Improve / Crossover / PaperImpl / Ablate
L1  Environment   隔离 trial、超时、禁区文件、kit evaluator
L0  Contract      官方 FM = s0、只做 ranking、干预 = 0
```

### 4.1 三套日志，三个读者

| 平面 | 读者 | 压缩？ |
|---|---|---|
| 评委日志 `journal.jsonl` | 交付物 3 | 不能少四字段 |
| 人看的观测面 `status.json` / `train.log` / `dashboard.html` | 随时确认 | 磁盘全量 |
| 塞进 LLM 的 \(\Sigma(T)\) | Planner / Coder | 必须摘要 |

只读 dashboard / `tail` 日志 **不算**人工干预。改代码、改臂、手动晋升才计数。

### 4.2 不能同时打开的冲突（避免被问「为什么不用 SOTA 搜索」）

- Greedy vs 完整 MCTS：默认 greedy（样本效率 + Feasibility）；停滞再升级。  
- 开放 idea 生成 vs 官方 FM 作 root：root 必须是 kit FM。  
- 岛屿进化 vs 逐步 diff 日志：保日志。  
- 并行多 GPU vs GPU-hours 计分：并行度有帽，默认 1。

---

## 5. 已实现 vs 仅接口

### 5.1 已实现并跑通

- 六层包结构、`config/default.toml`、git 仓库。  
- Orchestrator：Draft → Improve/Debug → 隔离执行 → `evaluate.py` → 晋升。  
- Dummy LLM：无 API key，对 `trial_config.json` 做原子改动（lr / l2 / loss）。  
- 官方 numpy FM 模板 + 可选 BPR 损失。  
- Journal / events / cost / heartbeat / status / dashboard。  
- Thompson arm 路由；`features` / `capacity` 标为 avoid。  
- Error Memory：归一化签名 + token overlap 检索。  
- 禁区：不能 patch kit `evaluate.py`。  
- 提交校验：`row_id` 连续、对齐、拒绝 NaN/Inf。  
- 18 个单元测试通过。

### 5.2 接口已留、模型未接

| 开关 | 配置 | 现状 |
|---|---|---|
| Error Memory | `[error_memory]` | 存取已工作；尚未用真实 traceback 大规模验证 |
| Jump DeepFM / DCNv2 | `[jump]` | 停滞 3 轮后解锁 `architecture`；`PaperImpl` 仍 `ReservedModuleError` |
| 多任务辅助头 | `[multitask]` | `main=long_view`，aux 含 `is_click`；默认 `enabled=false` |
| 2–4 并行 trial | `[parallel]` | `max_workers=4`；默认 `n_workers=1` |
| UCT | `agent/search/uct.py` | 函数在，策略未切换 |
| 统一 diff 应用到 `.py` | Coder | Dummy 走 `config_patch`；完整 patch parser 未做 |
| 真 LLM | `[llm] provider` | 已接 OpenAI 兼容接口；无 key 时回退 dummy |

外部资源只**引用路径**，不拷进仓库：`kuairand-starter-kit`、`KuaiRand-Pure/data`、`NISE`、`torch-rechub`、`CWM`。

---

## 6. 烟雾实验（可写进报告的「系统自检」，不可当成绩）

命令：`python -m agent run --smoke --max-iters 2`  
设定：train 截断 4000 行，1 epoch。Validation 仍是完整 124,909 行。人工干预 0。LLM token 0。

| Trial | 阶段 | Arm | 假设 | valid GAUC | nDCG@5 | primary | 结果 |
|---|---|---|---|---|---|---|---|
| `000_fm_baseline` | draft | draft | 复现官方 numpy FM | 0.5423 | 0.4889 | 0.5156 | 作为 root |
| `001_loss` | improve | loss | 将 logloss 换成 BPR，对齐排序指标 | 0.5734 | 0.4994 | **0.5364** | 晋升为 incumbent |

说明：

- 验证了「假设 → 配置 diff → 隔离训练 → kit 打分 → 仅 val 变好才晋升」。  
- **不能**与官方 FM valid 0.6016 比：训练数据被裁且只跑 1 epoch。  
- 全量对齐官方分数是下一阶段的第一件事。

过程中的工程修复：kit `evaluate.py` 返回 numpy `float32`，直接 `json.dumps` 会崩。现已在写入 `metrics.json` 前转为 Python float。这是 Robustness 的一个实例（失败可定位、可修、可留日志）。

### 6.1 全量复现官方 FM（Task 1，可写进报告）

命令：`python -m agent run --max-iters 1`（无 `--smoke`）  
数据：完整 train 1,141,112 / valid 124,909。人工干预 0。

| | GAUC | nDCG@5 | **primary** |
|---|---|---|---|
| Kit 公布 valid | 0.6674 | 0.5357 | **0.6016** |
| 本机 `000_fm_baseline` | 0.6671 | 0.5358 | **0.6015** |

差 0.0001，落在 5-seed std 0.0008 内。Early stop 在 epoch 11，约 48s。  
**结论：Task 1「复现官方 baseline」成立。** 实现上是把 kit `baseline.py` 的 FM 数学搬进 `templates/fm.py`，数据/划分/5 域特征/`evaluate.py` 均走 kit，不是另起模型。

### 6.2 全量 Dummy 搜索（合法但不充分）

在 `000` 晋升之后继续跑，`--max-iters 10`，实际 5 个 trial 后按 ε=0.002、N=3 收敛。

| Trial | 改动 | valid primary | 相对本机 FM | 晋升 |
|---|---|---|---|---|
| `000_fm_baseline` | 官方 FM，logloss，lr=0.001 | 0.6015 | — | 是 |
| `001_loss` | loss → BPR | 0.6039 | +0.0024 | 是 |
| `002_sequence` | Dummy 不会改序列，整次重跑 | 0.6039 | +0.0024 | 否（空操作） |
| `003_loss` | Dummy 又切回 logloss | 0.6015 | 0 | 否 |
| `004_optimizer` | 在 BPR 上 lr 0.001→0.0005 | **0.6041** | **+0.0026** | 是（incumbent） |

Incumbent：`loss=bpr`，`lr=0.0005`，其余与官方 FM 相同。  
相对 kit 公布 valid 0.6016：**+0.0025**（约 3σ）。**只是 validation，不是 hidden-test 成绩。**

暴露的 Dummy 缺陷（正是要接真 LLM 的原因）：

- `sequence` 臂不会写代码，却完整重训一遍。  
- 已涨分的 BPR 被下一轮 `loss` 臂切回去。  
- 搜索仍停在配置空间，没有改 pipeline 源码。

### 6.3 官方 baseline 在规则里的位置（避免写报告时说错）

题目同时要求两件事，不要混成「只能在官方代码上拧螺丝」或「完全不要官方代码」：

1. **必须复现官方 baseline 的 validation 分数**（Task 1）。官方 pipeline 是固定参照；队伍自制起步管线不能当评分基准。  
2. **必须自主迭代并超过它**。各阶段代码原则上由 agent 写，不限于改官方 FM 的超参。最终排名是 hidden test 相对官方 FM test primary 0.5946 的绝对提升。

Starter kit 的「从哪里开始改」指向 `baseline.py`，因此从官方 FM **起步是允许的**。本阶段做完了（1）和极弱的（2）。

---

## 7. 观测面（报告里可截图的路径）

```
run/
  status.json           当前快照（阶段、incumbent、预算、干预次数）
  journal.jsonl         评委四字段
  events.jsonl          trial_start / promoted / not_promoted / stagnation
  heartbeat.json        进程是否活着
  dashboard.html        只读页
  experiment_skill.md   蒸馏后的先验 + 最近节点
  summary.json          轮次汇总
  trials/<id>/train.log 完整训练 stdout
  trials/<id>/metrics.json
  trials/<id>/submission.csv
```

查看（零干预）：

```powershell
python -m agent status
Get-Content run\journal.jsonl -Tail 5
```

---

## 8. 下一阶段建议（给报告「Limitations / Next」用）

按对 Technical 分的 ROI：

1. ~~关掉 `--smoke`，全量复现官方 FM。~~ **已完成**（valid 0.6015 vs 0.6016）。  
2. ~~Dummy 下 loss=BPR。~~ **已完成**（valid 0.6039）；真 LLM 后应禁止无理由切回。  
3. **接真 LLM**：Planner 出 hypothesis，Coder 出 config_patch；空改动跳过，避免 `002` 那种空训。  
4. 让 Improve 开始改 `train.py` / 序列 / listwise，而不只拧 `trial_config`。  
5. 把 DeepFM / DCNv2 从 reserved 做成 PaperImpl，仅在 jump 解锁后启用。  
6. 多任务辅助头：主任务仍是 `long_view`。  
7. 收敛后跑消融，再 `finalize` 一次 hidden test。  
8. 从 Journal 自动生成 Devpost 表格：delta、干预次数、token、GPU-hours。

不要做：Pure 收敛前开 1K/27K；召回/ANN；用测试标签预训练；无日志进化搜索。

---

## 9. 报告可用的固定表述（可直接粘贴后改数字）

> 我们把本题建模为代码空间上的预算约束优化：解是一份可执行 ranking pipeline，目标函数是主办方 `evaluate.py` 的 validation primary。搜索策略硬编码为 Draft / Debug / Improve，方向由 Thompson 路由在推荐先验臂上选择，LLM 只在选中臂内提出一条可回滚改动。官方 Factorization Machine 是唯一评分参照，而不是队伍自制的 baseline。静态特征与 embedding 容量被组委会证伪，故搜索优先对准损失、序列与多任务。系统默认零人工干预；人可以随时只读 `run/status.json` 与 per-trial 日志，但不进入控制回路。

---

## 10. 本阶段未覆盖（写报告时不要夸大）

- Hidden test 一次提交（valid +0.0025 **不能**写成比赛成绩）。  
- 真 LLM 驱动的代码级改动与 token 成本（接口已接，跑数取决于 key）。  
- DeepFM / DCNv2 / DIN / ESMM 的可运行实现。  
- 并行 trial 的实测加速。  
- 1K / 27K bonus。
