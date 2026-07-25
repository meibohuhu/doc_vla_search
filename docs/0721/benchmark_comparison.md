# 四个 E2E 驾驶 Benchmark 对比：NAVSIM / nuScenes / Waymo WOD-E2E / Bench2Drive

> 用途：为「用 reasoning 提升 AutoVLA performance」这条主线做 benchmark 选型与报数规范。
> 配套：[AutoVLA_summary.md](AutoVLA_summary.md)（架构/配方）· [AutoVLA_nocot_sft_runbook.md](AutoVLA_nocot_sft_runbook.md)（跑通手册）
> 日期：2026-07-21

---

## 0. 总表

| | **NAVSIM** | **nuScenes** | **Waymo WOD-E2E** | **Bench2Drive** |
|---|---|---|---|---|
| 底层数据 | OpenScene（nuPlan 2Hz 重分发） | nuScenes v1.0-trainval | WOD-E2E | CARLA 仿真 |
| 真实 / 仿真 | 真实 | 真实 | 真实 | **仿真** |
| 开环 / 闭环 | **伪闭环**（4s 非反应式仿真） | **纯开环** | **纯开环** | **真闭环** |
| 相机 | 8 路（AutoVLA 学生用 3） | 6 路环视 | 8 路（AutoVLA 教师用 4，学生 3） | 取决于 agent |
| 主指标 | **PDMS**（v1）/ EPDMS（v2） | **L2 + Collision Rate** | **RFS**（人类 rater 偏好） | **DS = RC × IP** |
| 预测时域 | 5s 轨迹 / 4s 仿真 | 3s（6 帧 @2Hz） | 5s | 整条 route |
| 误差累积 | 部分（LQR 跟踪） | ❌ | ❌ | ✅ |
| 多解容忍 | 部分（看结果不看轨迹） | ❌ **单一 GT** | ✅ **显式多解** | ✅（看结果） |
| AutoVLA 代码支持 | ★★★★★ 全套 | ★★★★ 有 eval | ★★★ **只有预处理，无 RFS** | ❌ **完全没有** |
| 数据体量 | navtrain 445G | ~400G | **TB 级 ×3 份** | 无需下载 |
| 2026 现状 | 活跃（v2 leaderboard） | 活跃 | **无正式比赛**，leaderboard 开放 | 活跃 |

---

## 1. NAVSIM —— 主战场

### 评测方式不是纯开环

把预测的 5s 轨迹丢进 **4 秒非反应式仿真**：背景车按录制轨迹走（**不会对 ego 反应**），ego 由 **LQR 控制器**跟踪模型输出。

- ✅ 抓得到：轨迹物理上跟不跟得住、会不会撞、有没有出可行驶区域
- ❌ 抓不到：你的错误会不会引发别人的反应（背景车是"聋子"）

### 指标

**v1.1（AutoVLA 自带的版本）**：
```
PDMS = NC × DAC × (5·TTC + 5·EP + 2·C + 0·DDC) / 12
```
| 子项 | 含义 | 类型 |
|---|---|---|
| NC | 无责碰撞 No at-fault Collision | 乘子 {0, 0.5, 1} |
| DAC | 可行驶区域合规 | 乘子 {0, 1} |
| TTC | 碰撞时间在界内 | 权重 5 |
| EP | 自车推进度 Ego Progress | 权重 5，连续 [0,1] |
| C | 舒适度 | 权重 2 |
| DDC | 行驶方向合规 | **权重 0（被忽略）** |

**v2（当前 leaderboard）**：
```
EPDMS = NC × DAC × DDC × TL × (5·EP + 5·TTC + 2·LK + 2·C + 2·EC) / 16
```
新增 DDC（提为乘子）、TL（交通灯合规）、LK（车道保持）、EC（扩展舒适）。跑在 **navhard** 上（两阶段伪仿真：阶段 1 常规场景，阶段 2 考察**纠错行为**）。SOTA 量级约 **45–49 EPDMS**。

### ⚠️ 版本坑（必读）

**AutoVLA 仓库自带的 navsim 是 v1.1.0**（`navsim/setup.py:15`），不是 README 链接指向的 v2.0。

- 论文报的 **89.11 是 v1 PDMS on navtest**
- 想跟现在的 leaderboard 比 → 得自己把 navsim 升到 v2，是笔额外工作量
- **内部 A/B（no-CoT vs CoT）用 v1 PDMS 完全够**，先别折腾升级

另：`models/utils/score.py` 的 docstring 提到 "excluding the two_frame_extended_comfort metric"，那是 v2 的 EC 分项 —— **只是一句残留注释，仓库里没有任何 EPDMS 代码**。别被误导。

### AutoVLA 支持度：★★★★★

agent、metric cache、RFT 的 PDMS reward 全套齐备。这是唯一"开箱即用"的 benchmark。

---

### 1.5 ⚠️ PDMS 有很高的"免费底分"——不要把它当轨迹精度看

这一节是本地实测得出的，直接影响你怎么读自己的分数。

#### 实测标尺（同一批 navtest 场景）

| 模型 | **PDMS** | ADE | 自回归 action top-1 | 说明 |
|---|---|---|---|---|
| **恒速外推 agent** | **19.3** | — | — | **不看图、不看地图**，纯按当前速度直线外推（499 场景实测） |
| 我们的坏 ckpt（bf16 bug） | **60–62** | 2.65 m | 7.7% | 几乎没学会，见 [bf16_master_weights_bug.md](../0724/bf16_master_weights_bug.md) |
| 论文 action-only @100k | 71 | — | — | |
| 论文 SFT-only（CoT 混合，166k） | 80.54 | — | — | |
| 发布的 RFT ckpt | **89.06** | 0.23 m | 77.7% | 同批 1000 场景实测（全量 12,126 上 89.48） |
| 人类 GT 轨迹 | ~94 | 0 | 100% | 上限 |

#### 为什么 ADE 2.65 m 的模型还能拿 60 分

**PDMS 衡量的是"驾驶结果"，不是"轨迹像不像人类开的那条"。**
一条离 GT 2.65 米的轨迹，只要方向大致对、速度慢一点，照样可以不撞车、留在路上、开得平稳。

坏 ckpt 的分项拆解说明了一切：

```
NC  92.68   ← 不撞车
DAC 75.25   ← 75% 待在路面内
C  100.00   ← 舒适度满分（因为开得慢而稳）
TTC 84.97
EP  56.15   ← 唯一明显差的
```

实测该模型预测的 action token 均值是 **425**，而 GT 标签是 **601** ——
**系统性地更保守、更慢**。开得慢直接压低 EP，但同时抬高了 NC 和 C。

**它学到的是边缘分布（"大致沿路慢慢开"），没学到条件分布（"这个场景具体该怎么开"）。**
前者值约 60 分，后者才是 60 → 89 的那 29 分。

#### 三条实践含义

1. **60 分不代表"训出了能用的模型"**，它更接近"什么都没学会时的自然水位"。
   判断训练是否成功要看分项，尤其 **EP** 和 **DAC**。
2. **EP 和 DAC 是真正有区分度的两项。** NC / TTC / C 在当前方法上接近饱和
   （与 §5 引的开环↔闭环相关性研究一致：EP 是最强的闭环预测因子）。
3. **归零场景几乎全是 DAC=0。** 全量 12,126 场景里 554 个 PDMS=0，其中 **498 个（90%）是驶出可行驶区域**，
   只有 66 个是碰撞。DAC 是乘子，一旦为 0 整个 PDMS 直接归零 —— 直线外推在弯道出界就是典型成因。

#### 复现这个底分

恒速 agent **不需要 GPU**（纯 CPU 算术），可以和训练并行跑：

```bash
export NAVSIM_EXP_ROOT=/tmp/cv_eval NAVSIM_DEVKIT_ROOT=$PWD/navsim \
       OPENSCENE_DATA_ROOT=/data/autovla_data/nuplan \
       NUPLAN_MAPS_ROOT=/data/autovla_data/nuplan/maps \
       NUPLAN_MAP_VERSION=nuplan-maps-v1.0 PYTHONPATH=$PWD/navsim
python navsim/navsim/planning/script/run_pdm_score.py \
    train_test_split=<你的 scene_filter> \
    agent=constant_velocity_agent \
    metric_cache_path=/data/autovla_data/nuplan/navtest_metric_cache \
    experiment_name=cv_baseline
```

**建议每次报新模型的 PDMS 时，都把这个底分一起列出来**，读者才知道那个分数里有多少是"免费"的。

---

## 2. nuScenes —— 最便宜，但指标最不可信

### 数据与评测

1000 个场景 × 20s，6 路环视相机，2Hz 关键帧，700/150/150 划分。
纯开环：**L2 @1/2/3s + Collision Rate**。AutoVLA 用 UniAD 那套 `PlanningMetric`（`n_future=6` = 3s @2Hz），碰撞计算需要**额外下载一份预处理好的分割数据**（README 给了 Google Drive 链接）。

### ⚠️ 两个必须知道的坑

**(a) 两种统计口径，数字差很多。**
`tools/eval/nusc_eval.py` 会打印**两张表**：
- **ST-P3 口径**：累积平均（到 t 为止所有时刻的均值）
- **UniAD 口径**：瞬时值（就是 t 时刻）

同一个模型两张表数字差挺大。**论文数字对不上大多来自这里** —— 报数时必须写明用的哪种口径。

**(b) 指标本身有争议。**
nuScenes 开环规划指标被反复批评：光靠 ego status（当前速度/加速度）就能刷出接近 SOTA 的 L2（AD-MLP 那条线）。它跟真实驾驶能力的相关性很弱。

> **建议：当 sanity check，不要当主结论。**

### 真正的价值：唯一的免费 CoT 来源

DriveLM 的 `v1_1_train_nus.json` 是公开的，`dataset_utils/sft_dataset.py:196-225` 有现成的五段式解析分支（Scene Description / Critical Object / Reasoning on Intent / Best Driving Action）。

**这才是 nuScenes 在这个项目里的主要作用** —— 不是评测，是给你 CoT 数据。

### AutoVLA 支持度：★★★★

有 `nusc_eval.py`，但要单独建 conda env（`nuscenes-devkit` 与主环境依赖冲突，README 明说）。

---

## 3. Waymo WOD-E2E —— 长尾场景，成本最高

### 3.1 官方 split（不需要自己 clip 场景）

| split | segments | future GT | **rater 偏好标签** |
|---|---|---|---|
| **training** | **2,037** | ✅ | ❌ |
| **val** | **479** | ✅ | ✅ **只有这里有** |
| test | 1,505 | ❌ 封存 | ❌ 封存 |
| **合计** | **4,021** | | |

- 每 segment **20 秒**，8 路相机 **@10Hz**，轨迹标签 **@4Hz**，预测时域 **5s**，总计约 12 小时
- 全部是**出现频率 < 0.03% 的长尾场景** —— 这层筛选 Waymo 已经做完，**你不需要去原始数据里挖场景**

### 3.2 但需要两步转换（代码现成）

**Step A — 解图**：`tools/preprocessing/waymo_e2e_image_extraction.py`
tfrecord 不支持随机访问 → 先把帧解成 JPEG 落盘，并建 LMDB 索引 proto。

**Step B — 滑窗切样本**：`dataset_utils/preprocessing/waymo_e2e_dataset.py::scene_loader`
20s 太长，模型要的是「历史 + 5s 未来」的样本。按 `config/dataset/qwen2.5-vl-72B-waymo.yaml` 算：

```
frequency_ratio      = raw_images_freq / model_freq = 10 / 2 = 5    # 每 5 帧取 1
num_history_frames   = 5 × (model_his_frames - 1) + 1 = 5×8+1 = 41  # 4.1s 历史窗
num_fut_frames       = 5 × model_fut_frames = 5×10 = 50             # 5s 未来
scene_frame_interval = 20 帧 = 2 秒                                  # 滑窗步长
frame_shift          = 10
```

一个 200 帧（20s @10Hz）的 segment：
- **training**：`range(50, 201, 20)` → 约 **8 个样本/segment** → 2037 × 8 ≈ **1.6 万样本**
- **val / test**：需留出 5s 未来 → `range(50, 151, 20)` → 约 **6 个/segment**

> 论文里的 Waymo ~7.2k CoT 标注量是在此之上又筛过的子集，不是滑窗全量。

### 3.3 rater 偏好标签是什么

**要解决的问题：开环评测的「唯一 GT」假设。**

L2 / ADE 拿人类司机**实际开的那一条**轨迹当唯一正确答案。但驾驶本质是**多解**的 —— 前方慢车可以绕行也可以跟减速；黄灯可以冲也可以停。L2 把所有「合理但不同于录像」的行为一律判错，且有系统性偏置：**越保守、越贴录像的模型 L2 越好看**。

**rater 标签的做法**：人类标注员看场景，标出**多条**可接受的未来轨迹，每条配一个偏好分数。
→ GT 从「一条线」变成「**一个带权重的轨迹集合**」。

**RFS 计算（概念）**：在 **3s 和 5s** 两个时间点，每条 rater 轨迹周围划矩形**信任区（trust region）**；预测落在区内拿该轨迹的分，落在区外按距离**指数衰减**；最后取**所有 rater 轨迹中的最大值** —— 即「**匹配上任何一条人类认可的方案就算对**」。

> ⚠️ 精确公式（信任区尺寸、衰减系数、分数量纲）**实现前必须核对 `waymo-open-dataset` devkit 源码**，本文只到概念层。

**为什么对 reasoning 主线特别对口**：多解场景正是 CoT 应该发挥作用的地方 —— 模型需要**判断该选哪种方案**。L2 衡量不了这个（选了另一条同样合理的方案反而扣分），RFS 能。

### 3.4 ⚠️ 三个硬约束

**(a) training 没有 rater 标签 → RFS 无法直接用于训练。**
论文原文：rater feedback labels *"provided only for the validation set"*。
所以 RFS 只能当**评测指标**，本地只能在 val 的 479 个 segment 上算，样本量偏小。test 要拿数字只能提交 leaderboard。
这也解释了 `waymo_e2e_dataset.py:147` 那个 `if frame.preference_trajectories:` 判空 —— 字段是可选的。

**(b) 仓库里没有 RFS 实现。**
grep 全仓零命中。只在预处理时把 `preference_trajectories` / `preference_scores` 存进 JSON（`waymo_e2e_dataset.py:145-154`），**没有任何打分代码**。要么自己按 devkit 实现，要么走 leaderboard 提交。

**(c) 2026 年 Waymo 不办正式 challenge。**
官网原话："While we are not hosting formal Challenges in 2026, the Waymo Open Dataset leaderboards remain fully active."（2025 年的奖已发完）
→ 能提交拿数字，但**没有比赛名次可拿**。

### 3.5 磁盘代价（比想象的高）

两个放大因子：

**(a) 8 路相机全解，且缺一路就丢样本。**
`_make_scene_entry`（`waymo_e2e_dataset.py:322-335`）遍历完整的 8 个 `CAM_LIST`，**任何一路缺文件就 `return None` 丢弃整个 scene**。而 AutoVLA 教师标注只用 4 路（front / front_left / front_right / back）、学生模型只用 3 路。

> **可省 50%+ 磁盘的补丁**：`waymo_e2e_image_extraction.py` 的 `camera_mapping` 只留需要的 4 路，同时把 `REQUIRED_CAM_LIST` 从 `CAM_LIST` 改成那 4 路。否则为了 3 路能用，白存 5 路。

**(b) 数据落三份**：原始 tfrecord + 解出的 JPEG + LMDB。
作者把 LMDB 的 `map_size` 硬编码成 **1.5 TB**（`waymo_e2e_dataset.py:299`）—— 这是他自己给的量级提示。

再加上必须装 `waymo-open-dataset-tf-2-12-0`（整个 TensorFlow，会跟 `numpy==1.23.4` 打架）。**这条线的固定成本是四个里最高的。**

### AutoVLA 支持度：★★★（有预处理，无评测）

---

## 4. Bench2Drive —— 唯一真闭环，用 SimLingo

### AutoVLA 没有这部分代码

grep 全仓零命中 CARLA / Bench2Drive 相关实现。论文报了 Bench2Drive 数字，但**那部分没开源**。
→ **用 SimLingo 走这条线是唯一现实选择**，且本地 pipeline 已完全跑通（`eval-score` / `gen-eval-scripts` / `retry-audit` 三个 skill + bench2drive220 的 220 条 route）。

### 指标

```
DS (Driving Score) = RC × IP
```
- **RC** Route Completion：路线完成度
- **IP** Infraction Penalty：违规惩罚（碰撞、闯红灯、压线等的乘性折扣）
- **Success Rate**：本地定义为 DS == 100 的 route 占比

**唯一同时具备误差累积和 timeout 惩罚的 benchmark。** 开得太慢/太保守会被直接扣分 —— 这一点在下一节至关重要。

---

## 5. ⚠️ 交叉发现：开环指标能不能预测闭环

有一篇专门研究这个问题的论文（[arXiv:2605.00066](https://arxiv.org/html/2605.00066)，NAVSIM ↔ Bench2Drive 相关性研究），结论直接影响实验设计：

| 发现 | 内容 |
|---|---|
| 总体相关性 | PDMS 与 Bench2Drive DS 的 **Spearman ρ = 0.90** |
| **但是** | **非单调，存在明显的排名反转** |
| 最强预测因子 | **EP（推进度）**，明显强于其他单项 |
| 已饱和的项 | **NC、TTC、Comfort** 在当前方法上几乎没有区分度 |
| **最要命的** | 「优化安全指标、牺牲 progress」的方法：**开环好看，闭环因 timeout 和慢速惩罚掉分** |

### 对 reasoning 主线的直接含义

**如果 CoT 让模型变保守（更愿意等、开得更慢），NAVSIM PDMS 可能涨，Bench2Drive DS 反而跌。**
而"变保守"恰恰是 CoT 最容易产生的副作用。

→ **四个 benchmark 一起报不是凑数，是必要的交叉验证。**
→ **报 PDMS 时务必单独把 EP 分项拎出来看**，这是最早能发现"保守化"的信号。

---

## 6. 推进顺序

| 阶段 | Benchmark | 理由 |
|---|---|---|
| **1** | **NAVSIM v1 PDMS** | AutoVLA 原生、代码全套、迭代最快。**主要 A/B（no-CoT baseline vs CoT）都在这里做** |
| **2** | **Bench2Drive**（SimLingo） | 已有 pipeline，且是唯一闭环 —— 专门用来抓上面说的"保守化"副作用 |
| **3** | **nuScenes** | 做 DriveLM CoT 时数据本来就下了，顺手。当 sanity check，注明口径 |
| **4** | **Waymo WOD-E2E** | 成本最高（TB×3 + TF 依赖 + 自己实现 RFS），2026 无比赛 |

### 关于 Waymo 的排序，有个反向理由

它的独特价值是**长尾 + 多解容忍**。如果你的核心假设是「**CoT 在困难/罕见/多解场景才有用**」，那 WOD-E2E 是验证这个假设**最对口**的数据集 —— 其他三个都做不到这件事：

- nuScenes 单一 GT，选了另一条合理方案反而扣分
- NAVSIM 看仿真结果，对"哪种方案"不敏感
- Bench2Drive 是仿真长尾，不是真实长尾

→ **它值得做，但应该在用 NAVSIM 证明 reasoning 有效之后**，作为"reasoning 在长尾上收益更大"的佐证，而不是首发。

---

## 7. 报数 Checklist

每次报数字前对一遍，避免不可比：

- [ ] **NAVSIM**：写明 **v1 PDMS** 还是 **v2 EPDMS**；写明 split（navtest / navhard）；**单独列出 EP 分项**
- [ ] **NAVSIM**：推理时**开 CoT、关 LoRA**（发布 ckpt 已 merge）；确认 navsim / nuplan-devkit 版本 + metric cache 正确（issue #48：搞错会从 89.11 掉到 83.69）
- [ ] **nuScenes**：写明 **ST-P3 口径**还是 **UniAD 口径**；写明 collision 用的哪份 seg data
- [ ] **Waymo**：写明是 **val 本地 RFS**（479 segments）还是 **test leaderboard**；写明 RFS 实现来源
- [ ] **Bench2Drive**：写明 route 集合（bench2drive220）、是否含 retry、Success Rate 的定义（DS==100）
- [ ] **全部**：baseline 与实验组**配方必须一致**（尤其 ViT 冻结/解冻、全参/LoRA —— 见 runbook §7.2 / §8）

---

## 8. 参考

- [NAVSIM PDMS 定义](../../navsim/docs/metrics.md)（仓库内）
- [WOD-E2E: Waymo Open Dataset for End-to-End Driving in Challenging Long-tail Scenarios (CVPR 2026)](https://arxiv.org/abs/2510.26125)
- [Waymo Open Dataset Challenges（2026 无正式比赛）](https://waymo.com/open/challenges/)
- [Do Open-Loop Metrics Predict Closed-Loop Driving? NAVSIM ↔ Bench2Drive 相关性研究](https://arxiv.org/html/2605.00066)
- [Generalized Trajectory Scoring for End-to-end Multimodal Planning（EPDMS / navhard）](https://arxiv.org/pdf/2506.06664)
