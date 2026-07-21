# AutoVLA 架构 · 输入 · 训练配方总结

> A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning (NeurIPS 2025, UCLA)
> Paper: [arXiv:2506.13757](https://arxiv.org/abs/2506.13757) · Code: [ucla-mobility/AutoVLA](https://github.com/ucla-mobility/AutoVLA) · Ckpt: [Zewei-Zhou/AutoVLA](https://huggingface.co/Zewei-Zhou/AutoVLA)
> 本文档结论综合了代码 + 论文 + 作者在 GitHub closed issues 的一手回复。

---

## 1. 一句话概括

AutoVLA 是**统一自回归 VLA**：在同一序列里先（可选）生成 **CoT 推理文本**、再生成**离散动作 token**，解码成未来 5s 轨迹。SFT 学会"会思考"，RFT(GRPO) 用长度惩罚学会**自适应 fast/slow 切换**。

---

## 2. 架构

- **Backbone**：`Qwen2.5-VL-3B-Instruct`。**Vision encoder 全程冻结**（SFT 和 RFT 都冻，作者澄清架构图有笔误），只训 LLM。
- **动作 tokenizer**：轨迹 **K-disk 聚类**（非普通 k-means）成 **2048** 词表，映射为 `<action_i>` 特殊 token（`action_start_id=151665`）。`num_poses=10 × interval=0.5s = 5s` 未来轨迹。详见 §2.5。见 [models/action_tokenizer.py](AutoVLA/models/action_tokenizer.py)。
- **统一自回归**：输出 = `[CoT 文本(可选)] + [动作 token]`，一次前向"推理+规划"一起做。见 [models/autovla.py](AutoVLA/models/autovla.py)。
- **CoT 标注模型（离线，非主模型）**：`Qwen2.5-VL-72B(-AWQ)`，只用于给训练集造推理数据。
- **框架**：PyTorch Lightning + FSDP(FULL_SHARD, bf16)。**SFT 不用 LoRA；RFT 用 LoRA** 省显存。

### 2.5 Action Token 预处理（轨迹离散化 → 动作词表）
脚本 [action_token_cluster.sh](AutoVLA/scripts/action_token_cluster.sh) → [tools/action_token/action_token_cluster.py](AutoVLA/tools/action_token/action_token_cluster.py)，参数 `num_cluster=2048`、`n_trajs=2048000`、`tol_dist=0.05`、`--data_path ./dataset/nuplan`。

| 步骤 | 内容 |
|------|------|
| ① 收集轨迹段 | 遍历所有预处理 JSON 的 `gt_trajectory`，每步 pose 用 `transform_to_local`+`wrap_angle` 转成相对上一帧的**局部位姿 (Δx, Δy, Δheading)**，采样最多 ~204.8 万段 |
| ② 转车体轮廓 | 用 ego 尺寸 **width=2.0m, length=4.8m**，把轨迹点转成车体 **4 角点多边形**（`cal_polygon_contour`）——距离度量考虑朝向+占地，非仅中心点 |
| ③ **K-disk 聚类** | 贪心 disk-covering（**非普通 k-means**）：反复选点，把"4 角点平均距离 < `tol=0.05m`"的邻居归一簇取均值为簇心并移除，得 **2048 簇**；第 0 个固定为 `[0,0,0]`（静止 token） |
| ④ 存 codebook | 簇心转 contour 存 `codebook_cache/agent_vocab.pkl` 的 `token_all['veh']`，**shape (2048, 6, 4, 2)** = 2048 token × 每个 6 小步运动基元 × 4 角点 × xy；另存 `.jpg` 可视化 |

**如何使用**（[action_tokenizer.py](AutoVLA/models/action_tokenizer.py)）：
- 往 Qwen tokenizer 加 2048 个 special token `<action_0..2047>`（从 `action_start_id=151665` 起）。
- **Encode（训练标签）**：GT 未来轨迹按段匹配到**最近 codebook 项** → `<action_i>` 序列作监督目标。
- **Decode（推理）**：预测 `<action_i>` → 查表得 6-step 段 → `rollout` **自回归拼接**（每段按当前位姿 `transform_to_global`，段终点作下一段起点）→ 还原连续轨迹（5s/10 pose）。

**要点**：词表 2048、一次性预生成、训练/推理**全程固定**；本质是"**轨迹基元库**"式离散化，把连续规划变成"从固定动作库自回归选 token"，契合 LLM next-token 预测；codebook 只针对 `veh`(ego)，从 **nuPlan** GT 轨迹聚出。
⚠️ 小坑：发布版聚类脚本主循环 `trajs` 按单步累积，与 codebook 的 6-step 段维度看似不完全对齐——实跑用仓库自带 `agent_vocab.pkl`，若自行重聚类需先核对段长。

---

## 3. 模型输入（⚠️ 区分"学生模型" vs "教师标注"）

### 3.1 学生模型（SFT 训练/推理，真正的模型输入）— [`autovla.py::get_prompt`](AutoVLA/models/autovla.py)
| 输入 | 形式 | 说明 |
|------|------|------|
| **多视角相机视频（历史）** | video | **仅 3 个前向相机**(front / front-left / front-right)，各 **4 帧 @ 2Hz**（过去 ~2s） |
| **当前 ego 状态** | 文本 | 当前 velocity、当前 acceleration |
| **驾驶指令** | 文本 | driving command |
| **CoT 提示** | 文本 | Scene → Critical Objects → Prediction → Intent → Action |

### 3.2 教师标注（造 CoT 数据用，72B 的输入）— [`nuplan_dataset.py`](AutoVLA/dataset_utils/preprocessing/nuplan_dataset.py)
- 用 **4 个相机**(含 back)、并额外含一句 **`The ego vehicle behavior in the past 4s is {his_ego_action}`**（由历史轨迹算出）。

### 3.3 关于 Ego History（重要澄清，issue #39 / #44）
- ✅ **视觉时序历史**（多帧视频）：学生模型**有用**。
- ❌ **显式的历史动作/历史轨迹文本**：**发布版 NAVSIM 训练默认没有喂给学生模型**（学生只看到当前速度/加速度+指令）。论文 §3.1 写的 "historical actions" 是 Waymo Challenge 版本用的，NAVSIM 发布版未包含。
- **想加历史动作**，作者给了 3 种方式：① `get_action_instruction` 转成 `past 4s behavior` 文本；② 直接喂历史 waypoint 坐标文本；③ 用同一套 action token 表示历史动作序列。
- ⚠️ **相机不一致**（issue #44）：教师用 4 相机（含 back），学生只用 3 前向相机。作者称**去掉 back view 反而减少幻觉、效果更好**，是有意为之。

> **对比 SimLingo**：SimLingo 单帧+当前速度、完全无历史；AutoVLA 学生模型有**视觉多帧历史**但**默认无显式 ego-motion 历史文本**（可选加）。

---

## 4. 输出
- **[slow]** CoT：场景描述 → 关键物体 → 行为预测 → 意图 → 最佳动作。
- **动作 token 序列** → 解码为未来 **5s / 10 poses** 轨迹。

---

## 5. 训练配方（Training Recipe）

### 5.1 SFT —— 可训 CoT，也可训 no-CoT（issue #41 作者确认）
| `model.use_cot` | 训练数据 | 得到的行为 |
|---|---|---|
| **`false`** | 纯 action-only | **100% fast thinking**（不出 CoT） |
| **`true`** | CoT + 非 CoT **混合** | 学会生成 CoT，并**初步具备** fast/slow 切换；调推理参数后，简单场景可跳过 CoT |

- 真正的**自适应 fast/slow** 不是 SFT 给的，而是 **RFT 用 CoT 长度惩罚**训出来的（见 5.2）。
- **SFT 损失**（issue #23 / #33 作者澄清）：
  `L_SFT_i = w_i · (L_LM,i + λ_a · L_action,i)`
  - `L_LM`：对**所有** target token（推理+动作）的标准 next-token loss。**无 CoT 样本时 `L_LM ≈ L_action`**。
  - `L_action`：给 CoT 样本额外加的动作辅助损失——因为长推理文本会稀释动作 token 的梯度，而动作需要更高精度。
  - `w_i`(=λ_cot)= **40**（CoT 样本）/ 1（其它）；`λ_a = 1`。CoT 样本稀少，故加权放大。
  - ⚠️ 已知坑：若不加 `L_action`，SFT 后模型可能**只输出 CoT 文本、不出动作 token**（issue #23 社区反馈）。

### 5.2 RFT —— GRPO 自适应推理
- **用 LoRA**（省显存）。参考模型 = SFT checkpoint（冻结）。
- 流程：采样 → 算 reward → 组内归一化 advantage → policy loss + KL(到参考模型)。见 [`GRPOAutoVLA`](AutoVLA/models/autovla.py)。
- **Reward = NAVSIM PDMS**（[`PDM_Reward`](AutoVLA/models/utils/score.py)），需**预先构建 metric cache**（`navsim/docs/cache.md`）。
- **CoT 长度惩罚** → 只在"推理收益 > 惩罚"时保留 CoT → 自适应。RFT 后 navtest 绝大多数是 fast thinking（issue #46）。

### 5.3 关键超参（issue #33 确认 + config）
| 项 | 值 |
|---|---|
| Optimizer | AdamW |
| LR | 1e-5（config 里 2e-5；作者建议 **2e-5 收敛更快更好**） |
| weight decay | 0.01 |
| LR schedule | warmup 500 步 → step decay γ=0.98 / 2000 步 |
| grad clip | 1.0 |
| 精度 | bf16, FSDP FULL_SHARD |
| batch | per-GPU 1 × grad-accum 4 × 8 GPU = **effective 32** |
| epochs | 5 |
| backbone | **vision 冻结, LLM 全参训练（无 LoRA）** |
| **显存** | **~30GB / GPU（8×L40S）**（issue #51）→ 单卡 48G 可行 |

### 5.4 数据构建 & 混合训练（issue #33 / #35）
- SFT 与 RL **共用**同一套预处理→统一 JSON→单一 dataloader；**区别**：RL 需额外接 NAVSIM metric cache 算 reward。
- **SFT = nuPlan + nuScenes 混合训练**（配置名就叫 `mix-sft`）。混合方式：**concat → shuffle → 随机采样**（作者原话），多源统一 JSON 走同一 dataloader。scaling 数据有提升，**论文报告结果用的就是混合集**；也可改成只用 nuScenes 或只用 nuPlan。
  ```yaml
  # config/training/qwen2.5-vl-3B-mix-sft.yaml —— json_dataset_path 是列表
  data.train.json_dataset_path:
    - ./dataset/nuplan/trainval
    - ./dataset/nuscenes/nuscenes_train
  data.train.sensor_data_path:
    - ./dataset/nuplan/sensor_blobs/trainval
    - null                    # nuScenes 图像走绝对路径，故为 null
  data.val: navtest（验证只在 nuPlan navtest 上做 PDMS）
  ```
- **RFT(GRPO) 只用 nuPlan**：reward = NAVSIM PDMS，需 metric cache，nuScenes 无对应闭环打分。
- SFT:RL 比例是开放问题，作者目前只做随机选取。

### 5.5 训练/评测脚本
`run_sft.sh` → `run_rft.sh` → navsim PDMS 评测。
- **评测复现要点**（issue #48）：推理时**开 CoT、关 LoRA**（发布 ckpt 已 merge），用对 navsim/nuplan-devkit 版本 + 正确 metric cache，否则会像有人一样只跑到 83.69 而非 89.11。

---

## 6. 数据集：共 4 个
| # | 数据集 | 角色 | CoT 标注 |
|---|--------|------|------|
| 1 | **nuPlan**(NAVSIM) | 开环 + RFT reward(PDMS) | ~45.6k |
| 2 | **Waymo E2E**(WOD-E2E) | 开环长尾 | ~7.2k |
| 3 | **nuScenes** | 开环 + QA | 复用 DriveLM |
| 4 | **CARLA**(Bench2Drive) | 闭环 | 复用 DriveLM |

- 自标 CoT ≈ **52.8k**；**Reasoning data 未公开**（README = TBD）。
- **CoT 标注很慢**（issue #38）：72B 标全 nuPlan ≈ **10 天**（bs=1 + 裸 HF generate）；提速靠 FlashAttention + 多卡多实例并行 / 换 vLLM / 换 API。且生成的 CoT **常不严格遵守模板**，需后处理过滤。

---

## 7. 关键文件 & Issues 速查
| 功能 | 路径 / Issue |
|------|------|
| 主模型 / GRPO / get_prompt | [models/autovla.py](AutoVLA/models/autovla.py) |
| 动作离散化 | [models/action_tokenizer.py](AutoVLA/models/action_tokenizer.py) |
| 教师 CoT 标注 | [cot_annotation_model.py](AutoVLA/dataset_utils/preprocessing/cot_annotation_model.py) · [cot_prompts.py](AutoVLA/dataset_utils/preprocessing/cot_prompts.py) |
| nuPlan/Waymo 输入 | [nuplan_dataset.py](AutoVLA/dataset_utils/preprocessing/nuplan_dataset.py) · [waymo_e2e_dataset.py](AutoVLA/dataset_utils/preprocessing/waymo_e2e_dataset.py) |
| 训练配置 | [config/training/](AutoVLA/config/training/) |
| CoT/no-CoT & 冻结 & LoRA | issue #41 |
| SFT loss | issue #23 / #33 |
| 显存 | issue #51 |
| 复现 PDMS | issue #48 |
| 历史动作 | issue #39 |
| 4vs3 相机 | issue #44 |
| CoT 恒定 = 正常 | issue #46 |
| 标注耗时 10 天 | issue #38 |

---

## 8. 总结

- **SFT 确实可训 CoT / no-CoT**：`use_cot=false` → 纯动作(100% fast)；`use_cot=true` → CoT+非CoT 混合，初步会切换。**自适应 fast/slow 靠 RFT 的长度惩罚**成型。
- 配方 = **SFT（vision冻结 + 全参LLM + 双损失 L_LM/L_action + w=40 加权 CoT）→ RFT（LoRA + GRPO + PDMS reward + CoT长度惩罚）**。
- 输入 = **3 前向相机多帧视频 + 当前速度/加速度 + 指令（+可选历史动作）**；输出 = CoT + 动作token→5s轨迹。
- 4 个数据集（nuPlan/nuScenes/Waymo/CARLA），CoT 数据需自标且未公开。
