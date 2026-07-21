# AutoVLA no-CoT SFT 跑通手册（本机 8×A100-80G）

> 目标：在没有 CoT 数据的前提下，跑通 AutoVLA 的 no-CoT SFT，拿到一个干净的 **baseline**，为后续「用 reasoning 提升 performance」做对照。
> 配套文档：[AutoVLA_summary.md](AutoVLA_summary.md)（架构/配方/issue 速查）· [benchmark_comparison.md](benchmark_comparison.md)（四个 benchmark 选型与报数规范）
> 日期：2026-07-21

---

## 0. TL;DR 执行顺序

| # | 步骤 | 耗时量级 | 阻塞关系 |
|---|------|---------|---------|
| 1 | 建 conda 环境 + 装依赖 | ~30 min | — |
| 2 | 下 Qwen2.5-VL-3B（~7G） | ~10 min | — |
| 3 | **后台**下 nuPlan maps + mini（~91G） | 数小时 | 可与 1/2/4 并行 |
| 4 | 改 3 处 config + 新建 no-CoT SFT yaml | ~10 min | — |
| 5 | 预处理 mini → JSON | 1–3 h（8 核瓶颈） | 需 1+3 |
| 6 | 单卡 2000 样本冒烟 | ~30 min | 需 5 |
| 7 | 8 卡全量 SFT | 数小时–1 天 | 需 6 |

**不要做的事**：不要下 trainval（>2000G）、不要下 lidar、不要下 72B、不要重跑 action codebook、不要装 flash-attn/TF。理由见下文各节。

---

## 1. 环境安装

### 1.1 基础

```bash
cd /home/nvidia/workspace/doc_drive_search/other_repo/AutoVLA

conda env create -f environment.yml
conda activate autovla_codeclean          # ⚠️ 不是 README 里写的 autovla
pip install -e . --no-warn-conflicts
cd navsim && pip install -e . --no-warn-conflicts && cd ..
```

**坑 ①：环境名和 README 不一致。** `environment.yml` 里 `name: autovla_codeclean`，README 写的是 `conda activate autovla`。以 yml 为准。

### 1.2 `install.sh` 只需要执行 1/4

`install.sh` 里 4 条命令，no-CoT 路径只有最后一条有用：

| 命令 | 要不要 | 理由 |
|---|---|---|
| `pip install flash-attn==2.7.4.post1` | ❌ **跳过** | [autovla.py:480](../../models/autovla.py) 的 `from_pretrained` 没传 `attn_implementation`，默认走 **sdpa**。装它纯粹是编译地狱 |
| `pip install waymo-open-dataset-tf-2-12-0` | ❌ **跳过** | 只有 Waymo 预处理要。它会拖一整个 TF 进来跟 `numpy==1.23.4` 打架 |
| `pip install autoawq --no-deps` | ❌ **跳过** | 只有 72B CoT 标注要 |
| `pip install --upgrade typing_extensions` | ✅ **执行** | |

所以直接：
```bash
pip install --upgrade typing_extensions
```

### 1.3 环境变量

写进 `~/.bashrc` 或 conda 的 `activate.d`：

```bash
export AUTOVLA_ROOT=/home/nvidia/workspace/doc_drive_search/other_repo/AutoVLA
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/data/autovla_data/nuplan/maps"
export NAVSIM_DEVKIT_ROOT="$AUTOVLA_ROOT/navsim"
export NAVSIM_EXP_ROOT="$AUTOVLA_ROOT/navsim/exp"
export OPENSCENE_DATA_ROOT="/data/autovla_data/nuplan"
export TOKENIZERS_PARALLELISM=false
```

### 1.4 预训练模型

```bash
bash scripts/download_qwen.sh          # 只下 3B，约 7G，落到 ./Qwen2.5-VL-3B-Instruct
```

**72B 完全不用下。** 见 §3.1 —— no-CoT 预处理里那个 processor 的产物是被丢弃的。

---

## 2. 数据集

### 2.1 磁盘规划（**关键**）

本机磁盘现状：

| 挂载点 | 容量 | 可用 | 可写 | 用途 |
|---|---|---|---|---|
| `/` | 193G | **21G** | ✅ | repo 在这里 —— **数据绝对不能放这** |
| `/data` | 19T | **6.7T** | ✅ | ✅ **放数据集** |
| `/lp-dev` | 6.2T | 6.1T | ❌ 无权限 | 不能用 |

```bash
mkdir -p /data/autovla_data/nuplan
ln -s /data/autovla_data $AUTOVLA_ROOT/dataset
```

### 2.2 下哪个 split

| 方案 | 下载量 | 何时需要 |
|---|---|---|
| **maps** | **971 MB**（压缩） | 必需 |
| **mini** metadata + camera | 0.53G + **90G** | ✅ **第一步就下这个** |
| test metadata + camera | 0.48G + 128G | 做 navtest PDMS 评测时 |
| navtrain sensor | 445G（无 history 版 300G） | 出正式数字时 |
| trainval 全量 | **>2000G** | ❌ 永远不要 |

**省一半的关键**：`VlaAgent.get_sensor_config()` 里 `lidar_pc=False`（[vla_agent.py:150](../../navsim/navsim/agents/vla_agent.py)），AutoVLA 全程不用点云。
→ **把 `navsim/download/download_mini.sh` 里第二个 lidar 循环整段删掉。**

### 2.3 目录结构是硬约束

[nuplan_dataset.py:42-43](../../dataset_utils/preprocessing/nuplan_dataset.py) 把配置里 `dataset_path` 的字符串 `placeholder` 分别替换成 `navsim_logs` / `sensor_blobs`：

```python
data_path      = self.data_path.replace('placeholder', 'navsim_logs')
sensor_blobs   = self.data_path.replace('placeholder', 'sensor_blobs')
```

所以最终必须长成：

```
/data/autovla_data/nuplan/
├── maps/                      <- nuplan-maps-v1.1.zip 解开后把 nuplan-maps-v1.0 改名成 maps
├── navsim_logs/
│   └── mini/                  <- openscene_metadata_mini.tgz 里的 meta_datas
├── sensor_blobs/
│   └── mini/                  <- openscene_sensor_mini_camera_*.tgz 解出来的
└── mini_nocot/                <- §3 预处理产物（JSON）
```

⚠️ 官方 `download_mini.sh` 解出来的目录名是 `mini_navsim_logs` / `mini_sensor_blobs`（平铺），**跟上面的两级结构不一样**，下完要手动搬。

---

## 3. 预处理（生成 no-CoT JSON）

### 3.1 改 `config/dataset/qwen2.5-vl-72B-nuplan.yaml`

```yaml
pretrained_model_path: ./Qwen2.5-VL-3B-Instruct    # 原: ./Qwen2.5-VL-72B-Instruct-AWQ
dataset_path: ./dataset/nuplan/placeholder/mini    # 原: .../placeholder/mini（确认 split 名）
scene_filter: ./navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navmini.yaml
```

**为什么能把 72B 换成 3B**：no-CoT 路径下 processor 只用来 `apply_chat_template` 拼一个字符串，而 [nocot_sample_generation.py::process_sample](../../tools/preprocessing/nocot_sample_generation.py) **根本不保存那个字符串** —— 输出 JSON 只有 `token / velocity / acceleration / instruction / gt_trajectory / his_trajectory / *_camera_paths`。所以 processor 是谁无所谓。

### 3.2 跑

```bash
python tools/preprocessing/nocot_sample_generation.py \
  --config dataset/qwen2.5-vl-72B-nuplan \
  --output_dir ./dataset/nuplan/mini_nocot \
  --num_workers 8                                  # ⚠️ 默认 32，本机只有 8 核
```

**坑 ②：本机只有 8 核 CPU。** 默认 `--num_workers 32` 会疯狂抢占。

**坑 ③（性能，值得打补丁）**：`NuplanCoTAnnotationDataset.__getitem__` 对每个样本会把 **16 张图 base64 编码**（`process_image_input`）拼进 messages，再跑 `process_vision_info` —— 然后这些结果在 no-CoT 路径下全被丢掉。在 8 核机器上这是压倒性的开销。**加个 `cot=False` 开关跳过 messages 构建，预处理能快数倍。**

### 3.3 切 train/val

`SFTDataset` 只读 `json_dataset_path` + `sensor_data_path` 两个 key，`scene_filter` / `metric_cache_path` 在 SFT 里**完全不被使用**（那两个是 RFT / PDMS 评测才要的）。所以 **val 不需要下 test split**，直接把 mini 的 JSON 切开即可：

```bash
cd ./dataset/nuplan
mkdir -p mini_nocot_val
ls mini_nocot/*.json | shuf -n 500 | xargs -I{} mv {} mini_nocot_val/
```

### 3.4 Action codebook —— **不要重跑**

仓库已自带 `codebook_cache/agent_vocab.pkl`（1.1 MB，2048×6×4×2）。
`scripts/action_token_cluster.sh` 的主循环按单步累积、与 codebook 的 6-step 段维度看似不对齐（见 summary §2.5 的 ⚠️）。**直接用仓库自带的。**

---

## 4. SFT 配置

### 4.1 新建 `config/training/qwen2.5-vl-3B-nuplan-nocot-sft.yaml`

仓库里**没有** no-CoT 的 SFT config（只有 `mix-sft`=CoT 和 `grpo-cot`），必须自己写：

```yaml
name: qwen2.5-vl-3B-nuplan-nocot-sft
description: AutoVLA no-CoT SFT baseline on nuPlan mini

model:
  use_cot: false                    # ★ 关键
  pretrained_model_path: ./Qwen2.5-VL-3B-Instruct
  train_vision_backbone: false      # 保持原配方，见 §6
  train_lm_backbone: true
  codebook_cache_path: "codebook_cache/agent_vocab.pkl"
  trajectory:
    num_poses: 10
    interval_length: 0.5
    time_horizon: 5.0
  tokens:
    action_start_id: 151665
    ignore_index: -100
    assistant_id: [151644, 77091]
  video:
    min_pixels: 109760              # 注意：训练路径实际不读这个，见 §5.1
    max_pixels: 109760

training:
  batch_size: 1
  learning_rate: 2.0e-5
  epochs: 1                         # 冒烟；正式设 5
  num_workers: 2                    # ⚠️ 8 核机器，别用 4
  weight_decay: 0.01
  lr_warmup_step: 500
  lr_step_frequency: 2000
  lr_step_gamma: 0.98
  train_sample_size: 2000           # 冒烟；正式设 null
  accumulate_grad_batches: 4

data:
  train:
    json_dataset_path: ./dataset/nuplan/mini_nocot
    sensor_data_path: ./dataset/nuplan/sensor_blobs/mini
  val:
    json_dataset_path: ./dataset/nuplan/mini_nocot_val
    sensor_data_path: ./dataset/nuplan/sensor_blobs/mini

inference:
  batch_size: 1
  num_workers: 2
  sample:
    max_length: 2048
    temperature: 0.01
    top_k: 0
    top_p: 1.0
```

### 4.2 跑

```bash
# 冒烟（单卡）
CUDA_VISIBLE_DEVICES=0 python tools/run_sft.py --config training/qwen2.5-vl-3B-nuplan-nocot-sft

# 全量（8 卡，改 train_sample_size=null / epochs=5 之后）
python tools/run_sft.py --config training/qwen2.5-vl-3B-nuplan-nocot-sft
```

**坑 ④**：`trainer(devices='auto')` 会直接吃满 8 卡。冒烟阶段务必加 `CUDA_VISIBLE_DEVICES=0`。

有效 batch = `1 × 4 (accum) × 8 (GPU) = 32`，与论文一致。

---

## 5. 数值参考 / 你该期待什么

### 5.1 视觉输入到底多大

Qwen2.5-VL-3B：`patch_size=14`，`spatial_merge_size=2` → `factor=28`；`temporal_patch_size=2`。
对 nuPlan 的 1920×1080，按 `smart_resize` 计算：

```
beta = sqrt(1080*1920 / 109760) = 4.347
h = floor(1080/4.347/28)*28 = 8*28 = 224
w = floor(1920/4.347/28)*28 = 15*28 = 420
```

| 项 | 值 |
|---|---|
| 每帧分辨率 | **224 × 420**（原图面积的 **1/22**） |
| 每帧 token | 8 × 15 = **120** |
| 每路相机（4 帧，时序 2 合 1） | **240** |
| 3 路相机总视觉 token | **720** |

参照：InternVL2 一个 448×448 tile = 256 token。**AutoVLA 每帧只有半个 tile 的信息量。**

> **不一致点（当前无害，改前必看）**：训练路径用的是硬编码的 `28*28*128 = 100352`（[sft_dataset.py:266](../../dataset_utils/sft_dataset.py)），config 的 `109760` 只在推理/RFT 的 `get_prompt` 生效（[autovla.py:531](../../models/autovla.py)）。两个预算在 16:9 输入下因为 floor 取整**恰好都落到 224×420**，所以目前没差。但改分辨率或换非 16:9 数据源时这俩会分叉 —— **改 config 记得同步改代码**。

### 5.2 监督信号极稀疏（别看到 loss 掉得慢就慌）

`use_cot=false` 时 `has_cot` 恒为 False → [autovla.py:358](../../models/autovla.py) 的 `loss*40 + action_loss` 分支**永不触发**，loss 就是纯 LM loss。而 assistant 段里除了 10 个 action token，其余全是固定模板（`<answer>\nThe final output action is: `），几百步就学死了。

**净效果：每个样本跑 720 视觉 token + 3B 前向，只换来约 10 个有效监督 token。** 数据效率天生就低 —— 这就是原配方要 5 epochs 的原因。

### 5.3 显存

作者在 8×L40S 上 ~30GB/卡（issue #51）。A100-80G 富余很多。

---

## 6. 代码坑清单

| # | 位置 | 内容 |
|---|---|---|
| ① | `environment.yml` | 环境名是 `autovla_codeclean`，不是 README 的 `autovla` |
| ② | `install.sh` | 4 条只需 1 条；flash-attn 不需要（默认 sdpa） |
| ③ | `download_mini.sh` | 删掉 lidar 循环，省一半；解出的目录名要手动搬成两级结构 |
| ④ | `nocot_sample_generation.py` | `--num_workers` 默认 32，本机 8 核；且每样本白白 base64 编码 16 张图 |
| ⑤ | `run_sft.py` | `devices='auto'` 吃满 8 卡；`num_workers=4`×8 卡 = 32 workers 打爆 8 核 CPU |
| ⑥ | [sft_dataset.py:334](../../dataset_utils/sft_dataset.py) | `apply_chat_template(add_generation_prompt=True)` 却带着 assistant turn → 答案后多拼一段空的 `<\|im_start\|>assistant\n`。collator 找**第一个** `[151644, 77091]` 做 mask 起点，功能上没错，但模型会被训着预测那段尾巴。loss 曲线诡异时先查这里 |
| ⑦ | `run_sft.py` 构造顺序 | 必须先建 `SFTDataset`（给 `processor.tokenizer` 加 2048 个 `<action_i>`）再建 model（内部 `resize_token_embeddings`）。**别调换这两行** |
| ⑧ | [autovla_agent.py:255-261](../../navsim/navsim/agents/autovla_agent.py) | nuPlan 的 JSON 用 `left/right_camera_paths` 当 front-left/right；其他数据集用 `front_left/right_camera_paths`。靠 `dataset_name` 分支 normalize |
| ⑨ | codebook | 别重跑 `action_token_cluster.sh`，用仓库自带的 `agent_vocab.pkl` |
| ⑩ | eval config | README 提到的 `config/training/qwen2.5-vl-3B-nusc-sft.yaml` **不存在**，只有 `config/eval/qwen2.5-vl-3B-nusc-sft-eval.yaml` |

---

## 7. 两个已决策的问题（附理由）

### 7.1 先 nuPlan-only，**不要**上 nuPlan+nuScenes 混合

论文正式结果用的是 `mix-sft`，但那个配方的存在意义**跟 CoT 绑定**：nuScenes 进来主要是蹭 **DriveLM 的免费 CoT**（[sft_dataset.py:196-225](../../dataset_utils/sft_dataset.py) 专门解 `len(gt_cot)==5` 的五段式标注）。`use_cot=false` 时该分支根本不执行，nuScenes 退化成「又一批没有 CoT 的轨迹」。

代价却不小：多下 ~400G、要单独建 conda env（`nuscenes-devkit` 冲突）、多一个 camera key 命名的出错面。而 PDMS 评测本来只在 nuPlan navtest 上做。

→ **加 nuScenes 的正确时机是开始做 CoT 的时候**，那时 DriveLM 是唯一不用自己标就能拿到的 CoT 来源。

### 7.2 no-CoT 也保持**全参训 LLM**（不上 LoRA）

**技术上不是必须，但有一条硬约束**：模型要学 2048 个全新 `<action_i>` token，`resize_token_embeddings` 给的是随机初始化。LoRA 只打 `q/k/v/o_proj`，**碰不到 embedding** → 那 2048 行永远随机，模型学不会输出动作。真上 LoRA 必须加 `modules_to_save=["embed_tokens"]`。
> Qwen2.5-VL-3B 的 `tie_word_embeddings: True`（已从 HF config.json 确认），embed 与 lm_head 共享矩阵，训 embedding 一次覆盖输入输出两端。

**建议全参的四个理由**：
1. `run_sft.py` 里**没有 peft 路径**，自己加是新的出错面
2. 8×A100-80G 完全够（作者 L40S ~30GB/卡），省不出什么
3. **最重要**：baseline 和后续 CoT 实验必须同配方才可比。baseline LoRA / CoT 全参 → 涨点归因作废
4. RFT 的设计前提就是 SFT ckpt 是全参的（LoRA 是叠在它之上的第二层）

---

## 8. ViT 冻结：先别动，留作后置消融

AutoVLA 与 SimLingo 的配方**完全反过来**：

| | vision encoder | LLM |
|---|---|---|
| **SimLingo** | **全量训练**（[config.py:11](../../../../simlingo_training/config.py) `freeze: bool = False`） | **LoRA** r=16 / α=32 |
| **AutoVLA SFT** | **冻结** | **全参，无 LoRA** |
| **AutoVLA RFT** | **冻结** | **LoRA** r=8, q/k/v/o_proj |

> RFT 的 `target_modules=["q_proj","v_proj","k_proj","o_proj"]` 跟 Qwen2.5-VL visual block 的 `qkv`/`proj` 命名对不上，所以 **LoRA 也只打在 LLM 上**，ViT 全程没被碰过。作者是有意为之（论文架构图那处是笔误）。

**为什么 AutoVLA 敢冻**，三个原因，第 3 个大概率是与 SimLingo 经验差异的真正来源：

1. **视觉 token 量级差 12 倍** —— AutoVLA 每样本 3 相机 × 4 帧 = 12 张图，SimLingo 单帧
2. **分辨率已经压死** —— 见 §5.1，每帧 224×420 / 120 token，ViT 本就提不出细粒度信息，解冻的收益天花板很低
3. **域差** —— SimLingo 训 **CARLA 仿真图**，跟 InternVL2 预训练分布差得远，解冻 ViT 收益自然大；AutoVLA 训 nuPlan **真实路面**，域差小得多

### 真要试解冻，三个必补的坑

改 `train_vision_backbone: true` 一行就生效（[autovla.py:384-386](../../models/autovla.py) 会照做），但直接开会踩：

- **(a) 没有分层 lr（最要命）** —— 解冻后 ViT 和 LLM 共用 `2e-5`，对 675M 的 ViT 偏大，容易几百步毁掉预训练特征。
  注意 SimLingo 那边 lr=3e-5 看着更大，但它 **LLM 走 LoRA**，等效更新量小，ViT 是主导；AutoVLA 是 LLM 全参，两边抢同一个 lr。
  → 在 `configure_optimizers` 里给 `vlm.visual` 单开 param group，lr 设 `2e-6`。约 5 行。
- **(b) FSDP wrap policy 漏了 ViT** —— [run_sft.py:107-112](../../tools/run_sft.py) 只 wrap `Qwen2_5_VLDecoderLayer`，`Qwen2_5_VLVisionBlock` 不在里面 → 解冻后 ViT 参数**不分片**，每卡各存一份 optimizer state，约 +8G/卡。
  → 把 `Qwen2_5_VLVisionBlock` 加进 wrap policy。
- **(c) gradient checkpointing 没覆盖 ViT** —— [run_sft.py:63](../../tools/run_sft.py) 只对 `vlm.model`（LLM）开了，12 张图的 ViT 激活全留着。

**结论：先按原配方（冻结）跑 baseline，解冻当成独立的后置消融。** 否则它会成为 reasoning 收益归因的混淆变量。

---

## 9. 下一步：接 reasoning

no-CoT baseline 跑出来 = 100% fast thinking，这是对照组。接 CoT 三条路，按成本排：

1. **nuScenes + DriveLM**（最省）—— `v1_1_train_nus.json` 公开，`nusc_sample_generation.py` 直接支持，`sft_dataset.py` 有现成 nuscenes CoT 分支。代价：下 nuScenes + 单独 conda env
2. **自标 nuPlan CoT** —— 作者用 72B + 裸 HF generate 要 **10 天**（issue #38）；换 vLLM 能压到 1–2 天，本机 8×A100 跑 72B-AWQ 现实。且生成的 CoT 常不遵守模板，需后处理过滤
3. 等作者放 reasoning data（README 标 TBD，不可控）

**另一个独立于 CoT 的改进方向**：§5.1 显示每帧只有 224×420 / 120 token。CoT 要描述 "critical objects"，这个分辨率下远处的红绿灯和行人基本是糊的。**提分辨率可能是比解冻 ViT 更直接的旋钮**，且与 reasoning 主线协同（给 CoT 更多可描述的东西）。
