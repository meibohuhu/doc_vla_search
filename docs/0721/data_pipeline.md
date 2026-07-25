# AutoVLA nuPlan 数据处理流水线：用到哪些代码、按什么顺序

> 本文记录**实际跑通**的流程（不是照抄 README），含每一步用到的源文件、输入输出、验证方法，以及踩过的坑。
> 配套：[AutoVLA_summary.md](AutoVLA_summary.md) · [AutoVLA_nocot_sft_runbook.md](AutoVLA_nocot_sft_runbook.md) · [benchmark_comparison.md](benchmark_comparison.md)
> 环境：8×A100-80G（共享，另有 8 卡训练在跑）· 8 核 CPU · 数据盘 `/data`
> 日期：2026-07-21

---

## 0. 总览

```
              ┌─── 阶段 1 下载 ────┐   ┌── 阶段 2 预处理 ──┐   ┌─ 阶段 3 消费 ─┐

                navsim_logs/*.pkl  ─┐
  HuggingFace                       ├─► nocot_sample      ─┬─► navtest_nocot/  ──► PDMS 评测
  + AWS S3      sensor_blobs/*.jpg ─┘   _generation.py     │   12,146 JSON
                                                           └─► navtrain_nocot/ ──► SFT 训练
                maps/  ─────────────────► metric_caching ──►    103,288 JSON
                                          .py                   navtest_metric_cache/
                                                                （只评测需要）
```

**关键认知**：预处理产出的 JSON 是**训练和评测共用的唯一输入格式**。
`SFTDataset.__init__` 就是 `data_path.glob('*.json')` —— 训练时**完全不碰 navsim 的 SceneLoader**，
只读 JSON 里的 `camera_paths` 再去磁盘取图。所以 train / test 两个 split 都要各跑一遍预处理。

---

## 1. 阶段一：下载

### 用到的代码

| 文件 | 角色 |
|---|---|
| `scripts/download_nuplan_autovla.sh` | **自建**。替代官方 `navsim/download/*.sh` |

### 产出

```
/data/autovla_data/nuplan/
├── maps/                      971 MB     ← metric cache 需要
├── navsim_logs/
│   ├── trainval/  1310 个 .pkl   14 GB   ← 场景元数据（ego 位姿/轨迹/相机路径/roadblock）
│   └── test/       147 个 .pkl
└── sensor_blobs/
    ├── trainval/  1192 个 log   445 GB   ← navtrain 的 JPEG
    └── test/       147 个 log   121 GB   ← navtest 的 JPEG
```

### 为什么不用官方脚本

| 官方脚本 | 问题 |
|---|---|
| `wget -qO- \| tar -xz` 流式管道 | **下载被截断时 gzip 报错，但管道退出码被忽略 → 分片静默丢失**。实际让 20 个 navtest log 消失而脚本报"完成" |
| 解出的目录是平铺的 `mini_navsim_logs` | 与代码要求的 `navsim_logs/<split>` 两级结构不符 |
| 无完整性核对 | 缺数据要到训练/评测时才发现 |

自建版每个分片走 **`wget -c` 落盘 → `gzip -t` 校验 → 解压 → 打 `.done` 标记 → 删 tgz**，全程幂等可续传；
并新增 `verify` 子命令，拿 `navtest.yaml` / `navtrain.yaml` 的 log 清单跟实际 sensor 目录对账。

```bash
bash scripts/download_nuplan_autovla.sh          # maps → logs → test → navtrain → verify
bash scripts/download_nuplan_autovla.sh verify   # 只核对
```

**只以 `verify` 的输出为准，不看阶段 echo。**

---

## 2. 阶段二 A：预处理（navsim → JSON）

### 调用链

```
tools/preprocessing/nocot_sample_generation.py          ← 入口，写 <token>.json
  └─ dataset_utils/preprocessing/nuplan_dataset.py
       NuplanCoTAnnotationDataset.__getitem__
         ├─ navsim/common/dataloader.py
         │    SceneLoader / filter_scenes                ← 按 scene_filter 切场景
         ├─ navsim/agents/vla_agent.py
         │    VlaAgent.get_sensor_config()               ← 8 相机全开, lidar_pc=False
         │    feature builders                           ← ego 速度/加速度/指令/历史轨迹
         └─ navsim/common/dataclasses.py
              Cameras.from_camera_dict                   ← 填 camera_path（并解码图像）
```

### 配置

| 文件 | split | scene_filter | 产出 |
|---|---|---|---|
| `config/dataset/0721/nuplan-navtest.yaml` | test | `navtest.yaml` | 12,146 JSON |
| `config/dataset/0721/nuplan-navtrain.yaml` | trainval | `navtrain.yaml` | 103,288 JSON |

### 产出格式

每个场景一个 `<token>.json`，约 **5.5 KB**（12,146 个共 96 MB）：

```json
{ "token": "...", "dataset_name": "nuplan", "cot_output": [],
  "velocity": [vx,vy], "acceleration": [ax,ay], "instruction": "keep forward",
  "gt_trajectory": [[x,y,heading] × 10], "his_trajectory": [...],
  "front_camera_paths": ["/abs/path/.../CAM_F0/xxx.jpg", ...共4帧],
  "left_camera_paths": [...], "right_camera_paths": [...], ... }
```

**只存路径，不复制图片。**

### 命令

```bash
export NUPLAN_MAPS_ROOT=/data/autovla_data/nuplan/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export OPENSCENE_DATA_ROOT=/data/autovla_data/nuplan
export NAVSIM_DEVKIT_ROOT=$PWD/navsim
export NAVSIM_EXP_ROOT=$PWD/navsim/exp
export TOKENIZERS_PARALLELISM=false

# navtest（评测用）
python tools/preprocessing/nocot_sample_generation.py \
  --config dataset/0721/nuplan-navtest \
  --output_dir /data/autovla_data/nuplan/navtest_nocot \
  --num_workers 4 --fast

# navtrain（训练用）
python tools/preprocessing/nocot_sample_generation.py \
  --config dataset/0721/nuplan-navtrain \
  --output_dir /data/autovla_data/nuplan/navtrain_nocot \
  --num_workers 4 --fast
```

---

## 3. 阶段二 B：`--fast` 加速补丁

`dataset_utils/preprocessing/0721/fast_nocot_patch.py`（**自建**，纯 monkeypatch，不改原文件）

### 问题

JSON 里只有元数据和路径字符串，**一个像素都不需要**。但产生它们的路上做了三件白工：

| # | 位置 | 干什么 | 用得上吗 |
|---|---|---|---|
| 1 | `Cameras.from_camera_dict` | 填 `camera_path` 的**同一行** `Image.open()`。8 相机 × 4 帧 = **32 张 1920×1080/场景** | ❌ 丢弃 |
| 2 | `nuplan_dataset.process_image_input` | 对其中 16 张 cvtColor + imencode + base64 | ❌ 丢弃 |
| 3 | `process_vision_info` | 把 16 张 base64 解回来再缩放到 400×400 | ❌ 丢弃 |

103,288 场景 = **约 330 万次无用的 JPEG 解码**。

### 补丁

替换上述三个函数：只填 `camera_path` / 返回空串 / 返回 `(None, None)`。
`camera_path` 取自同一个 `camera_dict[name]["data_path"]`，与原实现同源 → 输出应完全相同。

额外保留一道 `os.path.exists`，补回原实现靠 `Image.open` 隐式提供的"缺图立即报错"检查（开销测不出来）。

### 实测（`scripts/0721/bench_fast_nocot.py`，N=30）

```
正确性   : JSON 逐字节一致 ✅ 30/30
baseline : 1904 ms/场景
patched  :    3 ms/场景
加速比   : 625×
```

| | baseline | patched（实测） |
|---|---|---|
| navtest 12,146 | 6.4 小时 | **37.9 秒** |
| navtrain 103,288 | 54.6 小时 | ~6 分钟 |

> **⚠️ 只能用于 no-CoT 路径。** CoT 标注（72B 教师模型）真的需要图像内容。
> `apply()` 是显式调用，默认不生效。

---

## 4. 阶段二 C：metric cache（**只有评测需要**）

### 用到的代码

```
scripts/0721/run_navtest_metric_cache.sh                 ← 自建
  └─ navsim/planning/script/run_metric_caching.py
       └─ navsim/planning/metric_caching/metric_cache.py::MetricCache
```

### 产出

```
navtest_metric_cache/<log>/unknown/<token>/metric_cache.pkl    （pickle + lzma）
12,146 个场景 ≈ 3 GB
```

内容（[metric_cache.py:25-36](../../navsim/navsim/planning/metric_caching/metric_cache.py)）：

```python
trajectory:        InterpolatedTrajectory   # PDM-Closed 基线规划器的参考轨迹
ego_state:         EgoState
observation:       PDMObservation           # 他车未来轨迹（非反应式）
centerline:        PDMPath
route_lane_ids:    List[str]
drivable_area_map: PDMDrivableMap
```

### 通用性

✅ **对模型完全通用** —— 全是场景派生的，零模型相关内容。任何方法评同一批场景共用一份。

❌ **绑死四样东西**：

| 绑定 | 换了会怎样 |
|---|---|
| navsim 版本 | v2 的 EPDMS 需要 traffic light / lane keeping 字段，结构不同 |
| split / scene_filter | 按 token 组织，只覆盖 navtest 那 12,146 个 |
| 地图版本 | `drivable_area_map` / `centerline` 从 `nuplan-maps-v1.0` 算出 |
| nuplan-devkit 版本 | pickle 里存的是 devkit 对象 |

**它是 pickle，不兼容时可能不报错而是静默给出错误分数。** 这是 issue #48（复现只到 83.69 而非 89.11）的路径之一。
→ 存放在**共享路径**而非 repo 内，跨 repo 复用前先用 `constant_velocity` agent 对拍一次。

### 命令

```bash
bash scripts/0721/run_navtest_metric_cache.sh navtest ray_distributed 4
#                                        split   worker           并行度
```

官方 `navsim/scripts/evaluation/run_metric_caching.sh` 的三个问题（自建版都改了）：
默认是 `warmup_test_e2e`（navtest 两行被注释）· 全相对路径依赖 CWD · 无优先级控制。

默认 worker 是 **`Sequential` 单线程**，实测 **~2 秒/场景 → 12,146 个要 6.7 小时**。
改 `ray_distributed` + 4 worker 后约 1.2 小时。

---

## 5. 阶段三：谁在消费这些 JSON

### 训练

```
tools/run_sft.py
  └─ dataset_utils/sft_dataset.py::SFTDataset          ← glob('*.json')
       ├─ navsim/agents/autovla_agent.py
       │    AutoVLAAgentFeatureBuilder.compute_features ← 解析 JSON 字段
       │    TrajectoryTargetBuilder.compute_targets     ← GT 轨迹 → action token
       └─ models/action_tokenizer.py + codebook_cache/agent_vocab.pkl
```

### 评测

```
navsim/scripts/evaluation/run_autovla_agent_pdm_score_evaluation.sh
  └─ navsim/planning/script/run_pdm_score_cot.py
       ├─ json_data_path      = navtest_nocot/           ← 阶段二 A
       ├─ metric_cache_path   = navtest_metric_cache/    ← 阶段二 C
       └─ agent=autovla_agent, checkpoint, LORA=false
```

### action codebook —— **不要重新生成**

仓库自带 `codebook_cache/agent_vocab.pkl`（1.1 MB，shape 2048×6×4×2）。
`scripts/action_token_cluster.sh` 的主循环按单步累积，与 codebook 的 6-step 段维度看似不对齐（见 summary §2.5）。

---

## 5.5 文件存放约定（2026-07-21 起）

**新增**的文件按日期分目录；**直接修改上游原文件**的不动位置。

| 类型 | 位置 |
|---|---|
| 新增文档 | `docs/721/` |
| 新增脚本 | `scripts/0721/` |
| 新增 config | `config/dataset/0721/` |
| 新增运行时模块 | `dataset_utils/preprocessing/0721/` |
| 修改上游文件 | 原地不动（`tools/preprocessing/*.py` 等） |

### ⚠️ 由此产生的两个约束

**(1) 以数字开头的目录不能用普通 import 语法。**
`fast_nocot_patch.py` 被放进 `0721/` 后，Python 模块名不能以数字开头：

```python
from dataset_utils.preprocessing.0721.fast_nocot_patch import apply   # SyntaxError
```

调用方改用 `importlib` 按字符串导入（见 `nocot_sample_generation.py` 的 `--fast` 分支）：

```python
import importlib
apply = importlib.import_module("dataset_utils.preprocessing.0721.fast_nocot_patch").apply
```

> 如果以后觉得这层间接太别扭，可把这个**运行时被 import 的模块**移回
> `dataset_utils/preprocessing/`（日期目录更适合放文档/脚本/实验产物，而非被 import 的库代码）。

**(2) 脚本里的路径解析已改成"深度无关"。**
移进 `0721/` 后，原来 `dirname/..` 求仓库根的写法会指到 `scripts/`。
现在统一改成**向上找含 `setup.py` 的目录**，以后再挪层级也不会断：

```bash
# scripts/0721/run_navtest_metric_cache.sh
REPO="$(cd "$(dirname "$(readlink -f "$0")")" && while [ ! -f setup.py ] && [ "$PWD" != / ]; do cd ..; done; pwd)"
```
```python
# scripts/0721/bench_fast_nocot.py
REPO = next(p for p in Path(__file__).resolve().parents if (p / "setup.py").exists())
```

> 命名小不一致：docs 用 `721`，其余用 `0721`。不影响功能，但统一成 `0721` 会更省心。

---

## 6. 我们对上游代码的改动

| 文件 | 改动 | 原因 |
|---|---|---|
| `dataset_utils/preprocessing/0721/fast_nocot_patch.py` | **新增** | 见 §3 |
| `tools/preprocessing/nocot_sample_generation.py` | waymo 改惰性导入 | 它顶层 `import tensorflow`，**只跑 nuPlan 也被迫装 TF**（会跟钉死的 `numpy==1.23.4` 冲突） |
| 同上 | 加 `--fast` 开关 | 默认不生效，保持原行为 |
| `tools/preprocessing/cot_sample_generation.py` | waymo 改惰性导入 | 同上 |
| `scripts/download_nuplan_autovla.sh` | **新增** | 见 §1 |
| `scripts/0721/run_navtest_metric_cache.sh` | **新增** | 见 §4 |
| `config/dataset/0721/nuplan-navtest.yaml` | **新增** | |
| `config/dataset/0721/nuplan-navtrain.yaml` | **新增** | |

辅助/验证脚本（都是自建，不影响主流程）：
`scripts/count_scenes.py`（数各 filter 配置的场景数）·
`scripts/0721/count_frame_coverage.py`（估 sensor 包的帧覆盖）·
`scripts/0721/bench_fast_nocot.py`（补丁 A/B）

---

## 7. 坑清单（全部实际踩过）

| # | 坑 | 症状 | 解 |
|---|---|---|---|
| 1 | **`dataset_path` 用相对路径** | JSON 里 camera_path 是相对且已含 `sensor_blobs` 前缀；`SFTDataset` 再 `os.path.join(sensor_data_path, ...)` → **双重前缀**。预处理阶段**不报错**，训练/评测时全部找不到图 | config 里一律用**绝对路径**，`os.path.join` 遇绝对路径会丢弃前缀 |
| 2 | **`scene_filter` 留 null** | fallback 到 `frame_interval=4` 的密集切片（论文 166.3k 口径），需要完整 trainval 的 2TB sensor。只下了 navtrain 445GB 的话大量场景缺图 | 写死 `navtrain.yaml`（→ 103,288），切法必须与已下载的 sensor 配套 |
| 3 | **流式下载静默失败** | `wget -qO- \| tar -xz` 的管道退出码被忽略，截断分片消失，脚本却报"完成" | 落盘 + `gzip -t` 校验 + `.done` 标记 + `verify` 对账 |
| 4 | **`meta_datas/` 自带 split 子目录** | `mv meta_datas navsim_logs/trainval` → `navsim_logs/trainval/trainval/*.pkl` 多一层 | 先探测再 mv |
| 5 | **waymo 顶层 import tensorflow** | 只跑 nuPlan 也 `ModuleNotFoundError: No module named 'tensorflow'` | 改惰性导入 |
| 6 | **metric cache 默认单线程** | `Sequential` worker，~2 秒/场景，12,146 个要 6.7 小时 | `worker=ray_distributed` + `threads_per_node=N` |
| 7 | **`run_metric_caching.sh` 默认 split 是 warmup** | navtest 那两行是注释掉的 | 用自建脚本 |
| 8 | **`hf` / `huggingface-cli` 在本机是坏的** | `~/.local` 的 huggingface_hub 1.10.2 与 typer 冲突，报 `Typer.__init__() got an unexpected keyword argument 'suggest_commands'`，**且 traceback 后退出码仍是 0** | 用 conda env 里的 0.29.3，或直接 wget 公开 URL |
| 9 | **conda env / pip cache 默认落在 `/`** | `/` 只有 37G 且同机训练任务在往那写 checkpoint，撑满会连累别人 | env 和 `PIP_CACHE_DIR` 都放 `/data` |

---

## 8. 执行顺序速查

```bash
# 0) 环境（一次性）
conda env create -f environment.yml -p /data/autovla_data/envs/autovla
conda run -p /data/autovla_data/envs/autovla pip install -e . --no-warn-conflicts
conda run -p /data/autovla_data/envs/autovla pip install -e ./navsim --no-warn-conflicts
# 跳过 flash-attn / waymo-open-dataset(TF) / autoawq —— 见 runbook §1.2

# 1) 下载（可断点续传）
bash scripts/download_nuplan_autovla.sh
bash scripts/download_nuplan_autovla.sh verify      # ★ 以这个为准

# 2) 预处理（--fast）
python tools/preprocessing/nocot_sample_generation.py \
  --config dataset/0721/nuplan-navtest  --output_dir /data/autovla_data/nuplan/navtest_nocot  --num_workers 4 --fast
python tools/preprocessing/nocot_sample_generation.py \
  --config dataset/0721/nuplan-navtrain --output_dir /data/autovla_data/nuplan/navtrain_nocot --num_workers 4 --fast
# 核对数量: navtest 应 12,146 / navtrain 应 103,288

# 3) metric cache（只评测需要）
bash scripts/0721/run_navtest_metric_cache.sh navtest ray_distributed 4

# 4) 评测 / 训练
bash navsim/scripts/evaluation/run_autovla_agent_pdm_score_evaluation.sh   # 需改路径 + LORA=false
python tools/run_sft.py --config training/<your-nocot-sft-config>
```

**每一步都要核对产出数量，不要相信脚本的"完成"字样** —— 本次流程中出现过三次「看着完成了其实缺数据」。
