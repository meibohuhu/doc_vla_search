# 怎么生成 `navtrain_metric_cache`

RL 训练的硬前置。脚本 [`logs/0902/build_navtrain_metric_cache.sh`](../../logs/0902/build_navtrain_metric_cache.sh)。

---

## 1. 为什么需要

GRPO 的 reward 靠 `PDM_Reward.rl_pdm_score(traj, token)` 查缓存:

```python
metric_cache_path = self.metric_cache_loader.metric_cache_paths[token]   # KeyError → reward 恒 0
```

盘上只有 **navtest** 那份（12,146 个，完整）。
🔴 **不能拿 navtest 当训练集** —— 训完它就不再是 held-out，PDMS 这个论文主指标直接作废。

缓存里存的是 PDMScorer 判分要用的全部东西:`ego_state`、`trajectory`（PDM 参考轨迹）、
`observation`（各时刻的 agent 占用）、`centerline`、`route_lane_ids`、`drivable_area_map`。
**纯 CPU 产物，和模型无关，一次生成永久复用。**

## 2. 前置条件

> **[mh 2026/09/02] 本机（`/scratch/mh2803/vla`, 72 核）实测更新。**
> 下表与第 4 节原先写的是另一台机器（`/data/autovla_data`, 8 核）的数据，已改成本机实测值。

| 需要 | 本机路径 | 说明 |
|---|---|---|
| navsim trainval logs | `/scratch/mh2803/vla/nuplan/navsim_logs/trainval/*.pkl` | 1,310 个 ✅ |
| nuplan 地图 | `/scratch/mh2803/vla/nuplan/maps` | 需要 `NUPLAN_MAP_VERSION=nuplan-maps-v1.0` ✅ |
| scene_filter | `navsim/.../scene_filter/navtrain.yaml` | 仓库自带，含 **1,192 个 log** |
| train_test_split | `navsim/.../train_test_split/navtrain.yaml` | `data_split: trainval` |
| 磁盘 | **~26 GB** | 实测 navtest 3.1G/12,146 → 约 255 KB 一个 × 101,288 |

**不需要 sensor blobs**（图片），metric caching 只用 log + 地图。
**不需要 GPU** —— 实测确认:整个过程 `nvidia-smi --query-compute-apps` 里没有它的进程。

## 3. 跑

```bash
cd /home/mh2803/vla/doc_vla_search

# ① 先探速（算 200 个 scene，约 5 分钟）
#    ⚠️ 它只会用 1 个 worker，打印的"单 worker 外推"是假的，看"并行外推"那行。见 4.1
PROBE=200 bash logs/0902/build_navtrain_metric_cache.sh

# ② 正式跑（约 1.3 小时，挂 tmux/nohup）
tmux new -s cache
bash logs/0902/build_navtrain_metric_cache.sh

# ③ 验收（也可单独跑，不重新生成）
VERIFY=1 bash logs/0902/build_navtrain_metric_cache.sh

# 其他开关
NWORKERS=72 bash ...     # 用满全核（默认 36，留一半给训练）
NWORKERS=""  bash ...     # 空串 = null = ray 自己用满
NICE=15     bash ...     # 同机有训练时降优先级
WORKER=sequential PROBE=50 bash ...   # 换 worker 对比
```

脚本内部就是（本机路径）:

```bash
export OPENSCENE_DATA_ROOT=/scratch/mh2803/vla/nuplan
export NUPLAN_MAPS_ROOT=/scratch/mh2803/vla/nuplan/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NAVSIM_DEVKIT_ROOT="$REPO/navsim"
export NAVSIM_EXP_ROOT=/scratch/mh2803/vla/nuplan/exp
export PYTHONPATH="$REPO/navsim:$PYTHONPATH"

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
    train_test_split=navtrain \
    worker=ray_distributed_no_torch \
    worker.threads_per_node=36 \
    cache.cache_path=/scratch/mh2803/vla/nuplan/navtrain_metric_cache
```

脚本带三道 preflight（log 存在、地图存在、磁盘 ≥60G），跑完打印实测 s/token 和外推时长。

## 4. ⚠️ 耗时 & 一个会把人吓退的探速陷阱

### 4.1 🔴 PROBE 测出来的时间会高估几十倍 —— 别信它

[`caching.py:138`](../../navsim/navsim/planning/metric_caching/caching.py#L138) 是**按 log 文件**分发任务:

```python
data_points = [ ... for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items() ]
worker_map(worker, cache_scenarios, data_points)      # 1 个 log = 1 个 task
```

`PROBE=N` 用 `scene_filter.max_scenes=N` 截断，N 小的时候这 N 个 scene
**往往全落在同一个 log 里 → 只有 1 个 task → ray 起了 73 个 worker 但只有 1 个在干活。**

本机 2026/09/02 实测（`PROBE=200`）:

```
Ray objects: 100%|██████████| 1/1 [04:54]      ← 只有 1 个 task！
333s / 200 个  →  1.665 s/token（单 worker）
朴素外推 46.8 小时   ← 假的，因为全量不是单 worker
```

全量有 **1,192 个 log → 1,192 个 task**，才会真正铺满。所以外推必须除以并行度:

| worker 数 | 剩余 101,088 个的预计耗时 |
|---|---|
| 1（PROBE 的实际情况） | 46.8 小时 ← **别被这个数吓退** |
| **36（脚本默认）** | **≈1.3 小时** |
| 72（用满全核） | ≈0.65 小时 |

脚本已经会**同时打印这两个数**，不用自己换算。

### 4.2 本机全量实测

```
72 核，load 0.2 空闲，worker=ray_distributed_no_torch, threads_per_node=36
启动 90s 后: 23 个 worker 并发, load 16.8, 已产出 851 个
```

→ **CPU 是唯一瓶颈，RAM 和 GPU 都不是**（`nvidia-smi` 里全程没有它的进程）。
   `threads_per_node` 默认设 36 而不是 null(=72)，是给同机的训练任务留一半核
   —— 训练的 dataloader `num_workers=8` × 多卡很吃核，两边抢核会双输。

> 旧数据（8 核机器、且与 CoT-SFT 并发，load 39.8 过载 5 倍）:
> sequential 1.745 s/token→49.1h、ray 1.220→34.3h、ray 3.900→109.7h。
> 这三个数只反映"被 SFT 抢光核"的情况，与本机无关，仅存档。

小优化:`worker=ray_distributed_no_torch`（navsim 自带变体，worker 里不 import torch）。
反正不用 GPU，更轻、启动更快 —— 脚本已设为默认。

## 5. 断了怎么办 —— 直接重跑

`config/metric_caching/default_metric_caching.yaml` 里 `cache.force_feature_computation: false`，
**已经算好的 token 会被跳过**。中断之后原命令重跑即可，不用清目录、不用分片。

## 6. 验收

脚本自带验收，跑完会自动执行；也可单独跑:

```bash
VERIFY=1 bash logs/0902/build_navtrain_metric_cache.sh
```

它做三件事:数量、占盘、以及对照训练样本（本机是 `navtrain_nocot`，101,288 个）算覆盖率，
**并把缺失的 token 写到 `navtrain_missing_tokens.txt`**。

手动版:

```bash
# 数量（应为 ~101k）
find /scratch/mh2803/vla/nuplan/navtrain_metric_cache -name metric_cache.pkl | wc -l

# 覆盖率
python - <<'PY'
import os
from pathlib import Path
from navsim.common.dataloader import MetricCacheLoader
have = set(MetricCacheLoader(Path("/scratch/mh2803/vla/nuplan/navtrain_metric_cache")).tokens)
want = {os.path.splitext(n)[0] for n in os.listdir("/scratch/mh2803/vla/nuplan/navtrain_nocot")}
print(f"样本 {len(want)}   有 cache {len(want & have)}   缺 {len(want - have)}")
PY
```

⚠️ 缺几个是正常的（scene_filter 的 `has_route: true` 会滤掉一些）。
**但缺的 token 在训练时会让 `rl_pdm_score` 走异常分支 → reward 记 0**，
等于喂了一条"最差"的假信号。生成完应当把缺的 token 从训练样本里剔掉，
或者在 `rl_pdm_score` 里把查不到的 token 标成弃权（不进 advantage 的组）。

### 🔴 6.1 metric cache 不能跨机器直接拷 —— 元数据里是绝对路径

[`dataloader.py:176-180`](../../navsim/navsim/common/dataloader.py#L176-L180) 读
`<cache>/metadata/*.csv`，里面存的是**生成时的绝对路径**，`MetricCacheLoader` 直接拿它开文件:

```python
metadata_file = [f for f in metadata_dir.iterdir() if ".csv" in str(f)][0]   # 注意是子串匹配
cache_paths   = f.read().splitlines()[1:]                                    # 绝对路径，原样使用
```

2026/07/28 本机踩过:从那台机器拷过来的 `navtest_metric_cache`，元数据里还是
`/data/autovla_data/...`，结果**每个场景都 `Agent failed`、PDMS 全 0**，而 `.pkl` 其实就在本地。
修法是重写前缀:

```bash
sed -i 's#/data/autovla_data/nuplan#/scratch/mh2803/vla/nuplan#g' \
    <cache>/metadata/*.csv
```

**⚠️ 备份千万别放在 `metadata/` 里面** —— 上面那行是 `".csv" in str(f)` 的**子串**匹配，
`xxx.csv.bak` 也会被选中，且 `[0]` 可能挑到它，等于没改。备份放到 `metadata/` 之外。

本次是在本机直接生成的，路径天然正确，不用管这条 —— 但**以后要把这份 cache 拷到别的机器时必须处理**。

## 7. 生成后要改的 config

`config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml` 的 `data.train` 现在指着 navtest
（当初为了 smoke test），要改回:

```yaml
data:
  train:
    scene_filter: ./navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml
    metric_cache_path: /scratch/mh2803/vla/nuplan/navtrain_metric_cache
    json_dataset_path: /scratch/mh2803/vla/nuplan/navtrain_nocot   # 本机是 nocot 版
    sensor_data_path: null      # 样本里图片是绝对路径，不能再拼前缀
  val:
    ...navtest（保持不动）
```

⚠️ 该 config 里的 `pretrained_model_path` / `sft_model_path` 也要指向本机:
Qwen 在 `./Qwen2.5-VL-3B-Instruct`，SFT ckpt 在 `runs/sft/<时间戳>/epoch=N-loss=L.ckpt`
（当前最优:`runs/sft/2026-07-25_05-05-57/epoch=4-loss=0.6351.ckpt`，navtest 全量 PDMS 84.51）。
