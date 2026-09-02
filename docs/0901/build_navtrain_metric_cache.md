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

| 需要 | 本机路径 | 说明 |
|---|---|---|
| navsim trainval logs | `/data/autovla_data/nuplan/navsim_logs/trainval/*.pkl` | 1,310 个 |
| nuplan 地图 | `/data/autovla_data/nuplan/maps` | 需要 `NUPLAN_MAP_VERSION=nuplan-maps-v1.0` |
| scene_filter | `navsim/.../scene_filter/navtrain.yaml` | 仓库自带 |
| train_test_split | `navsim/.../train_test_split/navtrain.yaml` | `data_split: trainval` |
| 磁盘 | ~48 GB | 每个 cache ≈ 470 KB × 101,288 |

**不需要 sensor blobs**（图片），metric caching 只用 log + 地图。
**不需要 GPU** —— 实测确认:整个过程 `nvidia-smi --query-compute-apps` 里没有它的进程。

## 3. 跑

```bash
cd <REPO>/other_repo/AutoVLA

# ① 先探速（只算 200 个 scene，几分钟，打印 s/token 和外推总时长）
PROBE=200 bash logs/0902/build_navtrain_metric_cache.sh
WORKER=ray_distributed PROBE=200 bash logs/0902/build_navtrain_metric_cache.sh

# ② 正式跑（长任务，挂 tmux）
tmux new -s cache
WORKER=ray_distributed bash logs/0902/build_navtrain_metric_cache.sh
```

脚本内部就是:

```bash
export OPENSCENE_DATA_ROOT=/data/autovla_data/nuplan
export NUPLAN_MAPS_ROOT=/data/autovla_data/nuplan/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NAVSIM_DEVKIT_ROOT="$REPO/navsim"
export NAVSIM_EXP_ROOT=/data/autovla_data/nuplan/exp
export PYTHONPATH="$REPO/navsim:$PYTHONPATH"

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
    train_test_split=navtrain \
    worker=ray_distributed \
    cache.cache_path=/data/autovla_data/nuplan/navtrain_metric_cache
```

脚本带三道 preflight（log 存在、地图存在、磁盘 ≥60G），跑完打印实测 s/token 和外推时长。

## 4. ⚠️ 耗时:三次实测差 3 倍，全是 CPU 争抢

```
sequential        349s / 200  →  1.745 s/token  →  外推 49.1 小时
ray_distributed   244s / 200  →  1.220 s/token  →  外推 34.3 小时
ray_distributed   156s /  40  →  3.900 s/token  →  外推 109.7 小时
```

**这三个数没有一个能代表空闲机器上的速度。** 测的时候本机在跑 CoT-SFT:
4 GPU × `num_workers: 8` = 32 个 dataloader 进程抢 **8 个核**，load average 39.8。
ray 的 worker 只能拿到 4% 的核，"并行"根本没并起来。

```
实测资源占用（ray，与 SFT 并发时）
  GPU        0 MiB          ← 完全不用
  主进程     cpu 70%   rss 0.72 GB
  ray worker cpu  4%   rss 0.09 GB 每个
  load avg   39.8 / 8 核     ← 过载 5 倍
```

→ **CPU 是唯一瓶颈，RAM 和 GPU 都不是。** 在空闲机器上跑，或等 SFT 结束再跑。
   换机器的话**核数越多越好**，`worker=ray_distributed` 的 `threads_per_node: null`
   会自动用满所有核。**先 PROBE 再决定，别盲开一个几十小时的任务。**

小优化:`worker=ray_distributed_no_torch`（navsim 自带的变体，worker 里不 import torch）。
反正不用 GPU，更轻、启动更快。

## 5. 断了怎么办 —— 直接重跑

`config/metric_caching/default_metric_caching.yaml` 里 `cache.force_feature_computation: false`，
**已经算好的 token 会被跳过**。中断之后原命令重跑即可，不用清目录、不用分片。

## 6. 验收

```bash
# 数量（应为 ~101k，与 navtrain_cot 的 101,288 同量级）
ls /data/autovla_data/nuplan/navtrain_metric_cache/*/*/*/metric_cache.pkl | wc -l

# 覆盖率:训练样本的 token 是不是都能查到
python - <<'PY'
import os
from pathlib import Path
from navsim.common.dataloader import MetricCacheLoader
have = set(MetricCacheLoader(Path("/data/autovla_data/nuplan/navtrain_metric_cache")).tokens)
want = {os.path.splitext(n)[0] for n in os.listdir("/data/autovla_data/nuplan/navtrain_cot")}
print(f"样本 {len(want)}   有 cache {len(want & have)}   缺 {len(want - have)}")
PY
```

⚠️ 缺几个是正常的（scene_filter 的 `has_route: true` 会滤掉一些）。
**但缺的 token 在训练时会让 `rl_pdm_score` 走异常分支 → reward 记 0**，
等于喂了一条"最差"的假信号。生成完应当把缺的 token 从 `navtrain_cot` 剔掉，
或者在 `rl_pdm_score` 里把查不到的 token 标成弃权（不进 advantage 的组）。

## 7. 生成后要改的 config

`config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml` 的 `data.train` 现在指着 navtest
（当初为了 smoke test），要改回:

```yaml
data:
  train:
    scene_filter: ./navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml
    metric_cache_path: /data/autovla_data/nuplan/navtrain_metric_cache
    json_dataset_path: /data/autovla_data/nuplan/navtrain_cot
    sensor_data_path: null      # 样本里图片是绝对路径，不能再拼前缀
  val:
    ...navtest（保持不动）
```
