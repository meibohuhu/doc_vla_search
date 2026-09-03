# 166.3k 数据准备 runbook

日期 2026-09-01 · 背景/推导:[docs/0901/nuplan_103k_to_166k.md](../../docs/0901/nuplan_103k_to_166k.md)
脚本:[scripts/0901/](../../scripts/0901/) · 数据根:`/data/autovla_data`

> 目标:把 nuPlan 训练集从现有 navtrain **103,288** 扩到论文 Table S1 的 **166,282**
> (scene_filter=null, fi=4, 全 1310 log —— 已实测 SceneFilter 数出 166,282 吻合)。

---

## 状态总览

| 步骤 | 状态 | 说明 |
|---|---|---|
| 0. 清理无用 log 文件 | ✅ 已做 | 删顶层 34 个一次性 stdout 日志(4.9MB);数据集/ckpt 未动 |
| 1. eval 数据归到单独 folder | ✅ 已做 | 移到 `/data/autovla_data/eval/`(138G)+ 旧路径 symlink |
| 2. 下载全量 trainval sensor(~2TB) | ⏳ 待跑 | `download_trainval_full.sh`,5–8h |
| 3. 预处理 → 166k JSON | ⏳ 待跑 | `preprocess_166k.sh`,~0.5–1.5h |
| 4.(可选)建 train metric_cache | ⏳ 按需 | 仅 RFT/PDMS 需要 |
| 5. 训练指向 166k | ⏳ 待跑 | 改 config 的 `data.train.json_dataset_path` |

---

## Step 0 —— 清理无用 log 文件（已完成）

删除 `/data/autovla_data/` 顶层 **34 个一次性 stdout 日志**(`overfit500*.log`、`waymo_*.log`、
`nusc_*.log`、`sft_eval*.log`、`merge_*.log`、`coverage.log`、`count_trainval.log` 等,共 4.9MB)。

- **未动**:所有数据集(waymo 1.2T / checkpoints 198G / nuplan / nuscenes)、wandb 目录、
  eval 里的 run.log(随 eval 一起归档)。
- 复核:`find /data/autovla_data -maxdepth 1 -name '*.log' | wc -l` → 0。

## Step 1 —— eval 数据归单独 folder（已完成）

```bash
APPLY=1 bash scripts/0901/reorg_eval_data.sh
```

同盘 `mv` + 旧路径留 **symlink**(秒级、可逆、不破坏引用旧路径的 ~8 个脚本/config)。
归档到 `/data/autovla_data/eval/`(共 138G):

```
eval/
├── nuplan/
│   ├── navtest_nocot            (12,146 场景)
│   ├── navtest_metric_cache     (3.0G, PDMS 评测缓存)
│   ├── sensor_blobs/test        (121G, navtest 相机)
│   ├── navsim_logs/test         (147 log metadata)
│   └── sft_eval* / pdms_shards / exp / viz_tiers   (历史评测输出)
└── nuscenes/
    └── nusc_eval_seg            (6,019 个 .pt, UniAD 分割, L2+Collision 用)
```

已核对:seg .pt=6019、navtest_nocot=12146、test sensor=147 log,symlink 全部解析正确。

## Step 2 —— 下载全量 trainval sensor（~2TB，待跑）

```bash
tmux new -s dl
bash scripts/0901/download_trainval_full.sh          # 200 分片，5–8h
# 中断可续传；下完自动 verify
bash scripts/0901/download_trainval_full.sh verify   # 单独核对：应齐 1310 log
```

要点:
- 下 `openscene_sensor_trainval_camera_{0..199}.tgz`(全 1310 log)。**分片非 log 对齐,
  没法只补差量** → 下全 2TB(与现有 navtrain 446G 重叠部分被硬链接覆盖,不额外占空间)。
- robust 机制:`wget -c` 落盘 → `gzip -t` 校验 → `cp -rlf` 硬链接 → `.done` 标记。
  **绝不用 `wget -qO- | tar -xz` 流式管道**(截断静默丢分片,本项目中招过 3 次)。
- 磁盘:reorg 后 `/data` 余量充足(下载前确认 ≥2T free)。

## Step 3 —— 预处理 → 166k JSON（待跑）

```bash
bash scripts/0901/preprocess_166k.sh
```

- 用 [config/dataset/0901/nuplan-trainval-full-166k.yaml](../../config/dataset/0901/nuplan-trainval-full-166k.yaml)
  (`scene_filter: null` → 默认 fi=4 全 log)。
- `nocot_sample_generation --fast`(no-CoT 逐字节一致,~600x)→ `/data/autovla_data/nuplan/trainval_nocot_166k`。
- 末尾核对:应 ≈ **166,282** json。明显偏少 = Step 2 有分片没下满 → 回去 `verify` 续传。
- ⚠️ SceneFilter 只读 metadata(navsim_logs,已在盘),所以严格说这步在 sensor 下满前也能"跑通",
  但缺图的场景会被跳过 → **务必先 Step 2 下满再跑**。

## Step 4 —— train metric_cache（可选，仅 RFT/PDMS 需要）

SFT 训练不需要。若要在 166k 上做 PDMS reward(RFT)或 PDMS 评测:

```bash
bash scripts/0721/run_navtest_metric_cache.sh trainval ray_distributed 8   # 量大耗时数小时
```

## Step 5 —— 训练指向 166k

把训练 config 的 `data.train.json_dataset_path` 指到 `/data/autovla_data/nuplan/trainval_nocot_166k`,
其余同现有 nocot SFT(ViT frozen + LLM full-param + `fp32_master: true`)。
混训则加进列表(见 `config/training/qwen2.5-vl-3B-mix-nuplan-nuscenes-nocot-sft.yaml` 写法)。

---

## 时间估算（单流实测 ~22.6 MB/s）

| 步骤 | 估算 |
|---|---|
| Step 2 下载 2TB(6 并行 ~110MB/s) | ~5–8 h |
| Step 3 预处理 --fast | ~0.5–1.5 h |
| **到能训 SFT 合计** | **~6–10 h**(下载占大头) |
| Step 4 metric_cache(仅 RFT) | +2–4 h |

## 判读提醒

166k 相比 103k 多 ~60% 场景,但要多下 ~1.5T 净新数据。当前混训(nuplan 103k + nuScenes 19k)
已在 nuScenes 拿到 L2 0.64m(优于纯 nuScenes 0.68m),多域收益大于"nuplan 103k→166k"。
上 166k 主要为对齐论文 nuPlan 规模。
