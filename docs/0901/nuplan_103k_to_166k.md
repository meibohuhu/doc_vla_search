# 从 navtrain 103k 扩到 AutoVLA Table S1 的 166.3k

日期 2026-09-01 · 相关:[data_pipeline.md](../0721/data_pipeline.md)(下载/预处理坑全集)·
[count_frame_coverage.py](../../scripts/0721/count_frame_coverage.py)(103k vs 166k 的来源核算)

> 目标:把 nuPlan 训练集从现有 **103,288**(navtrain 标准 split)扩到 **166,282**(论文 Table S1 口径)。
> 结论先行:**预处理代码现成,唯一成本是重下全量 trainval sensor(~2TB);端到端约 6–10 小时。**

---

## 1. 差别:不是"采样更密",是"用的 log 更多"

| | log 数 | 场景数 | scene_filter | sensor 数据 |
|---|---|---:|---|---|
| **现有 navtrain** | 1192(navtrain.yaml 白名单) | **103,288**(train 101,288 + val 2,000) | `navtrain.yaml`,`frame_interval=1` | 446G(**只覆盖这 1192 log 的相关帧**) |
| **AutoVLA 166.3k** | 全部 1310 | **166,282** | `null` → 默认 `frame_interval=4`,不限 log | 全量 trainval camera **~2TB** |

**为什么 fi=4(跳更多帧)反而比 navtrain 的 fi=1 多?** 因为 navtrain 只用了 1310 个 log 里精选的 1192 个;
166.3k 铺满全部 1310 个 log。**跨更多 log 的收益 > 在少数 log 上密采样。**

**卡点**:我们下的 446G sensor 包**只含 navtrain 那 1192 log 的帧**。要 166.3k 就得把全量 trainval
camera 补齐。而 HF 上的 sensor 是按 `openscene_sensor_trainval_camera_{0..199}.tgz` **任意分块**打包的
(不按 log 切),所以**没法只补差量**,只能重下全部 200 个分片 = ~2TB(与现有 446G 重叠但挑不出来)。

> 依据:`navtrain.yaml` 实测 1192 log;`count_frame_coverage.py` 头注释记 166,282 = 全 1310 log + fi=4;
> `data_pipeline.md §2` 记了"scene_filter 留 null → 166.3k → 要 2TB sensor"这个坑。

---

## 2. 步骤

### Step 1 — 下载全量 trainval sensor(~2TB,主要成本)

`navsim/download/download_trainval.sh` 现成,但它用的是 `wget -qO- | tar -xz` **流式管道**,
有致命坑:**下载被截断时 gzip 报错,但管道退出码被忽略 → 分片静默丢失,脚本却报"完成"**
(本项目已中招 3 次,`data_pipeline.md §55/§339`)。

**必须改用落盘 + 校验版**(`data_pipeline.md §358` 的 `download_nuplan_autovla.sh` 模式):

```bash
# 每个分片：先落盘 → gzip -t 校验完整性 → 通过才解包 → 打 .done 标记 → 断点续传
# 200 个分片，-P 8 并行；下完必跑 verify 对账，别信"完成"字样
bash scripts/download_nuplan_autovla.sh            # 需把 target split 从 navtrain 改成全量 trainval camera 0..199
bash scripts/download_nuplan_autovla.sh verify     # ★ 以这个的对账为准
```

落盘位置:`/data/autovla_data/nuplan/sensor_blobs/trainval/`(现有 446G 会被全量覆盖/补齐)。
lidar 分片**不下**(no-CoT / 轨迹任务用不到,脚本里已注释掉)。

### Step 2 — 预处理成 JSON(166.3k 口径)

关键就一处:**`scene_filter` 留 `null`**,让 `nuplan_dataset.py` fallback 到默认
`frame_interval=4` + 不限 log = 正好 166.3k。写一个 preprocessing config(照
`dataset/0721/nuplan-navtrain` 改,把 `scene_filter` 设为 null、`split` 指向 trainval):

```bash
python tools/preprocessing/nocot_sample_generation.py \
  --config dataset/0901/nuplan-trainval-full \
  --output_dir /data/autovla_data/nuplan/trainval_nocot_166k \
  --num_workers 32 --fast          # --fast: no-CoT 下逐字节一致，约 600x 提速
# 核对：应 ≈ 166,282 个 json（±少量因缺帧被跳过）
```

⚠️ 数量务必核对(`find ... -name '*.json' | wc -l`),别信脚本的"完成"。若明显少于 166k,
多半是 Step 1 有分片静默丢了 → 回去 `verify`。

### Step 3 —(可选,仅 RFT/PDMS 需要)建 metric_cache

SFT 训练**不需要**这步。只有要在这批数据上算 PDMS reward(RFT)或跑 PDMS 评测才要:

```bash
bash scripts/0721/run_navtest_metric_cache.sh trainval ray_distributed 8
```

166k 的 metric_cache 会很大(navtest 12k = 3.0G → 推算 166k ≈ 40G+),且耗时数小时,单列。

### Step 4 — 训练

把训练 config 的 `data.train.json_dataset_path` 指到 `trainval_nocot_166k`,其余同现有 nocot SFT
(ViT frozen + LLM full-param + `fp32_master: true`)。混训则加进列表(见
`config/training/qwen2.5-vl-3B-mix-nuplan-nuscenes-nocot-sft.yaml` 的写法)。

---

## 3. 时间估算

实测(2026-09-01,本机 → HF CDN):**单流 ~22.6 MB/s**。下载脚本 `-P 8` 并行,聚合按现实 ~5×
打折(CDN/带宽共享,达不到理论 8×)≈ **~110 MB/s**。

| 步骤 | 估算 | 说明 |
|---|---|---:|
| **Step 1 下载 2TB** | **~5–8 h** | 110 MB/s → 5.1h;保守留波动到 8h;解包(tar)与下载重叠,不额外计 |
| Step 2 预处理 --fast | **~0.5–1.5 h** | no-CoT --fast 极快(I/O 为主);103k 抽样 20k/1.6min 量级,166k 32 worker 约 1h |
| Step 3 metric_cache(可选) | ~2–4 h | 仅 RFT/PDMS 需要;ray 分布式 |
| **合计(到能训 SFT)** | **~6–10 h** | = Step1 + Step2;**Step1 下载占绝对大头** |

**注**:22.6 MB/s 是本沙盒实测值,brev 训练机若带宽更高,下载会更快;主要变量就是这条链路带宽。

---

## 4. 前置与风险清单

- **磁盘**:`/data` 现余 **5.3T**,装全量 trainval sensor(~2TB)+ 166k JSON(~几十 G)绰绰有余。
- **静默截断(最重要)**:流式下载会丢分片且不报错 → **必须落盘+`gzip -t`+`verify` 对账**,每步核对产出数量。
- **只补差量做不到**:分片非 log 对齐,只能重下全 2TB(现有 446G 无法复用)。
- **收益判断**:166k 相比 103k 多 ~60% 场景,但要多下 ~1.5T 净新数据。若不为严格复现 Table S1,
  当前混训(nuplan 103k + nuScenes 19k)已在 nuScenes 拿到 L2 0.64m(优于纯 nuScenes 0.68m),
  多域收益大于"nuplan 从 103k→166k"。要不要上 166k,取决于是否需要对齐论文的 nuPlan 规模。
