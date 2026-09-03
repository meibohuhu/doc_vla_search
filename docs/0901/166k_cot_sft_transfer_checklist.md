# 在另一台机器上跑全量 166k CoT SFT —— 需要哪些文件

日期 2026-09-03 · 配套 [nuplan_103k_to_166k.md](nuplan_103k_to_166k.md) · 启动脚本 `logs/0902/run_cot_sft_166k_0902.sh`

> 一句话:**代码 + 3B 底座 + 两份 JSON(1.3G)+ sensor 图(1.4T)**。
> 最大的坑:JSON 里 camera 是**绝对路径**,换机器要么同路径、要么 `sed` 改写。

---

## 1. 必须搬的东西

| # | 项目 | 大小 | 说明 |
|---|---|---|---|
| **A. 代码** | | | |
| 1 | AutoVLA repo(整个) | ~几百 MB | 含 `tools/run_sft.py`、`models/`、`dataset_utils/`、`navsim/`(devkit,import 要用) |
| 2 | `config/training/qwen2.5-vl-3B-nuplan-cot-sft-trainval166k-brev.yaml` | 1 KB | 训练 config(在 repo 里,①带着就有) |
| 3 | `logs/0902/run_cot_sft_166k_0902.sh` | 1 KB | 启动脚本(在 repo 里) |
| 4 | `codebook_cache/agent_vocab.pkl` | **1.2 MB** | ★ action 码本,**必须**,少了 action token 全错 |
| **B. 底座 & 环境** | | | |
| 5 | `Qwen2.5-VL-3B-Instruct` | **7.1 GB** | 基座 VLM。可从 HF 重下 `Qwen/Qwen2.5-VL-3B-Instruct`,repo 里放到 `./Qwen2.5-VL-3B-Instruct` |
| 6 | conda env | 7.3 GB | 用 repo 的 `environment.yml` 重建更干净(env 跟机器走);跳过 flash-attn/waymo-tf/autoawq |
| **C. 数据(大头)** | | | |
| 7 | `trainval_cot_166k/`(train JSON) | **1.3 GB** | 164,282 条,已灌 CoT `cot_output`,camera 绝对路径 |
| 8 | `trainval_cot_166k_val/`(val JSON) | **16 MB** | 2,000 条,166k 内部切,无泄漏 |
| 9 | `sensor_blobs/trainval/`(图) | **1.4 TB** | ★★ 实际 JPEG 像素,ViT 输入。**最大项** |

**合计**:小件(1-8)约 **~17 GB**;大件(9)**1.4 TB**。

---

## 2. sensor 1.4TB 怎么过去(二选一)

- **A. rsync 1.4T**:两机同机房/快链路时最省事。
  `rsync -a --info=progress2 <本机>:/data/autovla_data/nuplan/sensor_blobs/trainval/ <目标>/sensor_blobs/trainval/`
- **B. 目标机重下 2TB**(推荐跨网时):把 `scripts/0901/download_trainval_full.sh`(已修并行解压竞态)拷过去跑。
  实测本机 ~300MB/s 约 1.5–2h。**下完必跑帧级校验**(见 §4),别信 log 级 verify。

> 不需要搬:`metric_cache`(仅 RFT/PDMS 评测要)、`navsim_logs`/metadata(仅预处理/生成 CoT 要,已做完)、
> `maps`(训练**不读**地图 —— 实测 SFTDataset/AutoVLAAgent 训练路径不 load map)。

---

## 3. 🔴 换机器的关键一步:改 camera 绝对路径

JSON 里存的是**本机绝对路径** `/data/autovla_data/nuplan/sensor_blobs/trainval/...`。目标机:

- **若 sensor 放在同样的绝对路径** → 直接能用,啥都不用改。
- **若路径不同** → 批量 `sed`(和当初 `/scratch/mh2803`→`/data` 一个道理):
  ```bash
  NEW=/目标机/nuplan            # 里面应有 sensor_blobs/trainval/
  grep -rl '/data/autovla_data/nuplan' trainval_cot_166k trainval_cot_166k_val \
    | xargs -P8 -n500 sed -i "s#/data/autovla_data/nuplan#${NEW}#g"
  ```
  改完 config 的 `sensor_data_path` 保持 **null**(JSON 已是绝对路径,不能再拼前缀)。

---

## 4. 上机前自检(必做,防静默失败)

```bash
# ① camera 图帧级完整(sensor 残缺会让 ~9% 样本训练时找不到图)
python - <<'EOF'
import json,glob,os,random
fs=glob.glob("<train_dir>/*.json"); random.seed(0); random.shuffle(fs)
m=sum(1 for f in fs[:5000] for c in('front_camera_paths','front_left_camera_paths','front_right_camera_paths')
      for p in json.load(open(f))[c] if not os.path.exists(p))
print("缺图路径:",m,"(应 0)")
EOF
# ② cot_output 覆盖率(启动脚本 preflight 已含,≥90%)
# ③ global batch=32:4 卡 × accum 8;换卡数必须同步改 config 的 accumulate_grad_batches
```

---

## 5. 硬件要求

- **fp32_master → 单卡峰值 ~76 GB** → 需 **80GB GPU(A100-80G 级)**。
- **4 卡**跑 global batch 32(`accum 8`);卡数变了就改 `accum` 保持 32(启动脚本会强制核对)。
- ViT 冻结 + LLM 全参,DDP。

---

## 6. 跑起来

```bash
# 目标机 repo 根目录
export CUDA_VISIBLE_DEVICES=...           # 或用脚本默认
tmux new -s cot166k
bash logs/0902/run_cot_sft_166k_0902.sh   # GPUS=0,1,2,3 bash ... 换卡
```

启动脚本自带 preflight:数据存在 / cot_output 覆盖率 / camera 帧级完整 / global batch=32,任一不过直接退出。
wandb project `autovla-cot-sft-166k`,ckpt 存 `runs/sft/<时间戳>/`,按 val_loss 存 top-3。
