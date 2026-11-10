#!/bin/bash
# ============================================================================
# [mh 2026/07/25] 单卡（GPU2）评测一个 SFT checkpoint —— navtest PDMS
#
# 模仿 scripts/0723/eval_sft_ckpts.sh，但为【本机】改写：
#   * 数据根 /scratch/mh2803/vla/nuplan（那台机器是 /data/autovla_data）
#   * python 用 autovla_codeclean 环境
#   * eval config 用本机版 config/eval/0725/（pretrained 指向 ./Qwen2.5-VL-3B-Instruct）
#
# 布局：只有一张卡（GPU2），把 navtest 场景切成 3 片、3 个进程并发压在 GPU2 上提速，
#       跑完合并成【一个】PDMS。3B 模型单进程约 7GB，GPU2 有 97GB，3 并发绰绰有余。
#
# 用法:
#   bash scripts/0725/eval_ckpt_gpu2.sh                    # 默认 best ckpt，1000 场景(seed42)
#   LIMIT=12146 bash scripts/0725/eval_ckpt_gpu2.sh        # 跑全量 12,146
#   CKPT=/abs/path/to/x.ckpt bash scripts/0725/eval_ckpt_gpu2.sh   # 指定 ckpt
#   GPU=3 NSHARD=4 bash scripts/0725/eval_ckpt_gpu2.sh     # 换卡/换并发数
#   WAIT=1 bash scripts/0725/eval_ckpt_gpu2.sh             # ckpt 还没生成时，轮询等待
#
# 注意：
#   * LIMIT=1000 用 seed=42 抽样，与 0723 脚本【完全相同】的 1000 个场景，
#     可直接跟发布 ckpt 的 PDMS 89.06 对比。LIMIT=12146 或 0 = 全量。
#   * use_cot=false（本 ckpt 是 no-CoT SFT）；LoRA 关闭（全参 SFT，无 adapter）。
# ============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "$(readlink -f "$0")")" && while [ ! -f setup.py ] && [ "$PWD" != / ]; do cd ..; done; pwd)"
DATA=/scratch/mh2803/vla/nuplan
PY=/home/mh2803/miniconda3/envs/autovla_codeclean/bin/python

RUN_DIR="${RUN_DIR:-$REPO/runs/sft/2026-07-25_05-05-57}"   # checkpoint 所在目录
CKPT="${CKPT:-}"                    # 留空则自动挑 RUN_DIR 里 val_loss 最低的那个
GPU="${GPU:-2}"                     # 单张卡
GPUS="${GPUS:-$GPU}"               # [mh 2026-11-06] 多卡：GPUS="0 1 2" 时第 s 片放在第 s%n 张卡上；不设 = 全压在 $GPU 上（原行为）
read -ra GPU_LIST <<< "$GPUS"
NSHARD="${NSHARD:-3}"              # 并发进程数 = 分片数（都压在同一张 GPU 上）
LIMIT="${LIMIT:-1000}"            # 1000=可比子集(seed42)；12146 或 0 = 全量
SEED="${SEED:-42}"
WAIT="${WAIT:-0}"                 # 1 = ckpt 还没出现时轮询等待
EVAL_CONFIG="${EVAL_CONFIG:-$REPO/config/eval/0725/autovla-navtest-eval-nocot.yaml}"

export PYTHONPATH="$REPO/navsim:${PYTHONPATH:-}"
export NAVSIM_DEVKIT_ROOT="$REPO/navsim"
export OPENSCENE_DATA_ROOT="$DATA"
export NUPLAN_MAPS_ROOT="$DATA/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export TOKENIZERS_PARALLELISM=false

CACHE_PATH="$DATA/navtest_metric_cache"
JSON_DATA_PATH="$DATA/navtest_nocot"
SENSOR_DATA_PATH="$DATA/sensor_blobs/test"
FILTER_DIR="$REPO/navsim/navsim/planning/script/config/common/train_test_split/scene_filter"
WORK="$DATA/${WORK_SUBDIR:-sft_eval_0725}"
mkdir -p "$WORK"
cd "$REPO"

# ---- 选 checkpoint：没指定就挑 val_loss 最低的（文件名 epoch=E-loss=L.ckpt）----
if [ -z "$CKPT" ]; then
    # ckpt 可能还没落盘（训练刚存），WAIT=1 时轮询等
    tries=0
    while : ; do
        CKPT=$(ls "$RUN_DIR"/*.ckpt 2>/dev/null \
               | sed -E 's/.*loss=([0-9.]+)\.ckpt/\1 &/' \
               | sort -n | head -1 | cut -d' ' -f2-)
        [ -n "$CKPT" ] && break
        if [ "$WAIT" != "1" ]; then
            echo "❌ $RUN_DIR 下没有 .ckpt。训练还没存权重？可 WAIT=1 轮询等待，或用 CKPT=... 指定。"; exit 1
        fi
        tries=$((tries+1)); echo "  [$tries] 等待 ckpt 出现... ($RUN_DIR)"; sleep 60
    done
fi

# ---- 前置检查：缺什么直接停 ----
fail=0
for f in "$CACHE_PATH" "$JSON_DATA_PATH" "$SENSOR_DATA_PATH" "$EVAL_CONFIG" "$NUPLAN_MAPS_ROOT" "$CKPT"; do
    [ -e "$f" ] || { echo "❌ 缺失: $f"; fail=1; }
done
[ "$fail" -eq 0 ] || { echo "前置检查未通过。"; exit 1; }

USECOT=$($PY -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['model']['use_cot'])" "$EVAL_CONFIG")
NAME=$(basename "$CKPT" .ckpt | tr '=' '_' | tr '.' '_')
echo "=================================================================="
echo " checkpoint  : $CKPT"
echo " eval config : $EVAL_CONFIG   (use_cot=$USECOT，no-CoT SFT 须为 False)"
echo " GPU         : $GPUS   |   分片/并发: $NSHARD"
echo " 场景数      : $LIMIT   (seed=$SEED 随机抽样；12146/0=全量)"
echo " 输出        : $WORK/$NAME"
echo "=================================================================="

# ---- 生成 NSHARD 个 scene_filter 分片（seed 抽样后交错切分）----
TAG="${WORK_SUBDIR:-sfteval0725}"
TAG="$TAG" $PY - "$NSHARD" "$FILTER_DIR" "$LIMIT" "$SEED" <<'PY'
import sys, yaml, os, random
n, fdir, limit, seed = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
tag = os.environ["TAG"]
cfg = yaml.safe_load(open(os.path.join(fdir, "navtest.yaml")))
toks = cfg["tokens"]
if limit > 0 and limit < len(toks):
    # 与 0723 脚本一致的 seed=42 抽样，1000 场景结果才可跟 89.06 对比
    random.Random(seed).shuffle(toks)
    toks = toks[:limit]
for i in range(n):
    shard = dict(cfg); shard["tokens"] = toks[i::n]   # 交错切分，各片分布更均匀
    with open(os.path.join(fdir, f"{tag}_shard{i}of{n}.yaml"), "w") as f:
        yaml.safe_dump(shard, f, default_flow_style=False, sort_keys=False)
    print(f"  shard {i}: {len(shard['tokens']):,} tokens")
print(f"合计 {len(toks):,}")
PY
for i in $(seq 0 $((NSHARD-1))); do
    printf 'defaults:\n  - scene_filter: %s_shard%dof%d\n\ndata_split: test\n' "$TAG" "$i" "$NSHARD" \
        > "$FILTER_DIR/../${TAG}_shard${i}of${NSHARD}.yaml"
done
echo

# ---- 启动 NSHARD 个进程，全部压在同一张 GPU 上 ----
# checkpoint_path 外面那层单引号：ckpt 名含 '='（epoch=4-loss=1.05），
# hydra override 会把它当 key=value 分隔 -> "mismatched input '='"，加引号规避。
rm -rf "$WORK/$NAME"/shard*
for s in $(seq 0 $((NSHARD-1))); do
    d="$WORK/$NAME/shard$s"; mkdir -p "$d"
    g=${GPU_LIST[$((s % ${#GPU_LIST[@]}))]}
    echo "  GPU $g  <-  $NAME  shard$s"
    CUDA_VISIBLE_DEVICES=$g NAVSIM_EXP_ROOT="$d" \
    $PY "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_cot.py" \
        train_test_split="${TAG}_shard${s}of${NSHARD}" \
        agent=autovla_agent \
        agent.config_path="$EVAL_CONFIG" \
        agent.checkpoint_path="'$CKPT'" \
        agent.sensor_data_path="$SENSOR_DATA_PATH" \
        agent.lora_conf.use_lora=false \
        metric_cache_path="$CACHE_PATH" \
        +json_data_path="$JSON_DATA_PATH" \
        experiment_name="${NAME}_s${s}" \
        > "$d/run.log" 2>&1 &
done

echo
echo "已启动 $NSHARD 个进程（GPU：$GPUS）。等待..."
wait

# ---- 合并所有分片 -> 单个 PDMS ----
echo
echo "==================== 结果 ===================="
$PY - "$WORK/$NAME" "$CKPT" <<'PY'
import sys, glob, os
import pandas as pd
work, ckpt = sys.argv[1], sys.argv[2]
csvs = glob.glob(os.path.join(work, "shard*/**/*.csv"), recursive=True)
if not csvs:
    print("⚠️ 没有结果，检查各 shard 的 run.log"); sys.exit(1)
df = pd.concat([pd.read_csv(x) for x in csvs], ignore_index=True)
df = df[df.token.notna()].drop_duplicates(subset="token")
merged = os.path.join(work, "merged.csv")
df.to_csv(merged, index=False)
cols = [("score","PDMS"),("no_at_fault_collisions","NC"),("drivable_area_compliance","DAC"),
        ("ego_progress","EP"),("time_to_collision_within_bound","TTC"),("comfort","C"),
        ("driving_direction_compliance","DDC")]
print(f"checkpoint : {os.path.basename(ckpt)}")
print(f"场景数     : {len(df):,}   (valid {int(df['valid'].sum()):,})" if 'valid' in df else f"场景数     : {len(df):,}")
print()
print("".join(f"{z:>8}" for _,z in cols))
print("-"*(8*len(cols)))
print("".join(f"{df[c].mean()*100:8.2f}" for c,_ in cols))
print(f"\n合并结果 -> {merged}")
print("参照: 发布 RFT ckpt 同一批 1000 场景(seed42) PDMS=89.06；全量 12,126 上 89.48")
PY
