#!/bin/bash
# ============================================================================
# 评测【nuPlan CoT SFT】ckpt on navtest（PDMS），greedy 解码。
#   9 个 job（navtest 切 9 片）→ 3 张 GPU，每卡 3 个 job。
#
# ckpt : runs/sft/2026-09-02_05-47-39/epoch=2-loss=0.4732.ckpt  (use_cot=true)
# GPU  : 默认 4,5,6（0-3 有别的任务在跑）。每卡 3 进程 ≈ 3×23.5G=70G < 80G。
# greedy: 用 AUTOVLA_SAMPLE_TOP_K=1 —— 候选集只剩 argmax，严格贪心，无需改代码。
#
#   bash scripts/0901/eval_cot_sft_greedy_9job_3gpu.sh
#   LIMIT=1000 bash ...            # 只评抽样 1000（seed=42，可跟 89.06 对比）
#   GPULIST=4,5,6 bash ...         # 换卡
# ============================================================================
set -uo pipefail

REPO=/home/nvidia/workspace/doc_drive_search/other_repo/AutoVLA
DATA=/data/autovla_data/nuplan
PY=/data/autovla_data/envs/autovla/bin/python

CKPT="${CKPT:-$REPO/runs/sft/2026-09-02_05-47-39/epoch=2-loss=0.4732.ckpt}"
NSHARD=${NSHARD:-9}
GPULIST="${GPULIST:-4,5,6}"          # 3 张卡
PROC_PER_GPU=3                        # 每卡 3 个 job = 9
LIMIT="${LIMIT:-0}"                   # 0 = 全量 navtest(12146)；给数字=seed=42 抽样
SEED="${SEED:-42}"
EVAL_CONFIG="${EVAL_CONFIG:-$REPO/config/eval/0721/autovla-navtest-eval.yaml}"   # 默认 CoT;no-CoT ckpt 传 nocot config(config/eval/0723/autovla-navtest-eval-nocot.yaml)

export PYTHONPATH="$REPO/navsim:${PYTHONPATH:-}"
export NAVSIM_DEVKIT_ROOT="$REPO/navsim"
export OPENSCENE_DATA_ROOT="$DATA"
export NUPLAN_MAPS_ROOT="$DATA/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export TOKENIZERS_PARALLELISM=false
export AUTOVLA_SAMPLE_TOP_K=1          # ★★ greedy：top_k=1 → 只保留 argmax

CACHE_PATH="$DATA/navtest_metric_cache"      # reorg 后是 symlink→eval/，正常解析
JSON_DATA_PATH="$DATA/navtest_nocot"
SENSOR_DATA_PATH="$DATA/sensor_blobs/test"
FILTER_DIR="$REPO/navsim/navsim/planning/script/config/common/train_test_split/scene_filter"
# 按 ckpt 自动隔离输出/scene_filter，避免多 ckpt 评测时 CSV 混在一起聚合
CKPT_ID=$(basename "$CKPT" .ckpt | sed 's/[=.-]/_/g')      # 如 epoch_3_loss_0_4446
TAG="${TAG:-cg_${CKPT_ID}}"
WORK="${WORK:-$DATA/../eval/nuplan/cot_sft_greedy_eval/${CKPT_ID}}"     # 输出到 eval/<ckpt>/ 下
mkdir -p "$WORK"
cd "$REPO"

# --- preflight ---
fail=0
for f in "$CKPT" "$CACHE_PATH" "$JSON_DATA_PATH" "$SENSOR_DATA_PATH" "$EVAL_CONFIG" "$NUPLAN_MAPS_ROOT"; do
    [ -e "$f" ] || { echo "❌ 缺失: $f"; fail=1; }
done
[ "$fail" -eq 0 ] || { echo "前置检查未通过"; exit 1; }
USECOT=$($PY -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['model']['use_cot'])" "$EVAL_CONFIG")
echo "  eval config use_cot=$USECOT (确保与 ckpt 类型一致:CoT ckpt→True, no-CoT ckpt→False)"

USE_LORA="${USE_LORA:-false}"
LORA_ARGS="agent.lora_conf.use_lora=${USE_LORA}"
if [ "$USE_LORA" = "true" ]; then
  LORA_ARGS="$LORA_ARGS agent.lora_conf.r=${LORA_R:-8} agent.lora_conf.lora_alpha=${LORA_ALPHA:-8}"
fi
IFS=',' read -ra GPUS <<< "$GPULIST"
NGPU=${#GPUS[@]}
echo "=================================================================="
echo " CoT SFT greedy 评测 (PDMS on navtest)"
echo " ckpt      : $CKPT"
echo " eval cfg  : $EVAL_CONFIG (use_cot=$USECOT)"
echo " 解码      : greedy (AUTOVLA_SAMPLE_TOP_K=1)"
echo " 分片      : $NSHARD job → GPU [${GPULIST}] 各 $PROC_PER_GPU 个"
echo " 场景      : $([ "$LIMIT" -eq 0 ] && echo '全量 navtest(12146)' || echo "$LIMIT (seed=$SEED)")"
echo " 输出      : $WORK"
echo "=================================================================="

# --- 生成 9 个分片 scene_filter（取模切分，难度均匀）---
TAG="$TAG" $PY - "$NSHARD" "$FILTER_DIR" "$LIMIT" "$SEED" <<'PY'
import sys, yaml, os, random
n, fdir, limit, seed = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
tag = os.environ["TAG"]
cfg = yaml.safe_load(open(os.path.join(fdir, "navtest.yaml")))
toks = cfg["tokens"]
if limit > 0 and limit < len(toks):
    random.Random(seed).shuffle(toks); toks = toks[:limit]
for i in range(n):
    shard = dict(cfg); shard["tokens"] = toks[i::n]
    with open(os.path.join(fdir, f"{tag}_shard{i}of{n}.yaml"), "w") as f:
        yaml.safe_dump(shard, f, default_flow_style=False, sort_keys=False)
    print(f"  shard {i}: {len(shard['tokens']):,} tokens")
print(f"合计 {len(toks):,}")
PY
for i in $(seq 0 $((NSHARD-1))); do
    printf 'defaults:\n  - scene_filter: %s_shard%dof%d\n\ndata_split: test\n' "$TAG" "$i" "$NSHARD" \
        > "$FILTER_DIR/../${TAG}_shard${i}of${NSHARD}.yaml"
done

# --- 启动 9 个 job：job s → GPU 取模，每卡 3 个 ---
echo; echo "启动 $NSHARD 个 job ..."
for s in $(seq 0 $((NSHARD-1))); do
    gpu=${GPUS[$((s % NGPU))]}
    d="$WORK/shard$s"; mkdir -p "$d"
    echo "  job$s -> GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu NAVSIM_EXP_ROOT="$d" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_cot.py" \
        train_test_split="${TAG}_shard${s}of${NSHARD}" \
        agent=autovla_agent \
        agent.config_path="$EVAL_CONFIG" \
        agent.checkpoint_path="'$CKPT'" \
        agent.sensor_data_path="$SENSOR_DATA_PATH" \
        $LORA_ARGS \
        metric_cache_path="$CACHE_PATH" \
        +json_data_path="$JSON_DATA_PATH" \
        experiment_name="cotgreedy_s${s}" \
        > "$d/run.log" 2>&1 &
    sleep 3
done

echo "已启动 $NSHARD 个进程，等待全部完成 ..."
wait

# --- 聚合 9 片 → PDMS ---
echo; echo "==================== 结果 ===================="
$PY - "$WORK" "$CKPT" <<'PY'
import sys, glob, os
import pandas as pd
work, ckpt = sys.argv[1], sys.argv[2]
csvs = glob.glob(os.path.join(work, "shard*/**/*.csv"), recursive=True)
if not csvs:
    print("⚠️ 没有结果 CSV，检查 shard*/run.log"); sys.exit(1)
df = pd.concat([pd.read_csv(x) for x in csvs], ignore_index=True)
df = df[df.token.notna()].drop_duplicates(subset="token")
df.to_csv(os.path.join(work, "cotgreedy_merged.csv"), index=False)
cols = [("score","PDMS"),("no_at_fault_collisions","NC"),("drivable_area_compliance","DAC"),
        ("ego_progress","EP"),("time_to_collision_within_bound","TTC"),("comfort","C"),
        ("driving_direction_compliance","DDC")]
print(f"ckpt: {os.path.basename(ckpt)}   n={len(df)}")
print(f"{'':<12}" + "".join(f"{z:>8}" for _,z in cols))
print("".join(f"{df[c].mean()*100:8.2f}" for c,_ in cols))
print("\n参照: 发布 RFT ckpt 同批 1000 场景 PDMS=89.06 / 全量 12126 上 89.48")
print(f"合并 CSV: {os.path.join(work,'cotgreedy_merged.csv')}")
PY
