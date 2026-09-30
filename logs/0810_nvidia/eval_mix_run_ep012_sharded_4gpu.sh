#!/bin/bash
# ============================================================================
# 对混训 run 2026-08-11_07-35-22 的 epoch0/1/2 做【nuScenes L2+Collision】评测。
# 每个 epoch 切成【3 片】并行 => 共 9 个 job，round-robin 分到【4 张 GPU】(0-3)。
# 全部跑完后逐 epoch 合并 3 片 => 每个 epoch 一张最终表。
#
#   bash logs/0810_nvidia/eval_mix_run_ep012_sharded_4gpu.sh
# ============================================================================
set -u

REPO=/home/nvidia/workspace/doc_drive_search/other_repo/AutoVLA
PY=/data/autovla_data/envs/autovla/bin/python
cd "$REPO"

RUN=runs/sft/2026-08-11_07-35-22
CONFIG=config/training/0916_nvidia/qwen2.5-vl-3B-mix-nuplan-nuscenes-nocot-sft.yaml   # data.val = nuScenes val
SEG_DATA=/data/autovla_data/nuscenes/nusc_eval_seg/nusc_eval_seg_6s
NUM_SHARDS=3
GPUS=(0 1 2 3)                      # 只用 4 卡
STAMP=$(date +%Y-%m-%d_%H-%M-%S)
OUTDIR="$REPO/logs/0810_nvidia/eval_mix_ep012_${STAMP}"
mkdir -p "$OUTDIR"

# epoch -> ckpt
declare -A CKPT
CKPT[0]="$RUN/epoch=0-loss=1.2551.ckpt"
CKPT[1]="$RUN/epoch=1-loss=1.2868.ckpt"
CKPT[2]="$RUN/epoch=2-loss=1.4139.ckpt"

# --- preflight ---
for e in 0 1 2; do [ -f "${CKPT[$e]}" ] || { echo "ERROR: 找不到 ${CKPT[$e]}"; exit 1; }; done
[ -d "$SEG_DATA" ] || { echo "ERROR: seg data 不存在: $SEG_DATA"; exit 1; }

echo "=================================================================="
echo " 混训 run epoch0/1/2 分片评测  (nuScenes L2+Collision)"
echo " run      : $RUN"
echo " 每 epoch : $NUM_SHARDS 片；共 9 job；GPU: ${GPUS[*]} (round-robin)"
echo " 输出目录 : $OUTDIR"
echo "=================================================================="

# --- 启动 9 个 job ---
jobidx=0
pids=()
for e in 0 1 2; do
  for s in $(seq 0 $((NUM_SHARDS-1))); do
    gpu=${GPUS[$((jobidx % ${#GPUS[@]}))]}
    dump="$OUTDIR/raw_ep${e}_shard${s}.pt"
    log="$OUTDIR/ep${e}_shard${s}.log"
    echo "  job#$jobidx  epoch=$e shard=$s -> GPU$gpu  ($dump)"
    CUDA_VISIBLE_DEVICES=$gpu TOKENIZERS_PARALLELISM=false \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      NUPLAN_MAPS_ROOT=/data/autovla_data/nuplan/maps \
      NUPLAN_MAP_VERSION=nuplan-maps-v1.0 \
      OPENSCENE_DATA_ROOT=/data/autovla_data/nuplan \
      NAVSIM_DEVKIT_ROOT="$REPO/navsim" \
      $PY tools/eval/nusc_eval.py \
        --config "$CONFIG" \
        --checkpoint "${CKPT[$e]}" \
        --seg_data_path "$SEG_DATA" \
        --num_shards $NUM_SHARDS --shard_id $s \
        --dump_raw "$dump" \
        --output "$OUTDIR/shardtable_ep${e}_shard${s}.txt" \
        --device cuda:0 > "$log" 2>&1 &
    pids+=($!)
    jobidx=$((jobidx+1))
    sleep 3                        # 错开加载，避免同卡瞬时峰值
  done
done

echo
echo "已启动 ${#pids[@]} 个 job，PID: ${pids[*]}"
echo "等待全部完成..."
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "全部结束（失败 $fail 个）。"

# --- 逐 epoch 合并 3 片 ---
FINAL="$OUTDIR/FINAL_ep012_results.txt"
echo
echo "=================================================================="
echo " 合并结果"
echo "=================================================================="
for e in 0 1 2; do
  shards=$(ls "$OUTDIR"/raw_ep${e}_shard*.pt 2>/dev/null)
  n=$(echo "$shards" | grep -c pt)
  echo "---- epoch $e ($n 片) ----"
  $PY tools/eval/nusc_merge_shards.py --shards $shards --tag "epoch$e" --output "$FINAL" 2>&1 \
    | grep -avE "FutureWarning|weights_only|warnings.warn"
done
echo
echo "=================================================================="
echo " 最终三 epoch 汇总表: $FINAL"
echo " 各 job 日志/分片表:  $OUTDIR/"
echo "=================================================================="
