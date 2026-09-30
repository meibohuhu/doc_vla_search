#!/bin/bash
# ============================================================================
# 【nuScenes】open-loop 评测：L2 (m) + Collision Rate (%)
#   —— autoregressive 生成完整轨迹，用 UniAD 那套 PlanningMetric 算 L2/碰撞。
#   注意：这【不是】PDMS。nuScenes 报的就是 val split 上的 L2+Collision（社区惯例，
#         test split 无公开真值）。
#
# 用法：
#   bash logs/0810_nvidia/eval_nuscenes_l2_collision.sh                 # 评最优 ckpt 全量
#   CKPT=/path/to/x.ckpt bash logs/0810_nvidia/eval_nuscenes_l2_collision.sh
#   NUM=50 bash logs/0810_nvidia/eval_nuscenes_l2_collision.sh          # 快速冒烟 50 条
#   GPU=1 bash logs/0810_nvidia/eval_nuscenes_l2_collision.sh           # 换 GPU
#
# 依赖（均已就位）：
#   * UniAD 分割数据 nusc_eval_seg_6s（{token}.pt，6019 个，覆盖 val 100%）
#   * ckpt 加载已修复静默 bug（tools/eval/nusc_eval.py：strip "autovla." 前缀 + 断言）
# ============================================================================
set -u

REPO=/home/nvidia/workspace/doc_drive_search/other_repo/AutoVLA
PY=/data/autovla_data/envs/autovla/bin/python
cd "$REPO"

# --- 可调参数（env 覆盖）---
GPU=${GPU:-0}
NUM=${NUM:-}                                   # 空 = 全量；给数字 = 只评前 N 条（冒烟）
# nusc_eval 只读 config 的 model + data.val，直接复用训练 config（val 路径已是 /data）
CONFIG=${CONFIG:-config/training/0916_nvidia/qwen2.5-vl-3B-nuscenes-nocot-sft.yaml}
SEG_DATA=${SEG_DATA:-/data/autovla_data/nuscenes/nusc_eval_seg/nusc_eval_seg_6s}

# --- CKPT：默认取最新 run 里 val_loss 最低的那个 ---
if [ -z "${CKPT:-}" ]; then
  RUN_DIR=$(ls -dt "$REPO"/runs/sft/*/ 2>/dev/null | head -1)
  # 文件名形如 epoch=2-loss=1.3867.ckpt，按 loss 排序取最小
  CKPT=$(ls "$RUN_DIR"*.ckpt 2>/dev/null | sed 's/.*loss=//; s/\.ckpt$//' | paste -d' ' - <(ls "$RUN_DIR"*.ckpt) | sort -n | head -1 | awk '{print $2}')
fi

# --- preflight ---
[ -f "$CKPT" ] || { echo "ERROR: 找不到 ckpt: ${CKPT:-<空>}"; exit 1; }
[ -d "$SEG_DATA" ] || { echo "ERROR: 找不到 seg data 目录: $SEG_DATA"; exit 1; }
n_seg=$(find "$SEG_DATA" -maxdepth 1 -name '*.pt' | wc -l)
[ "$n_seg" -eq 0 ] && { echo "ERROR: $SEG_DATA 里没有 .pt"; exit 1; }
[ -f "$CONFIG" ] || { echo "ERROR: 找不到 config: $CONFIG"; exit 1; }

export CUDA_VISIBLE_DEVICES=$GPU
export TOKENIZERS_PARALLELISM=false
export NUPLAN_MAPS_ROOT=/data/autovla_data/nuplan/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export OPENSCENE_DATA_ROOT=/data/autovla_data/nuplan
export NAVSIM_DEVKIT_ROOT="$REPO/navsim"

STAMP=$(date +%Y-%m-%d_%H-%M-%S)
CKPT_TAG=$(basename "$(dirname "$CKPT")")_$(basename "$CKPT" .ckpt)
OUT="$REPO/logs/0810_nvidia/nusc_eval_${CKPT_TAG}_${STAMP}.txt"
LOG="$REPO/logs/0810_nvidia/nusc_eval_${CKPT_TAG}_${STAMP}.log"

NUM_ARG=""
[ -n "$NUM" ] && NUM_ARG="--num_samples $NUM"

echo "=================================================================="
echo " nuScenes L2 + Collision 评测"
echo " ckpt     : $CKPT"
echo " config   : $CONFIG"
echo " seg data : $SEG_DATA  ($n_seg .pt)"
echo " GPU      : $GPU   ${NUM:+(仅前 $NUM 条冒烟)}"
echo " 结果表   : $OUT"
echo " 日志     : $LOG"
echo "=================================================================="

$PY tools/eval/nusc_eval.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --seg_data_path "$SEG_DATA" \
    --output "$OUT" \
    --device cuda:0 \
    $NUM_ARG 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}

echo
echo "=================================================================="
echo " exit code : $STATUS"
echo " 结果表    : $OUT"
echo "   L2 越小越好；Collision(碰撞率) 越小越好。对比对象：UniAD/VAD/AutoVLA 论文表。"
echo "=================================================================="
exit "$STATUS"
