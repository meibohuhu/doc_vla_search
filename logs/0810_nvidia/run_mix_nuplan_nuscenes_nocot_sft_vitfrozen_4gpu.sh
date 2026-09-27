#!/bin/bash
# ============================================================================
# no-CoT SFT 混训【nuPlan 101k + nuScenes 19k = 120k】，ViT FROZEN + LLM full-param
#   验证集【只用 nuScenes val (5569)】—— 与最终评测口径（nuScenes L2/Collision）一致。
# 本机(brev, 8xA100-SXM4-80G)，4 卡 DDP（GPU 0-3）。
#
#   tmux new -s mix
#   bash logs/0810_nvidia/run_mix_nuplan_nuscenes_nocot_sft_vitfrozen_4gpu.sh
#
# 与纯 nuScenes 版(run_nuscenes_nocot_sft_vitfrozen_4gpu.sh)的差异：
#   * data.train 从【单路径】变【nuPlan + nuScenes 两路径列表】(SFTDataset 原生支持)
#   * config -> qwen2.5-vl-3B-mix-nuplan-nuscenes-nocot-sft
#   * val 不变（仍是 nuScenes val）
#   * 数据量 19k -> 120k，单 epoch 约 6.3x，墙钟相应变长
#   * ⚠️ 比例 nuPlan:nuScenes ≈ 5.3:1，nuScenes 仅占 ~16%（如需上采样要改代码）
#   * 评测同样走 tools/eval/nusc_eval.py（L2+Collision），非 PDMS
# ============================================================================
set -u

REPO=/home/nvidia/workspace/doc_drive_search/other_repo/AutoVLA
PY=/data/autovla_data/envs/autovla/bin/python
CONFIG="training/qwen2.5-vl-3B-mix-nuplan-nuscenes-nocot-sft"
TAG="mix_nuplan_nuscenes_nocot_sft_vitfrozen"

cd "$REPO"

# --- preflight: 两个训练目录 + nuScenes val 必须存在 ---
NUPLAN_DIR=/data/autovla_data/nuplan/navtrain_nocot
NUSC_DIR=/data/autovla_data/nuscenes/nuscenes_train
VAL_DIR=/data/autovla_data/nuscenes/nuscenes_val
count_json() { find "$1" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l; }
n_nuplan=$(count_json "$NUPLAN_DIR")
n_nusc=$(count_json "$NUSC_DIR")
n_val=$(count_json "$VAL_DIR")
[ "$n_nuplan" -eq 0 ] && { echo "ERROR: $NUPLAN_DIR 为空"; exit 1; }
[ "$n_nusc"   -eq 0 ] && { echo "ERROR: $NUSC_DIR 为空"; exit 1; }
[ "$n_val"    -eq 0 ] && { echo "ERROR: $VAL_DIR 为空"; exit 1; }
[ -e "$REPO/Qwen2.5-VL-3B-Instruct" ] || { echo "ERROR: 找不到 Qwen2.5-VL-3B-Instruct"; exit 1; }
# 抽查两个数据集的 camera 绝对路径真实存在
for D in "$NUPLAN_DIR" "$NUSC_DIR"; do
  S=$(find "$D" -maxdepth 1 -name '*.json' | head -1)
  $PY -c "
import json,os
d=json.load(open('$S')); p=d['front_camera_paths'][0]
assert os.path.isabs(p) and os.path.exists(p), f'camera 路径不存在: {p}'
print(f'  preflight OK: $D  dataset_name={d[\"dataset_name\"]}')
" || { echo "ERROR: $D 数据自检失败"; exit 1; }
done
n_train=$((n_nuplan + n_nusc))

# --- 4 GPUs ---
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
# A100-SXM4 有 NVLink，保持 NCCL 默认

export NUPLAN_MAPS_ROOT=/data/autovla_data/nuplan/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export OPENSCENE_DATA_ROOT=/data/autovla_data/nuplan
export NAVSIM_DEVKIT_ROOT="$REPO/navsim"
export WANDB_PROJECT=autovla-mix-nuplan-nuscenes-sft

ACCUM=$($PY -c "import yaml;print(yaml.safe_load(open('config/${CONFIG}.yaml'))['training']['accumulate_grad_batches'])")
GB=$((1 * ACCUM * 4))

STAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/0810_nvidia/${TAG}_${STAMP}.log"

echo "=================================================================="
echo " 混训 no-CoT SFT: nuPlan + nuScenes  (ViT frozen, LLM full-param)"
echo " config   : config/${CONFIG}.yaml"
echo " GPUs     : 0-3 (DDP, NVLink)"
echo " train    : $n_train json  (nuPlan $n_nuplan + nuScenes $n_nusc, 比例 $(awk "BEGIN{printf \"%.1f\",$n_nuplan/$n_nusc}"):1)"
echo " val      : $n_val json   (nuScenes only)"
echo " global batch = 1 x ${ACCUM} accum x 4 GPU = ${GB}"
echo " LOG      : $LOG"
echo "=================================================================="

$PY tools/run_sft.py --config "$CONFIG" 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}

echo
echo "=================================================================="
echo " exit code : $STATUS"
echo " LOG       : $LOG"
echo " CKPT dir  : $(ls -dt "$REPO"/runs/sft/*/ 2>/dev/null | head -1)"
echo " 评测: GPU=0 CKPT=<best> bash logs/0810_nvidia/eval_nuscenes_l2_collision.sh (L2+Collision)"
echo "=================================================================="
exit "$STATUS"
