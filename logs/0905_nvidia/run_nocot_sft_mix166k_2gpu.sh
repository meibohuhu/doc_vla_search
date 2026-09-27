#!/bin/bash
# no-CoT SFT on 【mix 154k】= navtrain_nocot(101,288) + 166k-only 分布对齐采样(53,172)，ViT FROZEN + LLM 全参。
#   分布对齐 navtrain(STOP~19%/ACCEL~27%)，机动帧比 navtrain 单用更多。
#   val = navtrain_nocot_val(2,000)。no-CoT：cot_output 被忽略。
#   数据分析见 docs/0901/166k_vs_103k_distribution.md。
#
#   tmux new -s mixnocot
#   bash logs/0905_nvidia/run_nocot_sft_mix166k.sh
#   GPUS=0,1,2,3 bash logs/0905_nvidia/run_nocot_sft_mix166k.sh   # 换卡
set -u

REPO=/home/nvidia/workspace/doc_drive_search/other_repo/AutoVLA
PY=/data/autovla_data/envs/autovla/bin/python
CONFIG="training/qwen2.5-vl-3B-nuplan-nocot-sft-mix166k-brev-2gpu"
TAG="mix166k_nocot_sft_2gpu_0905"
GPUS="${GPUS:-4,5}"      # 2 卡 x accum 16 = global batch 32

cd "$REPO"

# --- preflight 1: 两个 train 目录 + val 都在 ---
T1=/data/autovla_data/nuplan/navtrain_nocot
T2=/data/autovla_data/nuplan/trainval_mix166k_add
VAL=/data/autovla_data/nuplan/navtrain_nocot_val
count_json() { find "$1" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l; }
n1=$(count_json "$T1"); n2=$(count_json "$T2"); nv=$(count_json "$VAL")
[ "$n1" -eq 0 ] && { echo "ERROR: $T1 为空"; exit 1; }
[ "$n2" -eq 0 ] && { echo "ERROR: $T2 为空 -- 先建 trainval_mix166k_add"; exit 1; }
[ "$nv" -eq 0 ] && { echo "ERROR: $VAL 为空"; exit 1; }
[ -e "$REPO/Qwen2.5-VL-3B-Instruct" ] || { echo "ERROR: 找不到 Qwen2.5-VL-3B-Instruct"; exit 1; }
n_train=$((n1 + n2))

# --- preflight 2: 两个 train 目录各抽一条,camera 图存在（no-CoT 不查 cot_output）---
for D in "$T1" "$T2"; do
  S=$(find "$D" -maxdepth 1 -name '*.json' | head -1)
  $PY -c "
import json,os
d=json.load(open('$S'))
for c in ('front_camera_paths','front_left_camera_paths','front_right_camera_paths'):
    for p in d[c][:4]: assert os.path.exists(p), f'图不在盘: {p}'
print('  preflight OK: $D  camera 都在')
" || { echo "ERROR: $D 自检失败"; exit 1; }
done

# --- preflight 3: global batch = 32 ---
n_gpu=$(awk -F, '{print NF}' <<< "$GPUS")
accum=$(grep -E '^\s*accumulate_grad_batches:' "config/${CONFIG}.yaml" | awk '{print $2}')
bs=$(grep -E '^\s*batch_size:' "config/${CONFIG}.yaml" | head -1 | awk '{print $2}')
gb=$((bs * accum * n_gpu))
[ "$gb" -ne 32 ] && { echo "ERROR: global batch = $bs x $accum x $n_gpu = $gb ≠ 32。改 GPUS 要同步改 config accum。"; exit 1; }

export CUDA_VISIBLE_DEVICES="$GPUS"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PRINT_SFT_DEBUG="${PRINT_SFT_DEBUG:-0}"
export NUPLAN_MAPS_ROOT=/data/autovla_data/nuplan/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export OPENSCENE_DATA_ROOT=/data/autovla_data/nuplan
export NAVSIM_DEVKIT_ROOT="$REPO/navsim"
export WANDB_PROJECT=autovla-nocot-sft-mix166k

STAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/0905_nvidia/${TAG}_${STAMP}.log"

echo "=================================================================="
echo " no-CoT SFT mix166k : $TAG   (ViT frozen, LLM full-param)"
echo " config  : config/${CONFIG}.yaml"
echo " GPUs    : $GPUS  ($n_gpu 张, DDP)"
echo " train   : $n_train json  (navtrain_nocot $n1 + mix_add $n2)"
echo " val     : $nv json   (navtrain_nocot_val)"
echo " global batch = $bs x $accum accum x $n_gpu GPU = $gb"
echo " LOG     : $LOG"
echo "=================================================================="

$PY tools/run_sft.py --config "$CONFIG" 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
echo; echo " exit=$STATUS  LOG=$LOG  CKPT=$(ls -dt "$REPO"/runs/sft/*/ 2>/dev/null | head -1)"
exit "$STATUS"
