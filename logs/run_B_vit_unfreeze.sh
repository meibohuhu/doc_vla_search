#!/bin/bash
# Ablation B: ViT UNFROZEN + LLM full-param
# Pair with run_A_vit_frozen.sh -- identical except train_vision_backbone
# (+ vision_learning_rate 2e-6, since 2e-5 is tuned for the LLM and is too
#  large for the 669M pretrained vision tower).
#
#   Usage:  bash logs/run_B_vit_unfreeze.sh
#
set -u
CONFIG="training/qwen2.5-vl-3B-nuplan-nocot-sft-vit-unfreeze"
TAG="B_vit_unfreeze"

REPO=/home/mh2803/vla/doc_vla_search
source /home/mh2803/miniconda3/etc/profile.d/conda.sh
conda activate autovla_codeclean
cd "$REPO"

# --- 2 GPUs ---
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false

# --- Blackwell RTX PRO 6000 (no NVLink): NCCL P2P + SHM both fault with
#     "illegal memory access". BOTH must be disabled -- P2P alone is not enough. ---
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1

export NUPLAN_MAPS_ROOT=/scratch/mh2803/vla/nuplan/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export OPENSCENE_DATA_ROOT=/scratch/mh2803/vla/nuplan
export NAVSIM_DEVKIT_ROOT="$REPO/navsim"

STAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/${TAG}_${STAMP}.log"

echo "=================================================================="
echo " run    : $TAG  (ViT unfrozen, LLM full-param)"
echo " config : config/${CONFIG}.yaml"
echo " LOG    : $LOG"
echo "=================================================================="

python tools/run_sft.py --config "$CONFIG" 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}

echo
echo "=================================================================="
echo " exit code : $STATUS"
echo " LOG       : $LOG"
echo " CKPT dir  : $(ls -dt "$REPO"/runs/sft/*/ 2>/dev/null | head -1)"
echo "=================================================================="
exit "$STATUS"
