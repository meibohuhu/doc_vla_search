#!/bin/bash
# STEP 3/3 -- no-CoT SFT on nuPlan navtrain (~101k), ViT FROZEN + LLM full-param
#
#   bash logs/0721/step3_run_vit_frozen.sh
#
# Expect ~2 days for 5 epochs on 2 GPUs (measured 0.71 it/s @ batch=2).
# Run it in tmux/screen so it survives a disconnect.
set -u
CONFIG="training/qwen2.5-vl-3B-nuplan-nocot-sft-navtrain"
TAG="navtrain_vit_frozen_0722_100k"

REPO=/home/mh2803/vla/doc_vla_search
source /home/mh2803/miniconda3/etc/profile.d/conda.sh
conda activate autovla_codeclean
cd "$REPO"

# --- preflight: the training data must actually exist ---
TRAIN_DIR=/scratch/mh2803/vla/nuplan/navtrain_nocot
VAL_DIR=/scratch/mh2803/vla/nuplan/navtrain_nocot_val
# count with find, not `ls dir/*.json`: ~103k files blow past ARG_MAX and the
# resulting "Argument list too long" would silently read as a count of 0.
count_json() { find "$1" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l; }
n_train=$(count_json "$TRAIN_DIR")
n_val=$(count_json "$VAL_DIR")
if [ "$n_train" -eq 0 ]; then
  echo "ERROR: no JSON in $TRAIN_DIR -- run step1_preprocess_navtrain.sh first."; exit 1
fi
if [ "$n_val" -eq 0 ]; then
  echo "ERROR: no JSON in $VAL_DIR -- run step2_make_val_split.sh first."; exit 1
fi

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
LOG="$REPO/logs/0721/${TAG}_${STAMP}.log"

echo "=================================================================="
echo " STEP 3/3 : $TAG  (ViT frozen, LLM full-param)"
echo " config   : config/${CONFIG}.yaml"
echo " train    : $n_train json"
echo " val      : $n_val json"
echo " global batch = 2 x 8 accum x 2 GPU = 32"
echo " LOG      : $LOG"
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
