#!/bin/bash
# STEP 1/3 -- preprocess nuPlan navtrain into no-CoT JSON (~103,288 scenes)
#
#   bash logs/0721/step1_preprocess_navtrain.sh
#
# This is the long pole. Run it in tmux/screen so it survives a disconnect.
# --fast is essential: it skips ~3.3M JPEG decodes + base64 round-trips whose
# results the no-CoT path throws away (see dataset_utils/preprocessing/0721/fast_nocot_patch.py).
set -u
REPO=/home/mh2803/vla/doc_vla_search
CONFIG=dataset/0721/nuplan-navtrain
OUT=/scratch/mh2803/vla/nuplan/navtrain_nocot
WORKERS=32                     # 72 logical cores on this box

source /home/mh2803/miniconda3/etc/profile.d/conda.sh
conda activate autovla_codeclean
cd "$REPO"

export TOKENIZERS_PARALLELISM=false
export NUPLAN_MAPS_ROOT=/scratch/mh2803/vla/nuplan/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export OPENSCENE_DATA_ROOT=/scratch/mh2803/vla/nuplan
export NAVSIM_DEVKIT_ROOT="$REPO/navsim"

STAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/0721/step1_preprocess_navtrain_${STAMP}.log"
mkdir -p "$OUT"

echo "=================================================================="
echo " STEP 1/3 : preprocess navtrain -> no-CoT JSON"
echo " config   : config/${CONFIG}.yaml"
echo " output   : $OUT"
echo " workers  : $WORKERS   (--fast enabled)"
echo " LOG      : $LOG"
echo "=================================================================="

python tools/preprocessing/nocot_sample_generation.py \
    --config "$CONFIG" \
    --output_dir "$OUT" \
    --num_workers "$WORKERS" \
    --fast 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}

echo
echo "=================================================================="
echo " exit code   : $STATUS"
echo " JSON count  : $(ls "$OUT"/*.json 2>/dev/null | wc -l)   (expect ~103,288)"
echo " LOG         : $LOG"
echo " next        : bash logs/0721/step2_make_val_split.sh"
echo "=================================================================="
exit "$STATUS"
