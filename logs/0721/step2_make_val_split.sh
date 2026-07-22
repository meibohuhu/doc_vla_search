#!/bin/bash
# STEP 2/3 -- carve a validation split out of navtrain
#
#   bash logs/0721/step2_make_val_split.sh [N_VAL]      (default 2000)
#
# Why not use navtest as val: navtest is the final PDMS benchmark. Selecting
# checkpoints on it would leak the test set, so val comes out of navtrain.
set -u
TRAIN=/scratch/mh2803/vla/nuplan/navtrain_nocot
VAL=/scratch/mh2803/vla/nuplan/navtrain_nocot_val
N_VAL=${1:-2000}

# NOTE: always count/iterate with find, never `ls dir/*.json` -- navtrain has
# ~103k files and glob expansion blows past ARG_MAX ("Argument list too long"),
# which silently reads as a count of 0.
count_json() { find "$1" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l; }

have=$(count_json "$TRAIN")
if [ "$have" -eq 0 ]; then
  echo "ERROR: $TRAIN is empty -- run step1_preprocess_navtrain.sh first."
  exit 1
fi

existing=$(count_json "$VAL")
if [ "$existing" -gt 0 ]; then
  echo "val split already exists: $existing json in $VAL -- nothing to do."
  echo "  (delete it first if you want to re-draw the split)"
  exit 0
fi

mkdir -p "$VAL"
# deterministic: sorted-name tail, so the split is reproducible across machines
find "$TRAIN" -maxdepth 1 -name '*.json' | sort | tail -n "$N_VAL" \
  | xargs -r mv -t "$VAL"

echo "=================================================================="
echo " train : $(count_json "$TRAIN") json   ($TRAIN)"
echo " val   : $(count_json "$VAL") json   ($VAL)"
echo " next  : bash logs/0721/step3_run_vit_frozen.sh"
echo "=================================================================="
