#!/bin/bash
# no-CoT SFT on 全量 nuPlan trainval【166k】，ViT FROZEN + LLM 全参。
#   train = trainval_cot_166k_clean (165,786，已剔除 navtrain_cot_val 泄漏)
#   val   = navtrain_cot_val (2,000，与 103k 同一个 val，可直接比 val_loss)
#   no-CoT：直接出 action，不带 reasoning（cot_output 字段被忽略）。
#
#   tmux new -s nocot166k        # 断线也能活
#   bash logs/0902/run_nocot_sft_166k_0903.sh
#   GPUS=0,1,2,3 bash logs/0902/run_nocot_sft_166k_0903.sh      # 换卡
#
# 和 CoT 版(logs/0902/run_cot_sft_166k_0903_bf16.sh)的差别只有 config 的 use_cot。
# 数据构建/校验见 docs/0901/*、logs/0901_nvidia/。
set -u

REPO=/home/nvidia/workspace/doc_drive_search/other_repo/AutoVLA
PY=/data/autovla_data/envs/autovla/bin/python
CONFIG="training/qwen2.5-vl-3B-nuplan-nocot-sft-trainval166k-brev"
TAG="trainval166k_nocot_sft_0903"

# global batch 必须 = 32：4 卡 x accum 8 = 32。默认用【0,4,5,6】(1/2/3 常被评测/GRPO 占)。
GPUS="${GPUS:-0,1,2,3}"

cd "$REPO"

# --- preflight 1: 数据 ---
TRAIN_DIR=/data/autovla_data/nuplan/trainval_cot_166k_clean
VAL_DIR=/data/autovla_data/nuplan/navtrain_cot_val
# 用 find 数,不用 `ls dir/*.json`:~166k 文件会撑爆 ARG_MAX,
# "Argument list too long" 会被静默当成 0。
count_json() { find "$1" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l; }
n_train=$(count_json "$TRAIN_DIR")
n_val=$(count_json "$VAL_DIR")
[ "$n_train" -eq 0 ] && { echo "ERROR: $TRAIN_DIR 为空 -- 先建 trainval_cot_166k_clean"; exit 1; }
[ "$n_val" -eq 0 ]   && { echo "ERROR: $VAL_DIR 为空"; exit 1; }
[ -e "$REPO/Qwen2.5-VL-3B-Instruct" ] || { echo "ERROR: 找不到 Qwen2.5-VL-3B-Instruct"; exit 1; }

# --- preflight 2: 抽一条确认 camera 图真实存在（sensor 帧级完整）---
#   no-CoT 不需要 cot_output，故不查它；只查图在不在(上次 sensor 残缺栽过 9%)。
S=$(find "$TRAIN_DIR" -maxdepth 1 -name '*.json' | head -1)
$PY -c "
import json,os
d=json.load(open('$S'));
for c in ('front_camera_paths','front_left_camera_paths','front_right_camera_paths'):
    for p in d[c][:4]:
        assert os.path.exists(p), f'camera 图不在盘: {p}'
print('  preflight: camera 图帧级完整 OK')
" || { echo "ERROR: camera 图缺失 -- 检查 sensor_blobs/trainval"; exit 1; }

# --- preflight 3: global batch 必须 = 32，和 103k / CoT 基线一致 ---
n_gpu=$(awk -F, '{print NF}' <<< "$GPUS")
accum=$(grep -E '^\s*accumulate_grad_batches:' "config/${CONFIG}.yaml" | awk '{print $2}')
bs=$(grep -E '^\s*batch_size:' "config/${CONFIG}.yaml" | head -1 | awk '{print $2}')
gb=$((bs * accum * n_gpu))
if [ "$gb" -ne 32 ]; then
  echo "ERROR: global batch = ${bs} x ${accum} accum x ${n_gpu} GPU = ${gb}，应为 32。"
  echo "       改 GPUS 就必须同步改 config 里的 accumulate_grad_batches。"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPUS"
export TOKENIZERS_PARALLELISM=false
# 🔴 不加这个,终端/日志会看起来"卡住不动"(进度条用 \r 刷新,行缓冲不 flush)。
#    判断死活看 nvidia-smi / ps / wandb,别看这个日志。
export PYTHONUNBUFFERED=1
export PRINT_SFT_DEBUG="${PRINT_SFT_DEBUG:-0}"
# A100-SXM4 全 NVLink,保持 NCCL 默认
export NUPLAN_MAPS_ROOT=/data/autovla_data/nuplan/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export OPENSCENE_DATA_ROOT=/data/autovla_data/nuplan
export NAVSIM_DEVKIT_ROOT="$REPO/navsim"
export WANDB_PROJECT=autovla-nocot-sft-166k

STAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/0902/${TAG}_${STAMP}.log"

echo "=================================================================="
echo " no-CoT SFT : $TAG   (ViT frozen, LLM full-param, 166k)"
echo " config  : config/${CONFIG}.yaml"
echo " GPUs    : $GPUS  ($n_gpu 张, DDP, NVLink)"
echo " train   : $n_train json  (trainval_cot_166k_clean)"
echo " val     : $n_val json   (navtrain_cot_val，与 103k 同口径)"
echo " global batch = ${bs} x ${accum} accum x ${n_gpu} GPU = ${gb}"
echo " LOG     : $LOG"
echo "=================================================================="

$PY tools/run_sft.py --config "$CONFIG" 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}

echo
echo "=================================================================="
echo " exit code : $STATUS"
echo " LOG       : $LOG"
echo " CKPT dir  : $(ls -dt "$REPO"/runs/sft/*/ 2>/dev/null | head -1)"
echo "=================================================================="
exit "$STATUS"
