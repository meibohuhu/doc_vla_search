#!/bin/bash
# no-CoT SFT on nuPlan navtrain (~103k)，【ViT 解冻(全参) + optimizer=adamw8bit】，4 卡 DDP。
#   本机(brev, A100-SXM4-80G) 版；参考 logs/0725_gcis/step3_run_vit_frozen_0723_100k_fix_vit_adamw8bit.sh
#   (那份是 gcis:scratch 路径/单卡/Blackwell 禁 NCCL)。
#
#   为什么 adamw8bit:解冻 ViT 后 3.8B 全参,单卡峰值 ~84.6G > 80G。8-bit AdamW 把优化器
#   状态 28G->7G(省 ~21G),配合 expandable_segments 才塞得下 80G 卡。
#
#   tmux new -s vitunfz
#   bash logs/0909_nvidia/run_nocot_sft_vit_unfreeze_adamw8bit_4gpu.sh
#   GPUS=0,4,5,6 bash logs/0909_nvidia/run_nocot_sft_vit_unfreeze_adamw8bit_4gpu.sh   # 换卡
set -u

REPO=/home/nvidia/workspace/doc_drive_search/other_repo/AutoVLA
PY=/data/autovla_data/envs/autovla/bin/python
CONFIG="training/0916_nvidia/qwen2.5-vl-3B-nuplan-nocot-sft-vit-unfreeze-adamw8bit-brev"
TAG="navtrain_vit_unfreeze_adamw8bit_4gpu"
GPUS="${GPUS:-0,1,2,3}"      # 4 卡 x accum 8 = global batch 32

cd "$REPO"

# --- preflight 0: adamw8bit 依赖 bitsandbytes,没装会在 configure_optimizers 崩 ---
$PY -c "import bitsandbytes" 2>/dev/null || {
  echo "❌ bitsandbytes 未安装,adamw8bit 无法运行。先装:"
  echo "   $PY -m pip install bitsandbytes"
  exit 1
}

# --- preflight 1: 数据(103k navtrain no-CoT)---
TRAIN_DIR=/data/autovla_data/nuplan/navtrain_nocot
VAL_DIR=/data/autovla_data/nuplan/navtrain_nocot_val
count_json() { find "$1" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l; }
n_train=$(count_json "$TRAIN_DIR")
n_val=$(count_json "$VAL_DIR")
[ "$n_train" -eq 0 ] && { echo "ERROR: $TRAIN_DIR 为空"; exit 1; }
[ "$n_val" -eq 0 ]   && { echo "ERROR: $VAL_DIR 为空"; exit 1; }
[ -e "$REPO/Qwen2.5-VL-3B-Instruct" ] || { echo "ERROR: 找不到 Qwen2.5-VL-3B-Instruct"; exit 1; }

# --- preflight 2: camera 图帧级完整 ---
S=$(find "$TRAIN_DIR" -maxdepth 1 -name '*.json' | head -1)
$PY -c "
import json,os
d=json.load(open('$S'))
for c in ('front_camera_paths','front_left_camera_paths','front_right_camera_paths'):
    for p in d[c][:4]: assert os.path.exists(p), f'图不在盘: {p}'
print('  preflight: camera 图帧级完整 OK')
" || { echo "ERROR: camera 图缺失"; exit 1; }

# --- preflight 3: config 确认 vit 解冻 + adamw8bit + global batch=32 ---
$PY -c "
import yaml; c=yaml.safe_load(open('config/${CONFIG}.yaml'))
assert c['model']['train_vision_backbone'] is True, 'ViT 未解冻'
assert str(c['training'].get('optimizer','')).lower()=='adamw8bit', 'optimizer 不是 adamw8bit'
print('  config OK: ViT 解冻 + adamw8bit')
" || exit 1
n_gpu=$(awk -F, '{print NF}' <<< "$GPUS")
accum=$(grep -E '^\s*accumulate_grad_batches:' "config/${CONFIG}.yaml" | awk '{print $2}')
bs=$(grep -E '^\s*batch_size:' "config/${CONFIG}.yaml" | head -1 | awk '{print $2}')
gb=$((bs * accum * n_gpu))
[ "$gb" -ne 32 ] && { echo "ERROR: global batch = $bs x $accum x $n_gpu = $gb ≠ 32。改 GPUS 要同步改 accum。"; exit 1; }

export CUDA_VISIBLE_DEVICES="$GPUS"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# 解冻 ViT 显存吃紧:变长视觉 token 造成碎片,expandable_segments 回收,不改数值不影响速度
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# A100-SXM4 全 NVLink,保持 NCCL 默认(不要禁 P2P/SHM —— 那是 gcis Blackwell 无 NVLink 才需要的)
export NUPLAN_MAPS_ROOT=/data/autovla_data/nuplan/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export OPENSCENE_DATA_ROOT=/data/autovla_data/nuplan
export NAVSIM_DEVKIT_ROOT="$REPO/navsim"
export WANDB_PROJECT=autovla-nocot-sft-vitunfreeze

STAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/0909_nvidia/${TAG}_${STAMP}.log"

echo "=================================================================="
echo " no-CoT SFT : $TAG   (★ViT 解冻 + adamw8bit, 103k navtrain)"
echo " config  : config/${CONFIG}.yaml"
echo " GPUs    : $GPUS  ($n_gpu 张, DDP, NVLink)"
echo " train   : $n_train json (navtrain_nocot)"
echo " val     : $n_val json"
echo " global batch = $bs x $accum accum x $n_gpu GPU = $gb"
echo " ⚠️ 解冻 ViT 显存紧(~84G→靠 adamw8bit+expandable 压到 80G 内),OOM 就减到 3 卡或看碎片"
echo " LOG     : $LOG"
echo "=================================================================="

$PY tools/run_sft.py --config "$CONFIG" 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
echo; echo " exit=$STATUS  LOG=$LOG  CKPT=$(ls -dt "$REPO"/runs/sft/*/ 2>/dev/null | head -1)"
exit "$STATUS"
