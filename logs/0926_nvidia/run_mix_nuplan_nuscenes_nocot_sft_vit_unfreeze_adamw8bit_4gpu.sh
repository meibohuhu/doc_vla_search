#!/bin/bash
# no-CoT SFT 混训【nuPlan navtrain (101k) + nuScenes train (19k)】，【ViT 解冻(全参) + optimizer=adamw8bit】，4 卡 DDP。
#   本机(brev, A100-SXM4-80G)；照 logs/0909_nvidia/run_nocot_sft_vit_unfreeze_adamw8bit_4gpu.sh
#   (navtest PDMS 84.22 那版) 改的，只多了 nuScenes 数据。
#
#   为什么 adamw8bit:解冻 ViT 后 3.8B 全参,单卡峰值 ~84.6G > 80G。8-bit AdamW 把优化器
#   状态 28G->7G(省 ~21G),配合 expandable_segments 才塞得下 80G 卡。
#
#   tmux new -s mixnusc
#   bash logs/0926_nvidia/run_mix_nuplan_nuscenes_nocot_sft_vit_unfreeze_adamw8bit_4gpu.sh
#   GPUS=4,5,6,7 bash logs/0926_nvidia/run_mix_nuplan_nuscenes_nocot_sft_vit_unfreeze_adamw8bit_4gpu.sh   # 换卡
set -u

REPO=/home/nvidia/workspace/other_repo/AutoVLA
PY=/data/autovla_data/envs/autovla/bin/python
CONFIG="training/0926_nvidia/qwen2.5-vl-3B-mix-nuplan103k-nuscenes-nocot-sft-vit-unfreeze-adamw8bit-brev"
TAG="mix_nuplan103k_nuscenes_nocot_vit_unfreeze_adamw8bit_4gpu"
GPUS="${GPUS:-0,1,2,3}"      # 4 卡 x accum 8 = global batch 32

cd "$REPO"

# --- preflight 0: adamw8bit 依赖 bitsandbytes,没装会在 configure_optimizers 崩 ---
$PY -c "import bitsandbytes" 2>/dev/null || {
  echo "❌ bitsandbytes 未安装,adamw8bit 无法运行。先装:"
  echo "   $PY -m pip install bitsandbytes"
  exit 1
}

# --- preflight 1: 数据(nuPlan 101k + nuScenes 19k, val 两边) ---
NUPLAN_TRAIN=/data/autovla_data/nuplan/navtrain_nocot
NUPLAN_VAL=/data/autovla_data/nuplan/navtrain_nocot_val
NUSC_TRAIN=/data/autovla_data/nuscenes/nuscenes_train
NUSC_VAL=/data/autovla_data/nuscenes/nuscenes_val
count_json() { find "$1" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l; }
for d in "$NUPLAN_TRAIN" "$NUPLAN_VAL" "$NUSC_TRAIN" "$NUSC_VAL"; do
  [ "$(count_json "$d")" -eq 0 ] && { echo "ERROR: $d 为空"; exit 1; }
done
n_nuplan=$(count_json "$NUPLAN_TRAIN");  n_nusc=$(count_json "$NUSC_TRAIN")
n_nuplan_val=$(count_json "$NUPLAN_VAL"); n_nusc_val=$(count_json "$NUSC_VAL")
[ -e "$REPO/Qwen2.5-VL-3B-Instruct" ] || { echo "ERROR: 找不到 Qwen2.5-VL-3B-Instruct"; exit 1; }

# --- preflight 2: 两个数据集的 camera 图帧级完整 ---
for d in "$NUPLAN_TRAIN" "$NUSC_TRAIN"; do
  S=$(find "$d" -maxdepth 1 -name '*.json' | head -1)
  $PY -c "
import json,os
d=json.load(open('$S'))
for c in ('front_camera_paths','front_left_camera_paths','front_right_camera_paths'):
    for p in d[c][:4]: assert os.path.exists(p), f'图不在盘: {p}'
print('  preflight: camera 图帧级完整 OK  ($d)')
" || { echo "ERROR: camera 图缺失 ($d)"; exit 1; }
done

# --- preflight 3: config 确认 vit 解冻 + adamw8bit + no-CoT + 两个 train 路径 + global batch=32 ---
$PY -c "
import yaml; c=yaml.safe_load(open('config/${CONFIG}.yaml'))
assert c['model']['train_vision_backbone'] is True, 'ViT 未解冻'
assert c['model']['use_cot'] is False, 'use_cot 不是 False'
assert str(c['training'].get('optimizer','')).lower()=='adamw8bit', 'optimizer 不是 adamw8bit'
assert len(c['data']['train']['json_dataset_path'])==2, 'train 不是 nuPlan+nuScenes 两个路径'
print('  config OK: ViT 解冻 + adamw8bit + no-CoT + nuPlan/nuScenes 混训')
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
# A100-SXM4 全 NVLink,保持 NCCL 默认
export NUPLAN_MAPS_ROOT=/data/autovla_data/nuplan/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export OPENSCENE_DATA_ROOT=/data/autovla_data/nuplan
export NAVSIM_DEVKIT_ROOT="$REPO/navsim"
export WANDB_PROJECT=autovla-mix-nuplan-nuscenes-sft

STAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/0926_nvidia/${TAG}_${STAMP}.log"

echo "=================================================================="
echo " no-CoT SFT : $TAG   (★ViT 解冻 + adamw8bit, nuPlan 103k + nuScenes 混训)"
echo " config  : config/${CONFIG}.yaml"
echo " GPUs    : $GPUS  ($n_gpu 张, DDP, NVLink)"
echo " train   : $n_nuplan nuPlan + $n_nusc nuScenes = $((n_nuplan + n_nusc)) json"
echo " val     : $n_nuplan_val nuPlan + $n_nusc_val nuScenes = $((n_nuplan_val + n_nusc_val)) json"
echo " global batch = $bs x $accum accum x $n_gpu GPU = $gb"
echo " ⚠️ 解冻 ViT 显存紧(~84G→靠 adamw8bit+expandable 压到 80G 内),OOM 就减到 3 卡或看碎片"
echo " LOG     : $LOG"
echo "=================================================================="

$PY tools/run_sft.py --config "$CONFIG" 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
echo; echo " exit=$STATUS  LOG=$LOG  CKPT=$(ls -dt "$REPO"/runs/sft/*/ 2>/dev/null | head -1)"
exit "$STATUS"
