#!/bin/bash
# ============================================================================
# no-CoT SFT on 【nuScenes】(train 19030 / val 5569), ViT FROZEN + LLM full-param
# 本机(brev, 8xA100-SXM4-80G) 专用，4 卡 DDP（GPU 0-3）。
#
#   tmux new -s nusc
#   bash logs/0810_nvidia/run_nuscenes_nocot_sft_vitfrozen_4gpu.sh
#
# 与 nuPlan 版(logs/0722_nvidia/step3_run_vit_frozen_0727_100k_brev_gpu8.sh)的差异：
#   * 数据集：nuPlan navtrain(101k) -> 【nuScenes】(19k)，数据量约 1/5
#   * config -> qwen2.5-vl-3B-nuscenes-nocot-sft
#   * 数据是【绝对 camera 路径】的预处理 JSON，故 config 里 sensor_data_path=null
#   * codebook 沿用 nuPlan 的 agent_vocab.pkl（round-trip 实测 nuScenes 均值 0.078m < 0.5m）
#   * ⚠️ 评测【不走 PDMS】：nuScenes 用 L2 + Collision（tools/eval/nusc_eval.py），
#        需另备 UniAD 分割数据，训完不能直接用 nuPlan 的 eval 脚本。
#   * fp32_master=true 保留（否则 bf16 舍入毁训练，见 docs/0724/bf16_master_weights_bug.md）
# ============================================================================
set -u

REPO=/home/nvidia/workspace/doc_drive_search/other_repo/AutoVLA
PY=/data/autovla_data/envs/autovla/bin/python
CONFIG="training/qwen2.5-vl-3B-nuscenes-nocot-sft"
TAG="nuscenes_nocot_sft_vitfrozen"

cd "$REPO"

# --- preflight: nuScenes 训练/验证 JSON 必须存在 ---
TRAIN_DIR=/data/autovla_data/nuscenes/nuscenes_train
VAL_DIR=/data/autovla_data/nuscenes/nuscenes_val
count_json() { find "$1" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l; }
n_train=$(count_json "$TRAIN_DIR")
n_val=$(count_json "$VAL_DIR")
[ "$n_train" -eq 0 ] && { echo "ERROR: $TRAIN_DIR 为空 -- 先跑 nuScenes 预处理"; exit 1; }
[ "$n_val" -eq 0 ]   && { echo "ERROR: $VAL_DIR 为空"; exit 1; }
[ -e "$REPO/Qwen2.5-VL-3B-Instruct" ] || { echo "ERROR: 找不到 Qwen2.5-VL-3B-Instruct"; exit 1; }
# 抽一个 JSON 确认 camera 路径真实存在（nuScenes 是绝对路径）
SAMPLE=$(find "$TRAIN_DIR" -maxdepth 1 -name '*.json' | head -1)
$PY -c "
import json,os,sys
d=json.load(open('$SAMPLE')); p=d['front_camera_paths'][0]
assert os.path.isabs(p) and os.path.exists(p), f'camera 路径不存在: {p}'
assert d['dataset_name']=='nuscenes', f'dataset_name 不是 nuscenes: {d[\"dataset_name\"]}'
print('  preflight: camera 路径 OK, dataset_name=nuscenes')
" || { echo "ERROR: nuScenes 数据自检失败"; exit 1; }

# --- 4 GPUs ---
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
# A100-SXM4 有 NVLink，保持 NCCL 默认（不要禁 P2P/SHM）

# nuScenes 预处理走绝对路径，不依赖 OPENSCENE_DATA_ROOT；但 navsim import 需要 maps 环境变量存在
export NUPLAN_MAPS_ROOT=/data/autovla_data/nuplan/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export OPENSCENE_DATA_ROOT=/data/autovla_data/nuplan
export NAVSIM_DEVKIT_ROOT="$REPO/navsim"
export WANDB_PROJECT=autovla-nuscenes-sft

# config 里 accum=8（4卡 global batch = 1 x 8 x 4 = 32，对齐论文）。
echo "  4 GPUs (DDP)"
ACCUM=$($PY -c "import yaml;print(yaml.safe_load(open('config/${CONFIG}.yaml'))['training']['accumulate_grad_batches'])")
GB=$((1 * ACCUM * 4))
echo "  global batch = 1 x ${ACCUM} accum x 4 GPU = ${GB}"

STAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/0810_nvidia/${TAG}_${STAMP}.log"

echo "=================================================================="
echo " nuScenes no-CoT SFT  (ViT frozen, LLM full-param)"
echo " config   : config/${CONFIG}.yaml"
echo " GPUs     : 0-3 (DDP, NVLink)"
echo " train    : $n_train json  (nuScenes)"
echo " val      : $n_val json   (nuScenes)"
echo " LOG      : $LOG"
echo "=================================================================="

$PY tools/run_sft.py --config "$CONFIG" 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}

echo
echo "=================================================================="
echo " exit code : $STATUS"
echo " LOG       : $LOG"
echo " CKPT dir  : $(ls -dt "$REPO"/runs/sft/*/ 2>/dev/null | head -1)"
echo " ⚠️ 评测: nuScenes 用 tools/eval/nusc_eval.py (L2+Collision)，非 PDMS，需先备分割数据"
echo "=================================================================="
exit "$STATUS"
