#!/bin/bash
# STEP 3/3 -- no-CoT SFT on nuPlan navtrain (~101k), ViT FROZEN + LLM full-param
# 本机(brev, 8xA100-SXM4-80G) 专用，GPU 4/5/6/7。
#
#   tmux new -s sft            # 断线也能活
#   bash logs/0721/step3_run_vit_frozen_0722_100k_brev_gpu4567.sh
#
# 与 mh2803 参考脚本(step3_run_vit_frozen_0722_100k.sh)的差异：
#   * REPO / conda / 数据路径 -> 本机 /data
#   * GPU 0,1 -> 4,5,6,7
#   * ❌ 不禁用 NCCL P2P/SHM：那是给无 NVLink 的 Blackwell RTX PRO 6000 用的；
#        本机 A100-SXM4 全 NVLink(NV12)，禁用反而绕开 NVLink 走慢路径
#   * config: accum 8 -> 4（4 卡而非 2 卡，保持 global batch=32）
set -u

REPO=/home/nvidia/workspace/doc_drive_search/other_repo/AutoVLA
PY=/data/autovla_data/envs/autovla/bin/python
CONFIG="training/qwen2.5-vl-3B-nuplan-nocot-sft-navtrain-brev"
TAG="navtrain_vit_frozen_0722_100k_gpu4567"

cd "$REPO"

# --- preflight: 训练/验证数据必须真实存在 ---
TRAIN_DIR=/data/autovla_data/nuplan/navtrain_nocot
VAL_DIR=/data/autovla_data/nuplan/navtrain_nocot_val
# 用 find 数，不用 `ls dir/*.json`：~101k 文件会撑爆 ARG_MAX，
# "Argument list too long" 会被静默当成 0。
count_json() { find "$1" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l; }
n_train=$(count_json "$TRAIN_DIR")
n_val=$(count_json "$VAL_DIR")
[ "$n_train" -eq 0 ] && { echo "ERROR: $TRAIN_DIR 为空 -- 先跑预处理"; exit 1; }
[ "$n_val" -eq 0 ]   && { echo "ERROR: $VAL_DIR 为空 -- 先切 val 集"; exit 1; }
[ -e "$REPO/Qwen2.5-VL-3B-Instruct" ] || { echo "ERROR: 找不到 Qwen2.5-VL-3B-Instruct"; exit 1; }

# --- 4 GPUs ---
export CUDA_VISIBLE_DEVICES=4,5,6,7
export TOKENIZERS_PARALLELISM=false

# --- A100-SXM4 有 NVLink，保持 NCCL 默认（不要禁 P2P/SHM）---

export NUPLAN_MAPS_ROOT=/data/autovla_data/nuplan/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export OPENSCENE_DATA_ROOT=/data/autovla_data/nuplan
export NAVSIM_DEVKIT_ROOT="$REPO/navsim"
export WANDB_PROJECT=autovla-nocot-sft

STAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/0721/${TAG}_${STAMP}.log"

echo "=================================================================="
echo " STEP 3/3 : $TAG  (ViT frozen, LLM full-param)"
echo " config   : config/${CONFIG}.yaml"
echo " GPUs     : 4,5,6,7 (DDP, NVLink)"
echo " train    : $n_train json"
echo " val      : $n_val json"
echo " global batch = 2 x 4 accum x 4 GPU = 32"
echo " LOG      : $LOG"
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
