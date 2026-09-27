#!/bin/bash
# ============================================================================
# GRPO (RFT) —— 3 卡(物理 GPU 1/2/3), DDP, LoRA r=8, per_prompt_G=4。
#   ★ 小 lr 版:learning_rate 4e-6(上次 3e-5 太大,PDMS 从 80.45 单调跌到 78.53,
#     退步全在 NC/TTC —— policy 往不安全方向漂)。lr 降到 1/7.5,配 kl_beta=0.1。
#
#   sft_model_path : runs/sft/2026-09-02_05-47-39/epoch=4-loss=0.4436.ckpt (best CoT SFT, PDMS 80.45)
#   train          : navtrain50k_cot (5万) + navtrain_metric_cache (reward)
#   config         : training/qwen2.5-vl-3B-nuplan-grpo-cot-brev-lowlr-3gpu (lr=4e-6, devices=[0,1,2])
#
#   tmux new -s grpo_lowlr
#   bash scripts/0901/run_grpo_cot_lowlr_3gpu.sh
#
# ⚠️ 盯 wandb 的 r_pdm:几百步内别再单调跌就算 lr 对了。根子问题(随机 5w → 简单帧
#    advantage≈0 → 信号稀噪)仍在,小 lr 主要是防漂,未必能真把 PDMS 推上去。
# ============================================================================
set -uo pipefail

REPO=/home/nvidia/workspace/doc_drive_search/other_repo/AutoVLA
PY=/data/autovla_data/envs/autovla/bin/python
CONFIG="training/qwen2.5-vl-3B-nuplan-grpo-cot-brev-lowlr-3gpu"
cd "$REPO"

# --- 3 卡：物理 GPU 1/2/3（0/4/5/6 常被 SFT/评测占），进程内映射为 0/1/2 ---
export CUDA_VISIBLE_DEVICES=1,2,3
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NUPLAN_MAPS_ROOT=/data/autovla_data/nuplan/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export OPENSCENE_DATA_ROOT=/data/autovla_data/nuplan
export NAVSIM_DEVKIT_ROOT="$REPO/navsim"
export PYTHONPATH="$REPO/navsim:${PYTHONPATH:-}"
export WANDB_PROJECT=autovla-grpo-cot

# --- preflight ---
SFT=$($PY -c "import yaml;print(yaml.safe_load(open('config/${CONFIG}.yaml'))['model']['sft_model_path'])")
LR=$($PY -c "import yaml;print(yaml.safe_load(open('config/${CONFIG}.yaml'))['training']['learning_rate'])")
NDEV=$($PY -c "import yaml;print(len(yaml.safe_load(open('config/${CONFIG}.yaml'))['training']['devices']))")
fail=0
[ -f "$SFT" ] || { echo "❌ sft_model_path 不存在: $SFT"; fail=1; }
[ "$NDEV" -eq 3 ] || { echo "❌ config devices 数=$NDEV，应为 3"; fail=1; }
for p in navtrain50k_cot navtrain_metric_cache navtest_cot; do
  [ -e "/data/autovla_data/nuplan/$p" ] || { echo "❌ 缺 /data/autovla_data/nuplan/$p"; fail=1; }
done
[ -e /data/autovla_data/nuplan/navtest_metric_cache ] || { echo "❌ 缺 navtest_metric_cache"; fail=1; }
[ "$fail" -eq 0 ] || { echo "前置检查未通过"; exit 1; }

STAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/0901_nvidia/grpo_cot_lowlr_3gpu_${STAMP}.log"
mkdir -p "$REPO/logs/0901_nvidia"

echo "=================================================================="
echo " GRPO (RFT) 小 lr 版  —— 3 卡 (物理 1/2/3), DDP, LoRA"
echo " config    : config/${CONFIG}.yaml"
echo " lr        : $LR   (上次 3e-5 太大)"
echo " sft_model : $SFT"
echo " kl_beta   : 0.1   per_prompt_G: 4"
echo " LOG       : $LOG"
echo "=================================================================="

$PY tools/run_rft.py --config "$CONFIG" 2>&1 | tee "$LOG"
echo "exit=${PIPESTATUS[0]}  LOG=$LOG"
