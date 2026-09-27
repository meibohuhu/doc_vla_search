#!/bin/bash
# ============================================================================
# GRPO (RFT) 冒烟/试跑 —— CoT SFT ckpt 上做 PDMS-reward 强化微调。
#   3 卡(物理 GPU 4/5/6)，DDP，LoRA r=8，per_prompt_G=4。
#
#   sft_model_path : runs/sft/2026-09-02_05-47-39/epoch=3-loss=0.4446.ckpt (best CoT SFT, PDMS 79.89)
#   train          : navtrain50k_cot (5万) + navtrain_metric_cache (reward)
#   val            : navtest_cot + navtest_metric_cache
#
#   tmux new -s grpo
#   bash scripts/0901/run_grpo_cot_3gpu.sh
#   SMOKE=1 bash scripts/0901/run_grpo_cot_3gpu.sh   # 只验证能起：跑几步看 reward/advantage 就停
# ============================================================================
set -uo pipefail

REPO=/home/nvidia/workspace/doc_drive_search/other_repo/AutoVLA
PY=/data/autovla_data/envs/autovla/bin/python
CONFIG="training/0916_nvidia/qwen2.5-vl-3B-nuplan-grpo-cot-brev-3gpu"
cd "$REPO"

# --- 6 卡：物理 GPU 1-6（GPU0 不用），进程内映射为 0-5 ---
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6
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
fail=0
[ -f "$SFT" ] || { echo "❌ sft_model_path 不存在: $SFT"; fail=1; }
for p in navtrain50k_cot navtrain_metric_cache navtest_cot; do
  [ -e "/data/autovla_data/nuplan/$p" ] || { echo "❌ 缺 /data/autovla_data/nuplan/$p"; fail=1; }
done
[ -e /data/autovla_data/nuplan/navtest_metric_cache ] || { echo "❌ 缺 navtest_metric_cache"; fail=1; }
[ "$fail" -eq 0 ] || { echo "前置检查未通过"; exit 1; }

STAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/0901_nvidia/grpo_cot_3gpu_${STAMP}.log"
mkdir -p "$REPO/logs/0901_nvidia"

echo "=================================================================="
echo " GRPO (RFT) on CoT SFT   —— 3 卡 (物理 4/5/6), DDP, LoRA"
echo " config      : config/${CONFIG}.yaml"
echo " sft_model   : $SFT"
echo " train       : navtrain50k_cot (5万) + navtrain_metric_cache"
echo " reward      : PDMS (rl_pdm_score)"
echo " ${SMOKE:+SMOKE 模式: 起来看 reward/advantage 非零即可}"
echo " LOG         : $LOG"
echo "=================================================================="

$PY tools/run_rft.py --config "$CONFIG" 2>&1 | tee "$LOG"
echo "exit=${PIPESTATUS[0]}  LOG=$LOG"
