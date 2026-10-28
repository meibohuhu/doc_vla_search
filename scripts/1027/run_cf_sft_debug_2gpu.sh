#!/bin/bash
# 反事实指令 SFT —— 本机（gcis，Blackwell）两卡调试。正式训练在 cluster 上跑。
#   bash scripts/1027/run_cf_sft_debug_2gpu.sh               # 默认 GPU 1,2（GPU 0 常被 VLM server 占着）
#   GPUS=0,1 bash scripts/1027/run_cf_sft_debug_2gpu.sh
#   CONFIG=training/1026/qwen2.5-vl-3B-nuplan-cf-sft-navtrain bash ...   # 本机跑正式配置
# 计划：docs/1026/cf_sft_autovla_plan.md §4、§5 第 2 条。
set -u
REPO=/home/mh2803/vla/doc_vla_search
CONFIG="${CONFIG:-training/1026/qwen2.5-vl-3B-nuplan-cf-sft-debug-2gpu}"
GPUS="${GPUS:-1,2}"

source /home/mh2803/miniconda3/etc/profile.d/conda.sh
conda activate autovla_codeclean
cd "$REPO" || exit 1

# --- preflight ---
[ -e Qwen2.5-VL-3B-Instruct/config.json ] || { echo "ERROR: 找不到 ./Qwen2.5-VL-3B-Instruct（本机放在 /scratch/mh2803/vla/checkpoints/，软链接过来）"; exit 1; }
for f in $(grep -E '^\s*cf_index:' "config/${CONFIG}.yaml" | awk '{print $2}'); do
  [ -f "$f" ] || { echo "ERROR: 找不到 cf 索引 $f（先跑 tools/cf_data/export_index.py）"; exit 1; }
done
n_train=$(find /scratch/mh2803/vla/nuplan/navtrain_cot -maxdepth 1 -name '*.json' | wc -l)
[ "$n_train" -gt 0 ] || { echo "ERROR: navtrain_cot 为空（D0）"; exit 1; }

export CUDA_VISIBLE_DEVICES="$GPUS"
# 🔴 这台机器的 Blackwell 卡上 NCCL 的 P2P 和 SHM 都会 illegal memory access，两个都要关（只关 P2P 不够）
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PRINT_SFT_DEBUG="${PRINT_SFT_DEBUG:-1}"   # 每个 dataloader worker 打印前 N 条样本
export NUPLAN_MAPS_ROOT=/scratch/mh2803/vla/nuplan/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export OPENSCENE_DATA_ROOT=/scratch/mh2803/vla/nuplan
export NAVSIM_DEVKIT_ROOT="$REPO/navsim"

mkdir -p logs/1027
LOG="logs/1027/cf_sft_debug_$(date +%Y-%m-%d_%H-%M-%S).log"
echo "config : config/${CONFIG}.yaml"
echo "GPUs   : $GPUS   train json: $n_train"
echo "LOG    : $LOG"
python tools/run_sft.py --config "$CONFIG" 2>&1 | tee "$LOG"
exit "${PIPESTATUS[0]}"
