#!/bin/bash
source /home/mh2803/miniconda3/etc/profile.d/conda.sh
conda activate autovla_codeclean
cd /home/mh2803/vla/doc_vla_search
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
# Blackwell RTX PRO 6000 (no NVLink, PCIe): NCCL P2P + SHM transports fault with
# "illegal memory access". Forcing the network path fixes multi-GPU collectives.
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NUPLAN_MAPS_ROOT=/scratch/mh2803/vla/nuplan/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export OPENSCENE_DATA_ROOT=/scratch/mh2803/vla/nuplan
export NAVSIM_DEVKIT_ROOT=/home/mh2803/vla/doc_vla_search/navsim
CONFIG="${1:-training/qwen2.5-vl-3B-nuplan-nocot-sft}"
echo "[launch] config=$CONFIG"
exec python tools/run_sft.py --config "$CONFIG"
