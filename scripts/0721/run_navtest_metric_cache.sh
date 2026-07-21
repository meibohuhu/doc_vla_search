#!/bin/bash
# ============================================================================
# 构建 navtest 的 metric cache —— PDMS 评测的前置条件
#
# 与 navsim/scripts/evaluation/run_metric_caching.sh 的区别：
#   * 那个脚本默认是 warmup_test_e2e，navtest 的两行被注释掉了
#   * 那个脚本用相对路径 ./dataset/nuplan，依赖 CWD；这里全部用绝对路径
#   * 这里用 nice 降优先级，避免跟同机的训练任务抢 CPU
# ============================================================================
set -euo pipefail

# 向上找到含 setup.py 的仓库根，与本脚本所在层级无关（文件被挪进日期目录也不会断）
REPO="$(cd "$(dirname "$(readlink -f "$0")")" && while [ ! -f setup.py ] && [ "$PWD" != / ]; do cd ..; done; pwd)"
DATA=/data/autovla_data/nuplan

export PYTHONPATH="$REPO/navsim:${PYTHONPATH:-}"
export NAVSIM_DEVKIT_ROOT="$REPO/navsim"
export NAVSIM_EXP_ROOT="$DATA/exp"
export OPENSCENE_DATA_ROOT="$DATA"
export NUPLAN_MAPS_ROOT="$DATA/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export TOKENIZERS_PARALLELISM=false

TRAIN_TEST_SPLIT="${1:-navtest}"
# worker: ray_distributed (并行) | sequential (~2s/场景单线程) | single_machine_thread_pool
WORKER="${2:-ray_distributed}"
# 本机 8 核，且同机有 8 卡训练任务在跑（其 dataloader num_workers=16）。
# 限制为 4，配合 nice -n 15 让出优先级，避免拖慢训练。null = 用满所有核。
NWORKERS="${3:-4}"
CACHE_PATH="$DATA/${TRAIN_TEST_SPLIT}_metric_cache"

mkdir -p "$NAVSIM_EXP_ROOT"
echo "split      : $TRAIN_TEST_SPLIT"
echo "worker     : $WORKER (threads_per_node=$NWORKERS)"
echo "cache_path : $CACHE_PATH"
echo "maps       : $NUPLAN_MAPS_ROOT"

EXTRA=()
case "$WORKER" in
    ray_distributed)           EXTRA+=("worker.threads_per_node=$NWORKERS") ;;
    single_machine_thread_pool) EXTRA+=("worker.max_workers=$NWORKERS") ;;
esac

cd "$REPO"
nice -n 15 /data/autovla_data/envs/autovla/bin/python \
    "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
    train_test_split="$TRAIN_TEST_SPLIT" \
    worker="$WORKER" \
    "${EXTRA[@]}" \
    cache.cache_path="$CACHE_PATH"
