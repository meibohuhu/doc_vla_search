#!/bin/bash
# 建 navtrain 的 metric_cache —— RL 正式训练的前置（flow §5.1）。
#
#   # 先探速（只算 200 个 scene，几分钟，打印 s/token）
#   PROBE=200 bash logs/0902/build_navtrain_metric_cache.sh
#
#   # 默认只建 5 万帧子集（SPLIT=navtrain50k）。要全量:
#   SPLIT=navtrain bash logs/0902/build_navtrain_metric_cache.sh
#
#   # 正式跑（长任务，挂 tmux）
#   tmux new -s cache
#   bash logs/0902/build_navtrain_metric_cache.sh
#
#   WORKER=ray_distributed PROBE=200 bash logs/0902/build_navtrain_metric_cache.sh   # 比较并行收益
#
# ─────────────────────────────────────────────────────────────────────────
# 为什么需要:RL 的 reward 靠 rl_pdm_score(traj, token) 查 metric_cache。
#   navtest 那份是齐的（12,146），**但不能拿 navtest 当训练集** ——
#   训完 navtest 就不再是 held-out，PDMS 这个论文主指标直接作废。
#
# 成本（按 navtest 实测外推）:
#   navtest  12,146 个 cache / 1.68 小时（目录 mtime 21:43 → 23:24）≈ 0.50 s/token
#   navtrain 101,288 个 → 同样设置约 14 小时、约 48 GB
#   /data 余量 5.3T，够。
#
# ⭐ 纯 CPU（PDM 仿真是 numpy），不占 GPU —— 应该和 CoT-SFT **同时**跑，两者不抢资源。
# ⚠️ 本机只有 8 核，而 navsim 默认 worker 是 sequential。先 PROBE 量一下，
#    再决定要不要换 ray_distributed —— 别盲开 14 小时。
set -u

REPO=/home/nvidia/workspace/doc_drive_search/other_repo/AutoVLA
PY=/data/autovla_data/envs/autovla/bin/python
SPLIT="${SPLIT:-navtrain50k}"      # navtrain50k（5 万帧子集，省一半时间）| navtrain（全量 101k）
CACHE_PATH="/data/autovla_data/nuplan/${SPLIT}_metric_cache"
WORKER="${WORKER:-ray_distributed}"
PROBE="${PROBE:-0}"

cd "$REPO"

export PYTHONPATH="$REPO/navsim:${PYTHONPATH:-}"
export NAVSIM_DEVKIT_ROOT="$REPO/navsim"
export OPENSCENE_DATA_ROOT=/data/autovla_data/nuplan
export NUPLAN_MAPS_ROOT=/data/autovla_data/nuplan/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NAVSIM_EXP_ROOT=/data/autovla_data/nuplan/exp

# --- preflight ---
n_log=$(ls "$OPENSCENE_DATA_ROOT"/navsim_logs/trainval/*.pkl 2>/dev/null | wc -l)
[ "$n_log" -eq 0 ] && { echo "ERROR: 找不到 navsim_logs/trainval/*.pkl"; exit 1; }
[ -d "$NUPLAN_MAPS_ROOT" ] || { echo "ERROR: 找不到 maps: $NUPLAN_MAPS_ROOT"; exit 1; }
avail=$(df -BG --output=avail /data | tail -1 | tr -dc '0-9')
[ "$avail" -lt 60 ] && { echo "ERROR: /data 只剩 ${avail}G，navtrain cache 约需 48G"; exit 1; }

EXTRA=""
if [ "$PROBE" -gt 0 ]; then
    CACHE_PATH="${CACHE_PATH}_probe"
    EXTRA="train_test_split.scene_filter.max_scenes=$PROBE"
    echo "[PROBE] 只算 $PROBE 个 scene → $CACHE_PATH"
fi

echo "=================================================================="
echo " navtrain metric_cache"
echo " logs    : $n_log 个 trainval pkl"
echo " split   : $SPLIT"
echo " worker  : $WORKER"
echo " out     : $CACHE_PATH"
echo " 磁盘余量: ${avail}G"
echo "=================================================================="

T0=$(date +%s)
$PY "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
    train_test_split="$SPLIT" \
    worker="$WORKER" \
    cache.cache_path="$CACHE_PATH" \
    $EXTRA
STATUS=$?
T1=$(date +%s)

n_cache=$(ls "$CACHE_PATH"/*/*/*/metric_cache.pkl 2>/dev/null | wc -l)
el=$((T1 - T0))
echo
echo "=================================================================="
echo " exit code : $STATUS"
echo " 产出      : $n_cache 个 metric_cache.pkl"
echo " 用时      : ${el}s"
if [ "$n_cache" -gt 0 ]; then
    echo " 速率      : $(awk "BEGIN{printf \"%.3f\", $el/$n_cache}") s/token"
    echo " 外推 50000 个 → $(awk "BEGIN{printf \"%.1f\", $el/$n_cache*50000/3600}") 小时"
fi
echo "=================================================================="
exit "$STATUS"
