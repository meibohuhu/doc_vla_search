#!/bin/bash
# ============================================================================
# 预处理全量 trainval → 166.3k no-CoT JSON（scene_filter=null, fi=4, 全 1310 log）
#
#   bash scripts/0901/preprocess_166k.sh
#
# 依赖：
#   * navsim_logs/trainval（14G metadata，已在盘）—— SceneFilter 只读它
#   * sensor_blobs/trainval 全量（~2TB，先跑 download_trainval_full.sh）
#     ⚠️ --fast 下 no-CoT 不解码图像，但仍需图像【路径存在校验】依赖 SceneLoader；
#        少数缺图的场景会被跳过（数量应仍 ≈166,282）。
# ============================================================================
set -uo pipefail

REPO=/home/nvidia/workspace/doc_drive_search/other_repo/AutoVLA
PY=/data/autovla_data/envs/autovla/bin/python
OUT=/data/autovla_data/nuplan/trainval_nocot_166k
CONFIG=dataset/0901/nuplan-trainval-full-166k

cd "$REPO"
export NUPLAN_MAPS_ROOT=/data/autovla_data/nuplan/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export OPENSCENE_DATA_ROOT=/data/autovla_data/nuplan
export NAVSIM_DEVKIT_ROOT="$REPO/navsim"
export TOKENIZERS_PARALLELISM=false

# preflight
[ -d /data/autovla_data/nuplan/navsim_logs/trainval ] || { echo "ERROR: 缺 navsim_logs/trainval metadata"; exit 1; }
nseg=$(ls /data/autovla_data/nuplan/sensor_blobs/.done_trainval_cam_* 2>/dev/null | wc -l)
[ "$nseg" -eq 200 ] || echo "⚠️ trainval sensor 分片只完成 $nseg/200 —— 缺图的场景会被跳过。要全量请先下满。"

STAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/0901_preprocess_166k_${STAMP}.log"
mkdir -p "$REPO/logs"

echo "=================================================================="
echo " 预处理 166.3k  (scene_filter=null → fi=4 全 1310 log)"
echo " config : config/${CONFIG}.yaml"
echo " 输出   : $OUT"
echo " 预期数 : ≈166,282 json"
echo " LOG    : $LOG"
echo "=================================================================="

$PY tools/preprocessing/nocot_sample_generation.py \
    --config "$CONFIG" \
    --output_dir "$OUT" \
    --num_workers 32 \
    --fast 2>&1 | tee "$LOG"

n=$(find "$OUT" -maxdepth 1 -name '*.json' | wc -l)
echo
echo "=================================================================="
echo " 产出 json: $n   (预期 ≈166,282)"
[ "$n" -ge 160000 ] && echo " ✅ 数量正常" || echo " ⚠️ 明显偏少 —— 多半是 sensor 分片没下满，回去 download verify"
echo " 训练时把 config 的 data.train.json_dataset_path 指到: $OUT"
echo "=================================================================="
