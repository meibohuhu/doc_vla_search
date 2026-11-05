#!/bin/bash
# ============================================================================
# [mh 2026/09/02] 预处理全量 trainval -> no-CoT JSON (166.3k) —— 166k 数据集的 Step 2
#
# 照 logs/0721/step1_preprocess_navtrain.sh 改的，两处不同：
#   * config 用 dataset/0902/nuplan-trainval-full（scene_filter: null -> 166.3k）
#   * 输出到 trainval_nocot_166k
#
# --fast 是必须的：它跳过 ~3.3M 次 JPEG 解码 + base64 往返，而这些结果在 no-CoT
# 路径下会被直接丢弃（见 dataset_utils/preprocessing/0721/fast_nocot_patch.py）。
# 实测 no-CoT 下与非 fast 逐字节一致，约 600x 提速。
#
# 用法:
#   tmux new -s prep
#   bash scripts/0902/preprocess_trainval_166k.sh
#   WORKERS=48 bash ...        # 默认 32（本机 72 核）
# ============================================================================
set -u
REPO=/home/mh2803/vla/doc_vla_search
CONFIG=dataset/0902/nuplan-trainval-full
OUT="${OUT:-/scratch/mh2803/vla/nuplan/trainval_nocot_166k}"
WORKERS="${WORKERS:-32}"
EXPECT=166282

source /home/mh2803/miniconda3/etc/profile.d/conda.sh
conda activate autovla_codeclean
cd "$REPO" || exit 1

export TOKENIZERS_PARALLELISM=false
export NUPLAN_MAPS_ROOT=/scratch/mh2803/vla/nuplan/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export OPENSCENE_DATA_ROOT=/scratch/mh2803/vla/nuplan
export NAVSIM_DEVKIT_ROOT="$REPO/navsim"

ENTRY=tools/preprocessing/nocot_sample_generation.py

# ---- 前置检查 ----
fail=0
[ -f "$ENTRY" ] || {
    echo "❌ 缺少预处理入口: $ENTRY"
    echo "   这个文件来自上游 AutoVLA，从没进过本仓库 git，且当前被 .gitignore 忽略。"
    echo "   需要从上游仓库/另一台机器复制 tools/preprocessing/ 回来。"
    fail=1
}
[ -f "config/${CONFIG}.yaml" ] || { echo "❌ 缺少 config/${CONFIG}.yaml"; fail=1; }

# sensor 覆盖度：只有 navtrain 那 446G 的话产出会远少于 166k
NLOG=$(ls /scratch/mh2803/vla/nuplan/sensor_blobs/trainval 2>/dev/null | wc -l)
NDONE=$(ls /scratch/mh2803/vla/nuplan/sensor_blobs/.done_trainval_camera_* 2>/dev/null | wc -l)
echo "sensor_blobs/trainval 的 log 目录数 : $NLOG   (全量应为 1310)"
echo "全量 camera 分片完成数              : $NDONE / 200"
if [ "$NDONE" -lt 200 ]; then
    echo "⚠️  全量 trainval camera 还没下完（$NDONE/200）。"
    echo "    现在跑会大量缺帧、产出远少于 166k。先跑："
    echo "      bash scripts/0902/download_trainval_camera_full.sh"
    echo "      bash scripts/0902/download_trainval_camera_full.sh verify"
    echo "    仍要继续请设 FORCE=1。"
    [ "${FORCE:-0}" = "1" ] || fail=1
fi
[ "$fail" -eq 0 ] || { echo "前置检查未通过。"; exit 1; }

STAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/0902/preprocess_trainval_166k_${STAMP}.log"
mkdir -p "$OUT" "$(dirname "$LOG")"

echo "=================================================================="
echo " STEP 2 : 全量 trainval -> no-CoT JSON (166.3k)"
echo " config : config/${CONFIG}.yaml   (scene_filter: null)"
echo " output : $OUT"
echo " workers: $WORKERS   (--fast)"
echo " LOG    : $LOG"
echo "=================================================================="

start=$(date +%s)
python "$ENTRY" \
    --config "$CONFIG" \
    --output_dir "$OUT" \
    --num_workers "$WORKERS" \
    --fast 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
end=$(date +%s); el=$((end-start))

# ⚠️ 别信脚本的“完成”字样，以实际 JSON 数量为准
N=$(find "$OUT" -maxdepth 1 -name '*.json' | wc -l)
echo
echo "=================================================================="
echo " 退出码    : $STATUS"
echo " 耗时      : $((el/3600))h $(( (el%3600)/60 ))m"
echo " JSON 数量 : $N   (期望 ≈ $EXPECT)"
if [ "$N" -lt $((EXPECT * 95 / 100)) ]; then
    echo " ⚠️  比期望少 >5%。最可能是 Step 1 有分片静默丢了："
    echo "     bash scripts/0902/download_trainval_camera_full.sh verify"
fi
echo " LOG       : $LOG"
echo "=================================================================="
exit "$STATUS"