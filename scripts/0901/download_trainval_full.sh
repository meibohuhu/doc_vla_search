#!/bin/bash
# ============================================================================
# 下载【全量 trainval camera】sensor → sensor_blobs/trainval  (200 分片, ~2TB)
# 用于 166.3k 口径（scene_filter=null, fi=4, 全 1310 log）。
#
# 与现有 scripts/download_nuplan_autovla.sh 的 step_navtrain 区别：
#   * 那个只下 navtrain 8 分片(445GB, 1192 log 子集)；
#   * 这个下 openscene_sensor_trainval_camera_{0..199}(~2TB, 全 1310 log)。
#   * 分片非 log 对齐，没法只补差量 → 只能下全 200 份（与现有 446G 重叠部分会被硬链接覆盖）。
#
# 复用同一套 robust 机制：wget -c 落盘 → gzip -t 校验 → tar 解压 → cp -rlf 硬链接 →
#   打 .done 标记 → 删 tgz。全幂等，中断重跑自动续传。
#   ★ 绝不用 `wget -qO- | tar -xz` 流式管道（截断静默丢分片，本项目中招过 3 次）。
#
# 用法:
#   bash scripts/0901/download_trainval_full.sh            # 下全部 200 分片
#   PAR=8 bash scripts/0901/download_trainval_full.sh      # 调并行度（默认 6）
#   bash scripts/0901/download_trainval_full.sh verify     # 只核对，不下载
#   bash scripts/0901/download_trainval_full.sh _one 137   # 内部：单分片（xargs 用）
# ============================================================================
set -uo pipefail

SELF="$(readlink -f "$0")"
REPO="$(cd "$(dirname "$SELF")/../.." && pwd)"
ROOT="${AUTOVLA_DATA_ROOT:-/data/autovla_data/nuplan}"
HF="https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1"
PAR="${PAR:-6}"          # 8 核机 + 单流实测 ~22MB/s，6 并行聚合约 100MB/s
NSPLIT=200

mkdir -p "$ROOT/sensor_blobs/trainval"
cd "$ROOT" || exit 1
log() { echo -e "\n\033[1;32m[$(date +%H:%M:%S)] $*\033[0m"; }

# 下载并解包一个 trainval camera 分片（幂等）
_one() {
    local i="$1"
    local marker="sensor_blobs/.done_trainval_cam_${i}"
    local f="_trainval_cam_${i}.tgz"
    [ -e "$marker" ] && { echo "  · 分片 $i 已完成"; return 0; }

    wget -c -q "$HF/openscene_sensor_trainval_camera/openscene_sensor_trainval_camera_${i}.tgz" -O "$f" \
        || { echo "  ✗ 分片 $i 下载失败"; [ -s "$f" ] || rm -f "$f"; return 1; }
    # ★ 完整性校验：截断的 gzip 会在这里被抓出来
    gzip -t "$f" 2>/dev/null \
        || { echo "  ✗ 分片 $i 校验失败(截断)，已删除，重跑可续"; rm -f "$f"; return 1; }
    # ★ 竞态修复：每个分片解压到【独立临时目录】，不再共用 openscene-v1.1/。
    #   原来 6 个并行 job 都往 openscene-v1.1/ 解压 + rm -rf，一个 job 的 rm 会删掉
    #   另一个正解压到一半的 log → log 目录在但帧残缺（v1 的坑，实测丢 ~9% 帧）。
    local ext="_ext_${i}"
    rm -rf "$ext"; mkdir -p "$ext"
    tar -xzf "$f" -C "$ext" || { echo "  ✗ 分片 $i 解压失败"; rm -rf "$ext"; return 1; }
    local src="$ext/openscene-v1.1/sensor_blobs"
    [ -d "$src/trainval" ] && src="$src/trainval"
    cp -rlf "$src"/* sensor_blobs/trainval/          # 同盘硬链接，秒级，不占额外空间
    rm -rf "$ext" "$f"
    touch "$marker"
    echo "  ✓ 分片 $i"
}

step_download() {
    log "下载全量 trainval camera: $NSPLIT 分片, ~2TB, 并行 $PAR"
    log "单流实测 ~22MB/s → 预计 5–8h（视带宽）。可随时中断，重跑续传。"
    seq 0 $((NSPLIT-1)) | xargs -I{} -P "$PAR" bash "$SELF" _one {}

    local n; n=$(ls sensor_blobs/.done_trainval_cam_* 2>/dev/null | wc -l)
    log "完成 $n/$NSPLIT 分片"
    [ "$n" -eq "$NSPLIT" ] || { log "⚠️ 未下满，重跑本脚本自动续传缺失分片"; return 1; }
    step_verify
}

# 核对：166k 需要的全部 1310 log 是否都有 sensor（用默认 filter 的 log 全集 = navsim_logs 里的全部 log）
step_verify() {
    log "完整性核对（全 1310 log 是否都有 sensor）"
    python3 - "$ROOT" <<'PY'
import os, sys
root = sys.argv[1]
meta = os.path.join(root, "navsim_logs", "trainval")
sens = os.path.join(root, "sensor_blobs", "trainval")
need = {os.path.splitext(f)[0] for f in os.listdir(meta)} if os.path.isdir(meta) else set()
have = set(os.listdir(sens)) if os.path.isdir(sens) else set()
miss = need - have
flag = "✅" if not miss else "❌"
print(f"{flag} 全量 trainval: 需要 {len(need)} log(按 metadata), 实有 sensor {len(have & need)}, 缺 {len(miss)}")
for m in sorted(miss)[:8]:
    print(f"      缺: {m}")
print("  注: 下满后应 = 1310。缺的多半是某分片静默失败 → 重跑 download 续传。")
PY
    n=$(ls sensor_blobs/.done_trainval_cam_* 2>/dev/null | wc -l)
    echo "  分片标记: $n/$NSPLIT"
}

case "${1:-download}" in
    download) df -h "$ROOT" | tail -1; step_download ;;
    verify)   step_verify ;;
    _one)     _one "$2" ;;
    *) echo "用法: $0 [download|verify]"; exit 1 ;;
esac
