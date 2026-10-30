#!/bin/bash
# ============================================================================
# 全量 nuPlan trainval sensor 下载（camera only）—— 166.3k 场景用
#
# 背景: runbook logs/0901_nvidia/166k_data_prep_runbook.md Step 2
#   navtrain 的 8 个 tgz 只覆盖 1192 log / 103,288 场景。要拿到论文 Table S1 的
#   166,282(scene_filter=null, fi=4, 全 1310 log)必须下 OpenScene 的
#   openscene_sensor_trainval_camera_{0..199}.tgz —— 分片非 log 对齐，没法只补差量。
#
# 产出:
#   $ROOT/
#   ├── maps/                    971MB
#   ├── navsim_logs/trainval/    7GB    (1310 log metadata)
#   └── sensor_blobs/trainval/   ~900GB 压缩 → 1310 log 相机图
#
# 用法:
#   bash download_trainval_full.sh              # maps + logs + sensor
#   bash download_trainval_full.sh sensor       # 只跑 sensor(200 分片)
#   bash download_trainval_full.sh verify       # 单独核对: 应齐 1310 log
#   步骤名: maps | logs | sensor | verify | all
#   PAR=6 bash download_trainval_full.sh        # 调并行度(默认 6)
#
# robust 机制(runbook 要求):
#   wget -c 落盘 → gzip -t 校验 → tar --strip-components=3 直接就位 → .done 标记
#   ⚠️ 绝不用 `wget -qO- | tar -xz` 流式管道 —— 截断会静默丢分片(本项目中招过 3 次)
#
# 全部步骤幂等 + 可断点续传。中断后重跑即可。
# ============================================================================
set -uo pipefail

ROOT="${AUTOVLA_DATA_ROOT:-/shared/rc/llm-gen-agent/mhu/auto_vla/nuplan}"
HF="https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1"
PAR="${PAR:-6}"
NSHARD="${NSHARD:-200}"          # openscene_sensor_trainval_camera_{0..199}
EXPECT_LOGS="${EXPECT_LOGS:-1310}"
MIN_FREE_GB="${MIN_FREE_GB:-30}" # 低于此值停手，避免把文件系统撑爆

STAGE="$ROOT/.stage"             # tgz 落盘区，与产出同盘(下完即删)
MARK="$ROOT/.markers"

mkdir -p "$ROOT" "$STAGE" "$MARK" || exit 1
cd "$ROOT" || exit 1

log()  { echo -e "\n\033[1;32m[$(date +%H:%M:%S)] $*\033[0m"; }
warn() { echo -e "\033[1;33m[$(date +%H:%M:%S)] $*\033[0m" >&2; }
err()  { echo -e "\033[1;31m[$(date +%H:%M:%S)] $*\033[0m" >&2; }
have() { [ -e "$1" ]; }
free_gb() { df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9'; }

# 解压加速: pigz 多核 gunzip，回退到 gzip
if command -v pigz >/dev/null 2>&1; then
    TARZ=(tar -I pigz -xf)
else
    TARZ=(tar -xzf)
fi

# ---------------------------------------------------------------------------
# 1. maps (971 MB)
# ---------------------------------------------------------------------------
step_maps() {
    if have maps; then log "maps/ 已存在，跳过"; return 0; fi
    log "下载 nuPlan maps (971MB)"
    wget -c -O nuplan-maps-v1.1.zip \
        https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip \
        || { err "maps 下载失败"; return 1; }
    unzip -q nuplan-maps-v1.1.zip -d . || { err "maps 解压失败"; return 1; }
    rm -f nuplan-maps-v1.1.zip
    # zip 里的目录名是 nuplan-maps-v1.0（跟 NUPLAN_MAP_VERSION 一致）
    [ -d nuplan-maps-v1.0 ] && mv nuplan-maps-v1.0 maps
    log "maps 完成"
}

# ---------------------------------------------------------------------------
# 2. trainval metadata → navsim_logs/trainval  (7GB, 1310 个 pkl)
#    SceneFilter 只读这份 metadata；scene_filter=null 时它数出 166,282 场景
# ---------------------------------------------------------------------------
step_logs() {
    if have navsim_logs/trainval; then log "navsim_logs/trainval 已存在，跳过"; return 0; fi
    log "下载 trainval metadata (7GB)"
    mkdir -p navsim_logs
    wget -c -O "$STAGE/openscene_metadata_trainval.tgz" \
        "$HF/openscene_metadata_trainval.tgz" || { err "metadata 下载失败"; return 1; }
    gzip -t "$STAGE/openscene_metadata_trainval.tgz" \
        || { err "metadata gzip 校验失败，已删除，请重跑"; rm -f "$STAGE/openscene_metadata_trainval.tgz"; return 1; }
    "${TARZ[@]}" "$STAGE/openscene_metadata_trainval.tgz" -C "$STAGE" || { err "metadata 解压失败"; return 1; }
    mv "$STAGE/openscene-v1.1/meta_datas" navsim_logs/trainval
    rm -rf "$STAGE/openscene-v1.1" "$STAGE/openscene_metadata_trainval.tgz"
    log "navsim_logs/trainval 完成 ($(find navsim_logs/trainval -name '*.pkl' | wc -l) 个 pkl)"
}

# ---------------------------------------------------------------------------
# 3. 全量 trainval sensor —— 200 分片
#
#    tar 内结构: openscene-v1.1/sensor_blobs/trainval/<log>/<CAM>/<hash>.jpg
#    --strip-components=3 剥掉前三级 → 直接落在 sensor_blobs/trainval/<log>/...
#    先 gzip -t 验完整再解，所以不会因截断把半个分片写进最终目录树。
# ---------------------------------------------------------------------------
one_shard() {
    local i="$1"
    local marker="$MARK/.done_sensor_$i"
    local tgz="$STAGE/openscene_sensor_trainval_camera_$i.tgz"

    have "$marker" && return 0

    local avail; avail=$(free_gb)
    if [ "${avail:-0}" -lt "$MIN_FREE_GB" ]; then
        err "分片 $i 跳过: 剩余空间 ${avail}G < ${MIN_FREE_GB}G 阈值"
        return 2
    fi

    echo "  -> [$i] 下载"
    wget -c -q --show-progress --progress=dot:giga -O "$tgz" \
        "$HF/openscene_sensor_trainval_camera/openscene_sensor_trainval_camera_$i.tgz" \
        || { err "[$i] 下载失败(重跑可续传)"; return 1; }

    echo "  -> [$i] gzip -t 校验"
    if ! gzip -t "$tgz" 2>/dev/null; then
        err "[$i] gzip 校验失败(下载被截断) → 删除重下"
        rm -f "$tgz"
        return 1
    fi

    echo "  -> [$i] 解压就位"
    mkdir -p "$ROOT/sensor_blobs/trainval"
    "${TARZ[@]}" "$tgz" -C "$ROOT/sensor_blobs/trainval" --strip-components=3 \
        || { err "[$i] 解压失败"; return 1; }

    rm -f "$tgz"
    touch "$marker"
    echo "  -> [$i] ✅ 完成 (剩余 $(free_gb)G)"
}

step_sensor() {
    local done_n; done_n=$(find "$MARK" -name '.done_sensor_*' 2>/dev/null | wc -l)
    log "全量 trainval sensor: $NSHARD 分片，已完成 $done_n，并行 $PAR"
    log "剩余空间 $(free_gb)G  (每分片 ~4.5G 压缩)"

    export -f one_shard log warn err have free_gb
    export ROOT HF STAGE MARK MIN_FREE_GB
    export TARZ_STR="${TARZ[*]}"

    # bash -c 里重建 TARZ 数组
    printf "%s\n" $(seq 0 $((NSHARD - 1))) | xargs -I{} -P "$PAR" bash -c '
        read -ra TARZ <<< "$TARZ_STR"
        one_shard {}
    '

    done_n=$(find "$MARK" -name '.done_sensor_*' 2>/dev/null | wc -l)
    log "sensor 分片完成 $done_n / $NSHARD"
    if [ "$done_n" -lt "$NSHARD" ]; then
        warn "有 $((NSHARD - done_n)) 个分片未完成 —— 重跑本脚本续传"
        return 1
    fi
}

# ---------------------------------------------------------------------------
# 4. verify —— 应齐 EXPECT_LOGS(1310) 个 log 目录
#
#    ⚠️ 这只比对 log 目录名。真正逐张核对图片存在请用:
#       python scripts/0721/verify_sensor_coverage.py navtrain 300
# ---------------------------------------------------------------------------
step_verify() {
    log "verify"
    local shards logs meta
    shards=$(find "$MARK" -name '.done_sensor_*' 2>/dev/null | wc -l)
    logs=$(find "$ROOT/sensor_blobs/trainval" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    meta=$(find "$ROOT/navsim_logs/trainval" -name '*.pkl' 2>/dev/null | wc -l)

    printf "  分片 .done       : %s / %s\n" "$shards" "$NSHARD"
    printf "  sensor log 目录  : %s / %s\n" "$logs" "$EXPECT_LOGS"
    printf "  navsim_logs pkl  : %s / %s\n" "$meta" "$EXPECT_LOGS"
    # ⚠️ 不要默认跑 `du -sh sensor_blobs/trainval`：1.4TB / 几百万个小文件，
    #    在 CephFS 上要走很久，而且会把 MDS 打满、拖垮同时在跑的解压。
    #    真要看占用用 DU=1 显式打开。
    if [ "${DU:-0}" = "1" ]; then
        printf "  占用             : %s\n" "$(du -sh "$ROOT/sensor_blobs/trainval" 2>/dev/null | cut -f1)"
    else
        printf "  占用             : (跳过，DU=1 可查；df 剩余 %sG)\n" "$(free_gb)"
    fi

    # metadata 有、sensor 没有的 log —— 预处理时这些场景会被静默跳过
    if [ "$meta" -gt 0 ] && [ "$logs" -gt 0 ]; then
        local missing
        missing=$(comm -23 \
            <(cd "$ROOT/navsim_logs/trainval" && ls *.pkl 2>/dev/null | sed 's/\.pkl$//' | sort) \
            <(cd "$ROOT/sensor_blobs/trainval" && ls -d */ 2>/dev/null | tr -d / | sort) | wc -l)
        printf "  缺 sensor 的 log : %s\n" "$missing"
        [ "$missing" -gt 0 ] && warn "有 $missing 个 log 只有 metadata 没有图 → 分片没下满，重跑续传"
    fi

    if [ "$shards" -eq "$NSHARD" ] && [ "$logs" -ge "$EXPECT_LOGS" ]; then
        log "✅ 齐了。下一步: bash scripts/0901/preprocess_166k.sh (应产出 ≈166,282 json)"
        return 0
    fi
    err "❌ 未齐 —— 重跑 bash scripts/0901/download_trainval_full.sh 续传"
    return 1
}

# ---------------------------------------------------------------------------
main() {
    log "目标目录: $ROOT"
    df -h "$ROOT" | tail -1

    case "${1:-all}" in
        maps)   step_maps ;;
        logs)   step_logs ;;
        sensor) step_sensor ;;
        verify) step_verify ;;
        all)    step_maps && step_logs && step_sensor; step_verify ;;
        *)      echo "未知步骤: $1  (maps|logs|sensor|verify|all)"; exit 1 ;;
    esac
}

main "$@"
