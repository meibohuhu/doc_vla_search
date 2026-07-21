#!/bin/bash
# ============================================================================
# AutoVLA nuPlan / NAVSIM 数据下载（camera only，无 lidar）
#
# 产出目录结构（AutoVLA 的 placeholder 替换机制要求的两级结构）：
#   $ROOT/
#   ├── maps/                    971MB
#   ├── navsim_logs/
#   │   ├── trainval/            7GB   (navtrain 的 log 来自 trainval metadata)
#   │   └── test/                476MB
#   └── sensor_blobs/
#       ├── trainval/            445GB (navtrain current+history)
#       └── test/                128GB (navtest camera)
#                                ------
#                          合计   ~582GB
#
# 用法:
#   bash download_nuplan_autovla.sh            # 全部
#   bash download_nuplan_autovla.sh maps       # 只跑某一步
#   步骤名: maps | logs | navtrain | test
#
# 全部步骤幂等 + 可断点续传（wget -c）。中断后重跑即可。
# ============================================================================
set -uo pipefail

ROOT="${AUTOVLA_DATA_ROOT:-/data/autovla_data/nuplan}"
HF="https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1"
S3="https://s3.eu-central-1.amazonaws.com/avg-projects-2/navsim"
# 并行度：本机 8 核，tar 解压是 CPU 密集的，4 比 8 更稳
PAR="${PAR:-4}"

mkdir -p "$ROOT"
cd "$ROOT" || exit 1

log() { echo -e "\n\033[1;32m[$(date +%H:%M:%S)] $*\033[0m"; }
have() { [ -e "$1" ]; }

# ---------------------------------------------------------------------------
# 1. maps (971 MB)
# ---------------------------------------------------------------------------
step_maps() {
    if have maps; then log "maps/ 已存在，跳过"; return; fi
    log "下载 nuPlan maps (971MB)"
    wget -c https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip
    unzip -q nuplan-maps-v1.1.zip -d .
    rm -f nuplan-maps-v1.1.zip
    # zip 里的目录名是 nuplan-maps-v1.0（跟 NUPLAN_MAP_VERSION 一致）
    [ -d nuplan-maps-v1.0 ] && mv nuplan-maps-v1.0 maps
    log "maps 完成"
}

# ---------------------------------------------------------------------------
# 2. metadata → navsim_logs/{trainval,test}
#    trainval metadata 是完整 log（7GB），navtrain 只是它的一个 scene_filter
# ---------------------------------------------------------------------------
step_logs() {
    mkdir -p navsim_logs
    for split in trainval test; do
        if have "navsim_logs/$split"; then log "navsim_logs/$split 已存在，跳过"; continue; fi
        log "下载 $split metadata"
        wget -c "$HF/openscene_metadata_${split}.tgz"
        tar -xzf "openscene_metadata_${split}.tgz"
        mv openscene-v1.1/meta_datas "navsim_logs/$split"
        rm -rf openscene-v1.1 "openscene_metadata_${split}.tgz"
        log "navsim_logs/$split 完成"
    done
}

# ---------------------------------------------------------------------------
# 3. navtrain sensors → sensor_blobs/trainval  (445GB, 8 个 tgz)
#
#    ⚠️ history 不能省：navtrain.yaml 是 num_history_frames=4，
#       AutoVLA 每路相机要读 4 帧。"300GB 无 history 版"对 AutoVLA 不可用。
#
#    优化：官方脚本用 `rsync -rv` 搬 445GB 小文件（慢）。同一文件系统下
#    改用 `cp -rl`（硬链接）几乎是瞬时的，随后 rm 源目录，数据靠硬链接存活。
# ---------------------------------------------------------------------------
step_navtrain() {
    mkdir -p sensor_blobs/trainval
    for kind in current history; do
        for i in 1 2 3 4; do
            marker="sensor_blobs/.done_navtrain_${kind}_${i}"
            if have "$marker"; then log "navtrain_${kind}_${i} 已完成，跳过"; continue; fi

            log "下载 navtrain_${kind}_${i}.tgz  (共 8 份，合计 445GB)"
            wget -c "$S3/navtrain_${kind}_${i}.tgz" || { echo "下载失败: ${kind}_${i}"; return 1; }

            log "解压 navtrain_${kind}_${i}.tgz"
            tar -xzf "navtrain_${kind}_${i}.tgz" || { echo "解压失败: ${kind}_${i}"; return 1; }

            log "硬链接搬运 ${kind}_split_${i} → sensor_blobs/trainval"
            cp -rl "${kind}_split_${i}"/* sensor_blobs/trainval/
            rm -rf "${kind}_split_${i}" "navtrain_${kind}_${i}.tgz"
            touch "$marker"
        done
    done
    log "navtrain 全部完成"
}

# ---------------------------------------------------------------------------
# 4. navtest camera → sensor_blobs/test  (128GB, 32 个 tgz, 流式解压不落 tgz)
#    lidar 循环不要（AutoVLA 的 SensorConfig 是 lidar_pc=False）
# ---------------------------------------------------------------------------
step_test() {
    if have sensor_blobs/test; then log "sensor_blobs/test 已存在，跳过"; return; fi
    log "下载 navtest camera (32 份, 128GB, 并行 $PAR)"
    printf "%s\n" {0..31} | xargs -I{} -P "$PAR" bash -c \
        "echo '  -> camera split {}'; wget -qO- '$HF/openscene_sensor_test_camera/openscene_sensor_test_camera_{}.tgz' | tar -xz"
    mkdir -p sensor_blobs
    mv openscene-v1.1/sensor_blobs sensor_blobs/test
    rm -rf openscene-v1.1
    log "sensor_blobs/test 完成"
}

# ---------------------------------------------------------------------------
main() {
    log "目标目录: $ROOT"
    df -h "$ROOT" | tail -1

    case "${1:-all}" in
        maps)     step_maps ;;
        logs)     step_logs ;;
        navtrain) step_navtrain ;;
        test)     step_test ;;
        all)      step_maps && step_logs && step_test && step_navtrain ;;
        *)        echo "未知步骤: $1  (maps|logs|navtrain|test|all)"; exit 1 ;;
    esac

    log "完成。当前占用："
    du -sh "$ROOT"/* 2>/dev/null
}

main "$@"
