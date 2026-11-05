#!/bin/bash
# ============================================================================
# AutoVLA nuPlan / NAVSIM 数据下载（camera only，无 lidar）
#
# 产出目录结构（AutoVLA 的 placeholder 替换机制要求的两级结构）：
#   $ROOT/
#   ├── maps/                    971MB
#   ├── navsim_logs/
#   │   ├── mini/                0.53GB
#   │   ├── trainval/            7GB   (navtrain 的 log 来自 trainval metadata)
#   │   └── test/                476MB
#   └── sensor_blobs/
#       ├── mini/                ~90GB (camera only)
#       ├── trainval/            445GB (navtrain current+history)
#       └── test/                128GB (navtest camera)
#
# 设计要点（v2，吸取教训）
#   * 每个分片：wget -c 落盘 -> gzip -t 校验 -> 解压 -> 打 .done 标记 -> 删 tgz
#     v1 用 `wget -qO- | tar -xz` 流式管道，下载被截断时 gzip 报错但管道退出码
#     被忽略 -> 分片静默丢失（实际丢了 20 个 navtest log 才发现）。
#   * 所有步骤幂等；中断后重跑自动续传，已完成的分片直接跳过。
#   * tar 解出的目录里本身带 <split>/ 子层，直接 mv 会多套一层，已处理。
#   * mini / test 的完成标记沿用 .done_mini_camera_* / .done_test_camera_*，
#     已下完的数据重跑会跳过，不会按新标记名再拉一遍。
#
# 用法:
#   AUTOVLA_DATA_ROOT=/scratch/mh2803/vla/nuplan bash scripts/download_nuplan_autovla.sh
#   bash scripts/download_nuplan_autovla.sh test          # 单步
#   bash scripts/download_nuplan_autovla.sh mini_navtrain # maps + mini + trainval logs + navtrain
#   bash scripts/download_nuplan_autovla.sh verify        # 只做完整性核对，不下载
#   步骤名: maps | mini | logs | trainval_logs | navtrain | test | verify
#           | mini_navtrain | trainval_test | all
# ============================================================================
set -uo pipefail

SELF="$(readlink -f "$0")"
REPO="$(cd "$(dirname "$SELF")/.." && pwd)"
ROOT="${AUTOVLA_DATA_ROOT:-/data/autovla_data/nuplan}"
HF="${HF:-https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1}"
S3="https://s3.eu-central-1.amazonaws.com/avg-projects-2/navsim"
PAR="${PAR:-4}"          # 本机 8 核且被训练任务占用，别调高

mkdir -p "$ROOT"
cd "$ROOT" || exit 1

log()  { echo -e "\n\033[1;32m[$(date +%H:%M:%S)] $*\033[0m"; }
have() { [ -e "$1" ]; }

# ---------------------------------------------------------------------------
# 通用：下载一个 tgz 分片并校验解压（幂等）
#   $1 = 标记名   $2 = URL   $3 = 本地临时文件名
# ---------------------------------------------------------------------------
fetch_split() {
    local marker="$1" url="$2" f="$3"
    if have "$marker"; then echo "  · $(basename "$marker") 已完成"; return 0; fi

    # wget 失败时也要清掉残留（可能是 0 字节），否则下次 gzip -t 会对空文件报错
    wget -c -q "$url" -O "$f" || { echo "  ✗ 下载失败: $url"; [ -s "$f" ] || rm -f "$f"; return 1; }
    # ★ 关键：校验完整性。v1 缺这一步导致截断的分片被当成功
    gzip -t "$f" 2>/dev/null   || { echo "  ✗ 校验失败(截断): $f  已删除，重跑可续"; rm -f "$f"; return 1; }
    tar -xzf "$f"              || { echo "  ✗ 解压失败: $f"; return 1; }
    rm -f "$f"
    touch "$marker"
    echo "  ✓ $(basename "$marker")"
}

# meta_datas/ 里本身带 <split>/ 子目录，直接 mv 会多套一层
install_meta() {
    local split="$1"
    if [ -d "openscene-v1.1/meta_datas/$split" ]; then
        mv "openscene-v1.1/meta_datas/$split" "navsim_logs/$split"
    else
        mv openscene-v1.1/meta_datas "navsim_logs/$split"
    fi
    rm -rf openscene-v1.1
}

# 同盘硬链接搬 sensor；源目录可能再套一层 <split>/
link_sensors() {
    local split="$1"
    local src="openscene-v1.1/sensor_blobs"
    [ -d "$src/$split" ] && src="$src/$split"
    mkdir -p "sensor_blobs/$split"
    cp -rlf "$src"/* "sensor_blobs/$split/"
    rm -rf openscene-v1.1
}

# ---------------------------------------------------------------------------
# 1. maps (971 MB)
# ---------------------------------------------------------------------------
step_maps() {
    if have maps; then log "maps/ 已存在，跳过"; return; fi
    log "下载 nuPlan maps (971MB)"
    wget -c https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip
    unzip -q nuplan-maps-v1.1.zip -d . && rm -f nuplan-maps-v1.1.zip
    [ -d nuplan-maps-v1.0 ] && mv nuplan-maps-v1.0 maps
    log "maps 完成"
}

# ---------------------------------------------------------------------------
# 2a. mini metadata + camera → navsim_logs/mini + sensor_blobs/mini (~90GB)
#     lidar 不要（AutoVLA SensorConfig lidar_pc=False）
# ---------------------------------------------------------------------------
_one_mini() {
    fetch_split "sensor_blobs/.done_mini_camera_$1" \
                "$HF/openscene_sensor_mini_camera/openscene_sensor_mini_camera_$1.tgz" \
                "_mini_$1.tgz"
}

step_mini() {
    mkdir -p navsim_logs sensor_blobs

    if have navsim_logs/mini; then
        log "navsim_logs/mini 已存在，跳过"
    else
        log "下载 mini metadata"
        fetch_split ".done_meta_mini" "$HF/openscene_metadata_mini.tgz" "meta_mini.tgz" || return 1
        install_meta mini
        log "navsim_logs/mini 完成 ($(ls navsim_logs/mini | wc -l) 个 log)"
    fi

    if have sensor_blobs/mini && have sensor_blobs/.done_mini_camera; then
        log "sensor_blobs/mini 已完成，跳过"
        return 0
    fi

    log "下载 mini camera (32 分片, ~90GB, 并行 $PAR，无 lidar)"
    seq 0 31 | xargs -I{} -P "$PAR" \
        env AUTOVLA_DATA_ROOT="$ROOT" HF="$HF" bash "$SELF" _one_mini {}

    local n
    n=$(find sensor_blobs -maxdepth 1 -regextype posix-extended \
            -regex '.*/\.done_mini_camera_[0-9]+' | wc -l)
    if [ "$n" -ne 32 ]; then
        log "⚠️  mini camera 只完成 $n/32 个分片。重跑本步骤会自动续传缺失分片。"
        return 1
    fi

    if [ -d openscene-v1.1/sensor_blobs ]; then
        link_sensors mini
    fi
    touch sensor_blobs/.done_mini_camera
    log "sensor_blobs/mini 完成 ($(ls sensor_blobs/mini | wc -l) 个 log)"
}

# ---------------------------------------------------------------------------
# 2b. metadata → navsim_logs/{trainval,test}
# ---------------------------------------------------------------------------
step_one_logs() {
    local split="$1"
    if have "navsim_logs/$split"; then log "navsim_logs/$split 已存在，跳过"; return 0; fi
    log "下载 $split metadata"
    fetch_split ".done_meta_$split" "$HF/openscene_metadata_${split}.tgz" "meta_${split}.tgz" || return 1
    install_meta "$split"
    log "navsim_logs/$split 完成 ($(ls "navsim_logs/$split" | wc -l) 个 log)"
}

step_logs() {
    mkdir -p navsim_logs
    step_one_logs trainval || return 1
    step_one_logs test || return 1
}

step_trainval_logs() {
    mkdir -p navsim_logs
    step_one_logs trainval
}

# ---------------------------------------------------------------------------
# 3. navtest camera → sensor_blobs/test  (32 分片, 128GB)
#    标记名 .done_test_camera_* 与本机已完成的下载一致
# ---------------------------------------------------------------------------
_one_test() {
    fetch_split "sensor_blobs/.done_test_camera_$1" \
                "$HF/openscene_sensor_test_camera/openscene_sensor_test_camera_$1.tgz" \
                "_test_$1.tgz"
}

step_test() {
    mkdir -p sensor_blobs
    if have sensor_blobs/test && have sensor_blobs/.done_test_camera; then
        log "sensor_blobs/test 已完成，跳过"
        return 0
    fi

    log "下载 navtest camera (32 分片, 128GB, 并行 $PAR)"
    seq 0 31 | xargs -I{} -P "$PAR" \
        env AUTOVLA_DATA_ROOT="$ROOT" HF="$HF" bash "$SELF" _one_test {}

    local n
    n=$(find sensor_blobs -maxdepth 1 -regextype posix-extended \
            -regex '.*/\.done_test_camera_[0-9]+' | wc -l)
    if [ "$n" -ne 32 ]; then
        log "⚠️  只完成 $n/32 个分片。重跑本步骤会自动续传缺失分片。"
        return 1
    fi

    if [ -d openscene-v1.1/sensor_blobs ]; then
        link_sensors test
    fi
    touch sensor_blobs/.done_test_camera
    log "sensor_blobs/test 完成 ($(ls sensor_blobs/test | wc -l) 个 log)"
}

# ---------------------------------------------------------------------------
# 4. navtrain sensors → sensor_blobs/trainval  (8 分片, 445GB)
#    ⚠️ history 不能省：navtrain.yaml 是 num_history_frames=4
# ---------------------------------------------------------------------------
step_navtrain() {
    mkdir -p sensor_blobs/trainval
    for kind in current history; do
        for i in 1 2 3 4; do
            local marker="sensor_blobs/.done_navtrain_${kind}_${i}"
            have "$marker" && { echo "  · navtrain_${kind}_${i} 已完成"; continue; }

            log "navtrain_${kind}_${i}.tgz  (共 8 份 / 445GB)"
            wget -c "$S3/navtrain_${kind}_${i}.tgz" || { echo "✗ 下载失败"; return 1; }
            gzip -t "navtrain_${kind}_${i}.tgz" 2>/dev/null \
                || { echo "✗ 校验失败(截断)，已删除，重跑可续"; rm -f "navtrain_${kind}_${i}.tgz"; return 1; }
            tar -xzf "navtrain_${kind}_${i}.tgz" || { echo "✗ 解压失败"; return 1; }

            local src="${kind}_split_${i}"
            [ -d "$src/trainval" ] && src="$src/trainval"
            echo "  [结构] $(ls "$src" | head -2 | tr '\n' ' ')"
            cp -rlf "$src"/* sensor_blobs/trainval/     # 硬链接，不复制数据
            rm -rf "${kind}_split_${i}" "navtrain_${kind}_${i}.tgz"
            touch "$marker"
        done
    done
    log "navtrain 完成 ($(ls sensor_blobs/trainval | wc -l) 个 log)"
}

# ---------------------------------------------------------------------------
# 5. 完整性核对：scene_filter 要的 log 是否都有 sensor
# ---------------------------------------------------------------------------
step_verify() {
    log "完整性核对"
    python3 - "$ROOT" "$REPO" <<'PY'
import os, sys, yaml
root, repo = sys.argv[1], sys.argv[2]
fdir = os.path.join(repo, "navsim/navsim/planning/script/config/common/train_test_split/scene_filter")
for name, split in (("navtest", "test"), ("navtrain", "trainval")):
    sd = os.path.join(root, "sensor_blobs", split)
    if not os.path.isdir(sd):
        print(f"{name:9s} sensor_blobs/{split} 不存在，跳过"); continue
    need = set(yaml.safe_load(open(os.path.join(fdir, f"{name}.yaml")))["log_names"])
    have = set(os.listdir(sd))
    miss = need - have
    flag = "✅" if not miss else "❌"
    print(f"{flag} {name:9s} 需要 {len(need):5d} log, 实有 {len(have & need):5d}, 缺 {len(miss)}")
    for m in sorted(miss)[:5]:
        print(f"      缺: {m}")
PY
}

main() {
    log "目标目录: $ROOT"; df -h "$ROOT" | tail -1
    case "${1:-all}" in
        maps)          step_maps ;;
        mini)          step_mini ;;
        logs)          step_logs ;;
        trainval_logs) step_trainval_logs ;;
        navtrain)      step_navtrain ;;
        test)          step_test ;;
        verify)        step_verify ;;
        # maps + mini + navtrain（含 trainval logs），不下 test
        mini_navtrain) step_maps && step_mini && step_trainval_logs && step_navtrain ;;
        # trainval sensors + test camera + 两边 metadata
        trainval_test) step_logs && step_navtrain && step_test ;;
        _one_mini)     _one_mini "$2" ;;
        _one_test)     _one_test "$2" ;;
        all)           step_maps && step_logs && step_test && step_navtrain && step_verify ;;
        *)
            echo "未知步骤: $1"
            echo "可选: maps | mini | logs | trainval_logs | navtrain | test | verify | mini_navtrain | trainval_test | all"
            exit 1
            ;;
    esac
}

main "$@"
exit $?
