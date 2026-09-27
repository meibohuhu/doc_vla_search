#!/bin/bash
# ============================================================================
# AutoVLA nuPlan / NAVSIM 数据下载（camera only，无 lidar）
#
# 产出目录结构（AutoVLA 的 placeholder 替换机制要求的两级结构）：
#   $ROOT/
#   ├── maps/                    971MB
#   ├── navsim_logs/{trainval,test}/*.pkl      1310 / 147 个 log
#   └── sensor_blobs/
#       ├── trainval/            445GB (navtrain current+history)
#       └── test/                128GB (navtest camera)
#
# 设计要点（v2，吸取教训）
#   * 每个分片：wget -c 落盘 -> gzip -t 校验 -> 解压 -> 打 .done 标记 -> 删 tgz
#     v1 用 `wget -qO- | tar -xz` 流式管道，下载被截断时 gzip 报错但管道退出码
#     被忽略 -> 分片静默丢失（实际丢了 20 个 navtest log 才发现）。
#   * 所有步骤幂等；中断后重跑自动续传，已完成的分片直接跳过。
#   * tar 解出的目录里本身带 <split>/ 子层，直接 mv 会多套一层，已处理。
#
# 用法:
#   bash download_nuplan_autovla.sh              # 全部
#   bash download_nuplan_autovla.sh test         # 单步: maps|logs|navtrain|test
#   bash download_nuplan_autovla.sh verify       # 只做完整性核对，不下载
# ============================================================================
set -uo pipefail

SELF="$(readlink -f "$0")"
REPO="$(cd "$(dirname "$SELF")/.." && pwd)"
ROOT="${AUTOVLA_DATA_ROOT:-/data/autovla_data/nuplan}"
HF="https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1"
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
# 2. metadata → navsim_logs/{trainval,test}
# ---------------------------------------------------------------------------
step_logs() {
    mkdir -p navsim_logs
    for split in trainval test; do
        if have "navsim_logs/$split"; then log "navsim_logs/$split 已存在，跳过"; continue; fi
        log "下载 $split metadata"
        fetch_split ".done_meta_$split" "$HF/openscene_metadata_${split}.tgz" "meta_${split}.tgz" || return 1
        # ⚠️ meta_datas/ 里本身带 <split>/ 子目录，直接 mv 会多套一层
        if [ -d "openscene-v1.1/meta_datas/$split" ]; then
            mv "openscene-v1.1/meta_datas/$split" "navsim_logs/$split"
        else
            mv openscene-v1.1/meta_datas "navsim_logs/$split"
        fi
        rm -rf openscene-v1.1
        log "navsim_logs/$split 完成 ($(ls "navsim_logs/$split" | wc -l) 个 log)"
    done
}

# ---------------------------------------------------------------------------
# 3. navtest camera → sensor_blobs/test  (32 分片, 128GB)
# ---------------------------------------------------------------------------
_one_test() {
    fetch_split "sensor_blobs/.done_test_$1" \
                "$HF/openscene_sensor_test_camera/openscene_sensor_test_camera_$1.tgz" \
                "_test_$1.tgz"
}

step_test() {
    mkdir -p sensor_blobs
    log "下载 navtest camera (32 分片, 128GB, 并行 $PAR)"
    seq 0 31 | xargs -I{} -P "$PAR" bash "$SELF" _one_test {}

    local n; n=$(ls sensor_blobs/.done_test_* 2>/dev/null | wc -l)
    if [ "$n" -ne 32 ]; then
        log "⚠️  只完成 $n/32 个分片。重跑本步骤会自动续传缺失分片。"
        return 1
    fi

    if [ -d openscene-v1.1/sensor_blobs ]; then
        local src=openscene-v1.1/sensor_blobs
        [ -d "$src/test" ] && src="$src/test"
        mkdir -p sensor_blobs/test
        cp -rlf "$src"/* sensor_blobs/test/     # 同盘硬链接，秒级；-f 覆盖已有
        rm -rf openscene-v1.1
    fi
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
        maps) step_maps ;; logs) step_logs ;;
        test) step_test ;; navtrain) step_navtrain ;;
        verify) step_verify ;;
        _one_test) _one_test "$2" ;;          # 内部：xargs 单分片入口
        all)  step_maps && step_logs && step_test && step_navtrain && step_verify ;;
        *) echo "未知步骤: $1 (maps|logs|test|navtrain|verify|all)"; exit 1 ;;
    esac
}
main "$@"
