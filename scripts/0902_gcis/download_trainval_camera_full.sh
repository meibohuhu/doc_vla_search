#!/bin/bash
# ============================================================================
# [mh 2026/09/02] 下载【全量 trainval camera】(200 分片) —— 166.3k 数据集的 Step 1
#
# 说明见 docs/0901/nuplan_103k_to_166k.md。本脚本是本机版 (/scratch/mh2803/vla)。
#
# 为什么要重下：现有 sensor_blobs/trainval 的 446G 只覆盖 navtrain 白名单的 1192 个 log；
# 166.3k 口径要铺满全部 1310 个 log。HF 的分片是任意打包的（不按 log 切），
# 所以没法只补差量，只能把 200 个分片全下一遍（已有的帧会被同路径覆盖，不会翻倍占盘）。
#
# 🔴 与上游 navsim/download/download_trainval.sh 的关键区别：
#   那个脚本用 `wget -qO- | tar -xz` 流式管道，下载被截断时 gzip 报错但
#   **管道退出码被忽略 -> 分片静默丢失，脚本却报“完成”**（本项目已中招 3 次）。
#   本脚本：落盘 -> `gzip -t` 完整性校验 -> 通过才解包 -> 打 .done 标记。
#   任何一步失败都不打标记，重跑会自动补。
#
# 落位：tgz 内部是 openscene-v1.1/sensor_blobs/trainval/<log>/<CAM>/*.jpg，
#   用 --strip-components=3 直接解到 sensor_blobs/trainval/，
#   省掉“先解到 openscene-v1.1/ 再 mv”的 1.3TB 搬运。
#
# 用法:
#   bash scripts/0902/download_trainval_camera_full.sh            # 下载（可断点续传）
#   bash scripts/0902/download_trainval_camera_full.sh verify     # ★ 只对账，以它为准
#   PAR=2 bash ...            # 降并发（默认 4；实测链路 105MB/s 封顶，4 和 8 一样快）
#   SHARDS="0 1 2" bash ...   # 只跑指定分片
# ============================================================================
set -uo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
ROOT="${AUTOVLA_DATA_ROOT:-/scratch/mh2803/vla/nuplan}"
HF="${HF:-https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1}"
# 实测（2026/09/02 本机）：单流 47.9 MB/s，4 流 104 MB/s，8 流 105.7 MB/s -> 链路 ~105MB/s 封顶。
# 所以 4 就够，再高只是多占中转磁盘和 CPU（tar 解压是 CPU 密集的）。
PAR="${PAR:-4}"
NSHARD=200
MODE="${1:-download}"

mkdir -p "$ROOT" && cd "$ROOT" || exit 1
log() { echo -e "\n\033[1;32m[$(date +%H:%M:%S)] $*\033[0m"; }
have() { [ -e "$1" ]; }

# ---------------------------------------------------------------- 单分片
# 作为子命令被 xargs 并行调用
_one() {
    local i="$1"
    local marker="sensor_blobs/.done_trainval_camera_${i}"
    local tgz="openscene_sensor_trainval_camera_${i}.tgz"
    local url="$HF/openscene_sensor_trainval_camera/openscene_sensor_trainval_camera_${i}.tgz"
    mkdir -p sensor_blobs/trainval

    have "$marker" && { echo "  [$i] 已完成，跳过"; return 0; }

    # 清掉上次失败留下的 0 字节文件，避免 wget -c 卡死
    [ -f "$tgz" ] && [ ! -s "$tgz" ] && rm -f "$tgz"

    local attempt=1
    while [ "$attempt" -le 3 ]; do
        echo "  [$i] 下载 (attempt $attempt/3)"
        if wget -c -q --show-progress --progress=dot:giga "$url" -O "$tgz" && [ -s "$tgz" ]; then
            # ★ 关键：校验 gzip 完整性。截断/损坏在这里就被拦下，不会静默进入解包。
            echo "  [$i] 校验 gzip 完整性"
            if gzip -t "$tgz" 2>/dev/null; then
                break
            fi
            echo "  [$i] ❌ gzip 校验失败（八成是下载被截断），删掉重下"
        fi
        rm -f "$tgz"
        attempt=$((attempt + 1))
        sleep $((attempt * 5))
    done

    if [ ! -s "$tgz" ] || ! gzip -t "$tgz" 2>/dev/null; then
        echo "  [$i] ❌ 三次都失败，跳过（重跑本脚本会再试）"
        rm -f "$tgz"
        return 1
    fi

    echo "  [$i] 解压 -> sensor_blobs/trainval/"
    # strip 3 层: openscene-v1.1/sensor_blobs/trainval/  -> 直接得到 <log>/<CAM>/*.jpg
    if ! tar -xzf "$tgz" -C sensor_blobs/trainval --strip-components=3; then
        echo "  [$i] ❌ 解压失败"; rm -f "$tgz"; return 1
    fi
    rm -f "$tgz"
    touch "$marker"
    echo "  [$i] ✅ 完成"
}

# 子命令入口（供 xargs 调用）
if [ "$MODE" = "_one" ]; then _one "$2"; exit $?; fi

# ---------------------------------------------------------------- 对账
verify() {
    echo "==================== 对账 ===================="
    local done_n=0 missing=""
    for i in $(seq 0 $((NSHARD-1))); do
        if have "sensor_blobs/.done_trainval_camera_${i}"; then
            done_n=$((done_n+1))
        else
            missing="$missing $i"
        fi
    done
    echo "已完成分片 : $done_n / $NSHARD"
    if [ -n "$missing" ]; then
        echo "❌ 缺失分片 :$missing"
        echo "   -> 直接重跑本脚本，会只补这些（已完成的跳过）"
    else
        echo "✅ 200 个分片全部完成"
    fi
    echo "残留 .tgz  : $(ls openscene_sensor_trainval_camera_*.tgz 2>/dev/null | wc -l) (应为 0)"
    echo "sensor_blobs/trainval 占用: $(du -sh sensor_blobs/trainval 2>/dev/null | cut -f1)"
    echo "log 目录数 : $(ls sensor_blobs/trainval 2>/dev/null | wc -l)  (全量应为 1310)"
    echo "磁盘剩余   : $(df -h "$ROOT" | tail -1 | awk '{print $4}')"
    [ -z "$missing" ]
}

if [ "$MODE" = "verify" ]; then verify; exit $?; fi

# ---------------------------------------------------------------- 前置检查
echo "前置检查:"
avail=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
echo "  磁盘可用 : ${avail}G"
echo "  需求     : 净新增约 860G（全量 1.31T - 已有 446G）+ 中转 tgz 峰值约 $((PAR*11))G"
if [ "${avail:-0}" -lt 1000 ]; then
    echo "  ⚠️ 可用空间 < 1000G，可能不够。继续请按 Ctrl-C 以外的任意键，10 秒后自动继续"
    sleep 10
fi
done_before=$(ls sensor_blobs/.done_trainval_camera_* 2>/dev/null | wc -l)
echo "  已完成   : $done_before / $NSHARD 个分片"
echo "  并发     : $PAR （实测链路 ~105MB/s 封顶）"

LIST="${SHARDS:-$(seq 0 $((NSHARD-1)))}"
log "开始下载全量 trainval camera（$NSHARD 分片，约 1.31TB，落盘+gzip校验+断点续传）"
start=$(date +%s)

printf "%s\n" $LIST | xargs -P "$PAR" -I{} \
    env AUTOVLA_DATA_ROOT="$ROOT" HF="$HF" bash "$SCRIPT_PATH" _one {}
rc=$?

end=$(date +%s); el=$((end-start))
done_after=$(ls sensor_blobs/.done_trainval_camera_* 2>/dev/null | wc -l)
made=$((done_after - done_before))

echo
echo "=================================================================="
echo " 耗时     : $((el/3600))h $(( (el%3600)/60 ))m"
echo " 本次完成 : $made 个分片 (总计 $done_after / $NSHARD)"
if [ "$made" -gt 0 ] && [ "$el" -gt 0 ]; then
    rest=$((NSHARD - done_after))
    python3 -c "
el,made,rest=$el,$made,$rest
per=el/made
print(f' 平均     : {per/60:.1f} 分钟/分片  (约 {6.7*1024/per:.0f} MB/s 有效吞吐)')
print(f' 剩余     : {rest} 个 -> 约 {rest*per/3600:.1f} 小时') if rest>0 else print(' ✅ 全部完成')
" 2>/dev/null
fi
echo "=================================================================="
echo
verify