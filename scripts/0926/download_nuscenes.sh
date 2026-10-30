#!/bin/bash
# ============================================================================
# nuScenes v1.0-trainval 下载（RIT tigris）—— nuScenes SFT / mix SFT / L2+Collision 评测用
#
# 下什么（按 tools/preprocessing/nusc_sample_generation.py 实际读的文件裁剪）:
#   * v1.0-trainval_meta.tgz                 0.5G   metadata 表 + maps/       (需登录)
#   * v1.0-trainval{01..10}_blobs_camera.tgz 18-27G/片  只含相机 (samples+sweeps) (需登录)
#       预处理只读 sample['data'][CAM_*] 的关键帧图 + LIDAR_TOP 的 ego_pose(在 metadata 里),
#       不读 sweeps → 默认只解出 samples/CAM_*。FULL=1 整包解压。
#       BLOB_KIND=all 改下全传感器的 v1.0-trainval{01..10}_blobs.tgz (~30G/片)。
#   * DriveLM v1_1_train_nus.json     193M   CoT 标注 (Google Drive 镜像, 免登录)
#   * nusc_eval_seg_6s.zip            34M → 解压 13.5G, UniAD 分割, nusc_eval.py --seg_data_path
#                                            (README 里的 Google Drive 链接, 免登录)
#
# 登录: 需 nuScenes 账号 (https://www.nuscenes.org 注册并同意 Terms of Use)。
#   token → archives API 换下载地址 (与网页下载同一接口)。token 两种来源:
#     * NUSCENES_TOKEN=<浏览器开发者工具里 archives 请求的 Bearer token> (1 小时有效)
#     * NUSCENES_EMAIL / NUSCENES_PASSWORD (没设就交互输入, 不回显), 脚本自己换 token
#   换到的是不带签名的 CloudFront 地址 (不过期), 缓存在 $STAGE/.url_*,
#   之后重跑/续传不再需要登录。
#
# 产出:
#   $ROOT/
#   ├── v1.0-trainval/          metadata json
#   ├── maps/
#   ├── samples/CAM_*/          6 × 34,149 张 (train 28,130 + val 6,019)
#   ├── drivelm/v1_1_train_nus.json
#   └── nusc_eval_seg/          6,019 个 .pt
#   之后预处理: bash scripts/run_nuscenes_preprocessing.sh --nuscenes_path $ROOT \
#       --output_dir $ROOT/nuscenes --drivelm_path $ROOT/drivelm/v1_1_train_nus.json
#
# 用法:
#   tmux new -s nusc
#   bash scripts/0926/download_nuscenes.sh               # 全部
#   bash scripts/0926/download_nuscenes.sh drivelm seg   # 只下免登录的两份
#   bash scripts/0926/download_nuscenes.sh status        # 看进度 (正在下的 tgz 大小)
#   bash scripts/0926/download_nuscenes.sh verify        # 核对
#   步骤名: meta | blobs | drivelm | seg | verify | status | urls (只换下载地址) | all (可多个)
#   PAR=4 (并行分片数)   FULL=1 (blobs 整包解压)   BLOB_KIND=camera|all   NUSC_REGION=us|asia
#
# robust 机制同 download_trainval_full_rit.sh:
#   wget -c 落盘 → gzip -t 校验 → tar 解压 → .done 标记 → 删 tgz。全幂等, 中断重跑续传。
#   ⚠️ 不用 `wget -qO- | tar -xz` 流式管道 (截断会静默丢分片)
# ============================================================================
set -uo pipefail

SELF="$(readlink -f "$0")"
ROOT="${NUSC_ROOT:-/shared/rc/llm-gen-agent/mhu/auto_vla/nuscenes}"
PAR="${PAR:-4}"
FULL="${FULL:-0}"
BLOB_KIND="${BLOB_KIND:-camera}"
MIN_FREE_GB="${MIN_FREE_GB:-80}"   # 每个 blob ≤30G 压缩, 低于此值停手
export NUSC_REGION="${NUSC_REGION:-us}"

STAGE="$ROOT/.stage"               # tgz 落盘区, 与产出同盘 (解完即删)
MARK="$ROOT/.markers"
BLOBS=(01 02 03 04 05 06 07 08 09 10)
CAMS=(CAM_FRONT CAM_FRONT_RIGHT CAM_FRONT_LEFT CAM_BACK CAM_BACK_RIGHT CAM_BACK_LEFT)
EXPECT_PER_CAM=34149               # trainval 关键帧数, 每路相机一张
EXPECT_SEG=6019                    # val 关键帧数

# HF 上的 OpenDriveLab/DriveLM 是 gated repo (未授权 401), 用 DriveLM challenge README 里的
# Google Drive 镜像 (同名同大小 192,961,829 B)
DRIVELM_GDRIVE_ID="1CvTPwChKvfnvrZ1Wr0ZNVqtibkkNeGgt"
SEG_GDRIVE_ID="1YMiKqUi_rhdxdrNVXKFuwEg3xAGm-dvU"   # README "nuScenes Evaluation" 的链接
gdrive_url() { echo "https://drive.usercontent.google.com/download?id=$1&export=download&confirm=t"; }

mkdir -p "$ROOT" "$STAGE" "$MARK" || exit 1
cd "$ROOT" || exit 1

log()  { echo -e "\n\033[1;32m[$(date +%H:%M:%S)] $*\033[0m"; }
warn() { echo -e "\033[1;33m[$(date +%H:%M:%S)] $*\033[0m" >&2; }
err()  { echo -e "\033[1;31m[$(date +%H:%M:%S)] $*\033[0m" >&2; }
have() { [ -e "$1" ]; }
free_gb() { df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9'; }

# 解压/校验加速: pigz 多核, 回退到 gzip
if command -v pigz >/dev/null 2>&1; then
    TARZ=(tar -I pigz -xf); GZTEST=(pigz -t)
else
    TARZ=(tar -xzf);        GZTEST=(gzip -t)
fi

# ---------------------------------------------------------------------------
# 登录 & 下载地址
# ---------------------------------------------------------------------------
# $1 = 归档文件名 → stdout 输出下载地址。
#   有 NUSCENES_TOKEN 直接用; 否则用 NUSCENES_EMAIL / NUSCENES_PASSWORD 走 Cognito 换 token
presign() {
    python3 - "$1" <<'PY'
import json, os, sys, urllib.request, urllib.error

def call(req):
    try:
        return json.load(urllib.request.urlopen(req, timeout=60))
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}")

token = os.environ.get("NUSCENES_TOKEN", "").removeprefix("Bearer ").strip()
if not token:
    auth = call(urllib.request.Request(
        "https://cognito-idp.us-east-1.amazonaws.com/",
        data=json.dumps({
            "AuthFlow": "USER_PASSWORD_AUTH",
            "ClientId": "7fq5jvs5ffs1c50hd3toobb3b9",
            "AuthParameters": {"USERNAME": os.environ["NUSCENES_EMAIL"],
                               "PASSWORD": os.environ["NUSCENES_PASSWORD"]},
            "ClientMetadata": {}}).encode(),
        headers={"Content-Type": "application/x-amz-json-1.1",
                 "X-Amz-Target": "AWSCognitoIdentityProviderService.InitiateAuth"}))
    if "AuthenticationResult" not in auth:
        sys.exit(f"登录需要额外验证 ({auth.get('ChallengeName')}), 先去官网网页登录处理")
    token = auth["AuthenticationResult"]["IdToken"]

url = ("https://o9k5xn5546.execute-api.us-east-1.amazonaws.com/v1/archives/v1.0/"
       f"{sys.argv[1]}?region={os.environ.get('NUSC_REGION', 'us')}&project=nuScenes")
print(call(urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"}))["url"])
PY
}

need_login() {
    [ -n "${NUSCENES_TOKEN:-}" ] && { export NUSCENES_TOKEN; return 0; }
    if [ -z "${NUSCENES_EMAIL:-}" ]; then
        [ -t 0 ] || { err "未设置 NUSCENES_TOKEN / NUSCENES_EMAIL (非交互环境无法输入)"; exit 1; }
        read -rp "nuScenes 账号邮箱: " NUSCENES_EMAIL
    fi
    if [ -z "${NUSCENES_PASSWORD:-}" ]; then
        [ -t 0 ] || { err "未设置 NUSCENES_PASSWORD (非交互环境无法输入)"; exit 1; }
        read -rsp "nuScenes 密码: " NUSCENES_PASSWORD; echo
    fi
    export NUSCENES_EMAIL NUSCENES_PASSWORD
}

blob_name() {
    if [ "$BLOB_KIND" = "all" ]; then echo "v1.0-trainval${1}_blobs.tgz"
    else echo "v1.0-trainval${1}_blobs_camera.tgz"; fi
}

# 串行把缺的下载地址都换好并缓存 —— token 只在这一步用到, 之后的并行下载/续传都不需要
resolve_urls() {
    local name missing=()
    for name in "$@"; do [ -s "$STAGE/.url_$name" ] || missing+=("$name"); done
    [ ${#missing[@]} -eq 0 ] && return 0
    need_login
    for name in "${missing[@]}"; do
        if presign "$name" > "$STAGE/.url_$name.tmp"; then
            mv "$STAGE/.url_$name.tmp" "$STAGE/.url_$name"
        else
            rm -f "$STAGE/.url_$name.tmp"
            err "[$name] 申请下载地址失败: token 过期 / 账号密码错 / 未在 nuscenes.org 同意 Terms of Use"
            return 1
        fi
    done
    log "下载地址就绪 (新申请 ${#missing[@]} 个)"
}

resolve_all() {
    local i names=()
    have "$MARK/.done_meta" || names+=(v1.0-trainval_meta.tgz)
    for i in "${BLOBS[@]}"; do blob_done "$i" || names+=("$(blob_name "$i")"); done
    resolve_urls "${names[@]}"
}

# 下载一个需登录的 tgz 到 $STAGE (地址来自 resolve_urls 的缓存; wget -c 续传)
fetch_nusc() {
    local name="$1" url try
    url=$(cat "$STAGE/.url_$name" 2>/dev/null) || { err "[$name] 没有缓存的下载地址"; return 1; }
    for try in 1 2 3; do
        wget -c -q --tries=20 -O "$STAGE/$name" "$url" && return 0
        warn "[$name] 下载中断 (第 $try 次), 30s 后续传"
        sleep 30
    done
    return 1
}

# ---------------------------------------------------------------------------
# 1. metadata (~0.4G)
# ---------------------------------------------------------------------------
step_meta() {
    local tgz="$STAGE/v1.0-trainval_meta.tgz"
    if have "$MARK/.done_meta"; then log "metadata 已完成, 跳过"; return 0; fi
    resolve_urls v1.0-trainval_meta.tgz || return 1
    log "下载 v1.0-trainval_meta.tgz"
    fetch_nusc v1.0-trainval_meta.tgz || { err "metadata 下载失败"; return 1; }
    "${GZTEST[@]}" "$tgz" 2>/dev/null \
        || { err "metadata 校验失败 (截断), 已删除, 请重跑"; rm -f "$tgz"; return 1; }
    "${TARZ[@]}" "$tgz" -C "$ROOT" || { err "metadata 解压失败"; return 1; }
    rm -f "$tgz"
    touch "$MARK/.done_meta"
    log "metadata 完成 ($(ls "$ROOT"/v1.0-trainval/*.json 2>/dev/null | wc -l) 张表)"
}

# ---------------------------------------------------------------------------
# 2. blobs —— 10 分片 (camera: 18-27G/片; all: ~30G/片)
#    tar 内结构: samples/<SENSOR>/*, sweeps/<SENSOR>/*
#    默认只解 samples/CAM_* (--no-anchored 兼容带 ./ 前缀的成员名)。
#    标记分 _cam / _full 两种: 先下 cam 再改 FULL=1 会重下补全。
# ---------------------------------------------------------------------------
blob_done() {
    have "$MARK/.done_blob_$1_full" && return 0
    [ "$FULL" != "1" ] && have "$MARK/.done_blob_$1_cam"
}

one_blob() {
    local i="$1" name; name=$(blob_name "$1")
    local tgz="$STAGE/$name" mode
    mode=$([ "$FULL" = "1" ] && echo full || echo cam)

    blob_done "$i" && { echo "  · [$i] 已完成"; return 0; }

    local avail; avail=$(free_gb)
    if [ "${avail:-0}" -lt "$MIN_FREE_GB" ]; then
        err "[$i] 跳过: 剩余空间 ${avail}G < ${MIN_FREE_GB}G 阈值"
        return 2
    fi

    echo "  -> [$i] 下载"
    fetch_nusc "$name" || { err "[$i] 下载失败 (重跑可续传)"; return 1; }

    echo "  -> [$i] 校验 ($(du -h "$tgz" | cut -f1))"
    if ! "${GZTEST[@]}" "$tgz" 2>/dev/null; then
        err "[$i] 校验失败 (下载被截断) → 删除, 重跑重下"
        rm -f "$tgz"
        return 1
    fi

    echo "  -> [$i] 解压 ($mode)"
    if [ "$mode" = "full" ]; then
        "${TARZ[@]}" "$tgz" -C "$ROOT" || { err "[$i] 解压失败"; return 1; }
    else
        "${TARZ[@]}" "$tgz" -C "$ROOT" --wildcards --no-anchored 'samples/CAM_*' \
            || { err "[$i] 解压失败"; return 1; }
    fi

    rm -f "$tgz"
    touch "$MARK/.done_blob_${i}_${mode}"
    echo "  -> [$i] ✅ 完成 (剩余 $(free_gb)G)"
}

step_blobs() {
    local i n=0 todo=()
    for i in "${BLOBS[@]}"; do
        if blob_done "$i"; then n=$((n + 1)); else todo+=("$(blob_name "$i")"); fi
    done
    log "blobs ($BLOB_KIND): ${#BLOBS[@]} 分片, 已完成 $n, 并行 $PAR, 解压 $([ "$FULL" = "1" ] && echo '整包' || echo '只留 samples/CAM_*')"
    log "剩余空间 $(free_gb)G"
    resolve_urls "${todo[@]}" || return 1

    printf "%s\n" "${BLOBS[@]}" | xargs -I{} -P "$PAR" bash "$SELF" _blob {}

    n=0
    for i in "${BLOBS[@]}"; do blob_done "$i" && n=$((n + 1)); done
    log "blobs 完成 $n / ${#BLOBS[@]}"
    if [ "$n" -lt "${#BLOBS[@]}" ]; then
        warn "有 $((${#BLOBS[@]} - n)) 个分片未完成 —— 重跑本脚本续传"
        return 1
    fi
}

# ---------------------------------------------------------------------------
# 3. DriveLM CoT 标注 (193M, 免登录)
# ---------------------------------------------------------------------------
step_drivelm() {
    local out="$ROOT/drivelm/v1_1_train_nus.json"
    if have "$MARK/.done_drivelm"; then log "DriveLM 已完成, 跳过"; return 0; fi
    log "下载 DriveLM v1_1_train_nus.json (193M, Google Drive)"
    mkdir -p "$ROOT/drivelm"
    rm -f "$out"
    curl -fsSL --retry 3 -o "$out" "$(gdrive_url "$DRIVELM_GDRIVE_ID")" \
        || { err "DriveLM 下载失败"; rm -f "$out"; return 1; }
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(f'  DriveLM: {len(d)} 个 scene')" "$out" \
        || { err "DriveLM json 解析失败 (下载不完整), 已删除, 请重跑"; rm -f "$out"; return 1; }
    touch "$MARK/.done_drivelm"
}

# ---------------------------------------------------------------------------
# 4. UniAD 分割评测数据 (34M zip → 13.5G, 免登录)
#    zip 内: nusc_eval_seg_6s/<sample_token>.pt → 落到 $ROOT/nusc_eval_seg/
# ---------------------------------------------------------------------------
step_seg() {
    local zip="$STAGE/nusc_eval_seg_6s.zip"
    if have "$MARK/.done_seg"; then log "nusc_eval_seg 已完成, 跳过"; return 0; fi
    log "下载 UniAD 分割评测数据 nusc_eval_seg_6s.zip (34M, Google Drive)"
    rm -f "$zip"
    curl -fsSL --retry 3 -o "$zip" "$(gdrive_url "$SEG_GDRIVE_ID")" \
        || { err "分割数据下载失败"; return 1; }
    unzip -tq "$zip" >/dev/null || { err "zip 校验失败, 已删除, 请重跑"; rm -f "$zip"; return 1; }
    rm -rf "$STAGE/nusc_eval_seg_6s" "$ROOT/nusc_eval_seg"
    unzip -q "$zip" -d "$STAGE" || { err "分割数据解压失败"; return 1; }
    mv "$STAGE/nusc_eval_seg_6s" "$ROOT/nusc_eval_seg"
    rm -f "$zip"
    touch "$MARK/.done_seg"
    log "nusc_eval_seg 完成 ($(find "$ROOT/nusc_eval_seg" -name '*.pt' | wc -l) 个 .pt)"
}

# ---------------------------------------------------------------------------
# 5. status / verify
# ---------------------------------------------------------------------------
step_status() {
    log "status"
    printf "  完成标记: %s\n" "$(ls "$MARK" 2>/dev/null | tr '\n' ' ')"
    printf "  下载中:\n"
    ls -lh "$STAGE" 2>/dev/null | awk 'NR>1 {printf "    %-32s %s\n", $9, $5}'
    printf "  剩余空间: %sG\n" "$(free_gb)"
}

step_verify() {
    log "verify"
    local ok=1 c n
    n=$(ls "$ROOT"/v1.0-trainval/*.json 2>/dev/null | wc -l)
    printf "  metadata 表          : %s\n" "$n"
    [ -f "$ROOT/v1.0-trainval/sample.json" ] || ok=0

    for c in "${CAMS[@]}"; do
        n=$(find "$ROOT/samples/$c" -maxdepth 1 -name '*.jpg' 2>/dev/null | wc -l)
        printf "  samples/%-16s: %6s / %s\n" "$c" "$n" "$EXPECT_PER_CAM"
        [ "$n" -eq "$EXPECT_PER_CAM" ] || ok=0
    done

    if [ -f "$ROOT/drivelm/v1_1_train_nus.json" ] && have "$MARK/.done_drivelm"; then
        printf "  DriveLM json         : ✅\n"
    else
        printf "  DriveLM json         : ❌\n"; ok=0
    fi

    n=$(find "$ROOT/nusc_eval_seg" -maxdepth 1 -name '*.pt' 2>/dev/null | wc -l)
    printf "  nusc_eval_seg .pt    : %6s / %s\n" "$n" "$EXPECT_SEG"
    [ "$n" -eq "$EXPECT_SEG" ] || ok=0

    if [ "$ok" = "1" ]; then
        log "✅ 齐了。下一步预处理 (单独 env: conda env create -f environment_nusc_preprocess.yml)"
        return 0
    fi
    err "❌ 未齐 —— 重跑 bash scripts/0926/download_nuscenes.sh 续传"
    return 1
}

# ---------------------------------------------------------------------------
# 内部: xargs 并行时单分片入口
if [ "${1:-}" = "_blob" ]; then
    one_blob "$2"
    exit $?
fi

main() {
    log "目标目录: $ROOT"
    df -h "$ROOT" | tail -1

    [ $# -eq 0 ] && set -- all
    local s
    for s in "$@"; do
        case "$s" in
            meta)    step_meta ;;
            blobs)   step_blobs ;;
            drivelm) step_drivelm ;;
            seg)     step_seg ;;
            verify)  step_verify ;;
            status)  step_status ;;
            all)     resolve_all || exit 1   # 先把下载地址都换好, 之后可以放着不管
                     step_drivelm; step_seg
                     step_meta && step_blobs
                     step_verify ;;
            urls)    resolve_all ;;
            *)       echo "未知步骤: $s  (meta|blobs|drivelm|seg|verify|status|urls|all)"; exit 1 ;;
        esac
    done
}

main "$@"
