#!/bin/bash
# ============================================================================
# AutoVLA 环境安装 —— linux-aarch64 (GH200 / Grace-Hopper, neoverse_v2)
#
# 为什么要单独一份: README 的 environment.yml + requirements.txt 是 x86_64 的。
#   本机(tigris.rc.rit.edu 及其全部计算节点)是 aarch64，差异见
#   docs/0901/aarch64_gh200_env.md
#
# 用法:
#   bash scripts/install_aarch64.sh            # 全部
#   bash scripts/install_aarch64.sh torch      # 只跑某一步
#   步骤名: conda | torch | pip | editable | verify
#
# 前置: 一个 aarch64 的 conda。原来的 /home/mh2803/miniconda3 是 x86_64 装的，
#   在本机 `conda` 一律 Exec format error → 已另装 Miniforge3 到 ~/miniforge3。
# ============================================================================
set -uo pipefail

CONDA="${CONDA_EXE_AARCH64:-$HOME/miniforge3/bin/conda}"
ENV_NAME="${ENV_NAME:-auto_vla}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$($CONDA info --base 2>/dev/null)/envs/$ENV_NAME/bin/python"
CU_INDEX="https://download.pytorch.org/whl/cu124"

log() { echo -e "\n\033[1;32m[$(date +%H:%M:%S)] $*\033[0m"; }
err() { echo -e "\033[1;31m[$(date +%H:%M:%S)] $*\033[0m" >&2; }

# 用户 shell 里的 conda 函数可能指向坏掉的 x86 miniconda，屏蔽掉
unset -f conda 2>/dev/null; hash -r

# ---------------------------------------------------------------------------
# 1. conda 层 —— 这 4 个包 PyPI 在 cp39/aarch64 只有 sdist(要现场编译
#    GDAL / Rust / OpenCV)，conda-forge 的 linux-aarch64 恰好有与
#    requirements.txt 完全相同的版本号。
# ---------------------------------------------------------------------------
step_conda() {
    log "conda 层 (conda-forge, linux-aarch64)"
    $CONDA create -y -n "$ENV_NAME" python=3.9 pip || return 1
    $CONDA install -y -n "$ENV_NAME" -c conda-forge \
        numpy=1.23.4 \
        py-opencv=4.9.0 \
        rasterio=1.3.11 \
        fiona=1.10.1 \
        safetensors=0.5.3 || return 1
    # jupyter 三件套单独装：跟上面一起解会让 solver 慢很多，且非必需
    $CONDA install -y -n "$ENV_NAME" -c conda-forge jupyterlab ipykernel notebook \
        || err "jupyter 套件装失败(非必需，忽略)"
}

# ---------------------------------------------------------------------------
# 2. torch 栈 —— 必须用 --index-url(不是 --extra-index-url)
#
#    PyPI 的 aarch64 torch wheel 是 **CPU-only**(~90MB)；CUDA 版在
#    download.pytorch.org/whl/cu124 的 linux_aarch64 (sbsa) 分支。
#    两者版本号都叫 2.5.1(cu124 的 aarch64 wheel 没有 +cu124 local version)，
#    --extra-index-url 不保证优先级 → 会静默装上 CPU-only 版。
#
#    cp39 在 cu124 上最新是 torch 2.5.1 / torchvision 0.20.1
#    (2.4.1 有 cp39 wheel，但没有配套的 torchvision 0.19.x aarch64)
# ---------------------------------------------------------------------------
step_torch() {
    log "torch 栈 (cu124, linux_aarch64/sbsa)"
    "$PY" -m pip install --index-url "$CU_INDEX" torch==2.5.1 torchvision==0.20.1 || return 1
    "$PY" -c "
import torch
assert torch.version.cuda, 'CPU-only wheel 装上了！应该是 cu124 的 CUDA wheel'
print('torch', torch.__version__, 'cuda build', torch.version.cuda)
"
}

# ---------------------------------------------------------------------------
# 3. 其余 pip 依赖
# ---------------------------------------------------------------------------
step_pip() {
    log "pip 依赖 (requirements-aarch64.txt)"
    "$PY" -m pip install -r "$REPO/requirements-aarch64.txt" --no-warn-conflicts
}

# ---------------------------------------------------------------------------
# 4. 本仓库 + navsim devkit
# ---------------------------------------------------------------------------
#    ⚠️ 必须 --no-deps: setup.py / navsim/setup.py 都是写死
#       `open("requirements.txt")` 把里面的东西塞进 install_requires，
#       而那是 x86 的表(有 triton==3.0.0) → 在 aarch64 上直接
#       "No matching distribution found for triton==3.0.0"。
#       这一步只是把包挂到 sys.path 上，依赖已经由 step_pip 装齐了。
step_editable() {
    log "pip install -e . --no-deps (autovla + navsim)"
    "$PY" -m pip install -e "$REPO" --no-deps --no-warn-conflicts || return 1
    "$PY" -m pip install -e "$REPO/navsim" --no-deps --no-warn-conflicts
}

# ---------------------------------------------------------------------------
# 5. verify
#
#    install.sh 里这 3 个本平台装不上，都不影响 166k nocot SFT 主链路:
#      flash-attn      无 aarch64 wheel，要 nvcc 源码编译(在 GPU 节点上跑)。
#                      训练侧 models/autovla.py::load_vlm 没有指定
#                      attn_implementation → transformers 默认走 sdpa，无影响。
#                      只有 dataset_utils/preprocessing/cot_annotation_model.py
#                      写死了 flash_attention_2(CoT 标注用，nocot 链路不碰)。
#      waymo-open-dataset-tf-2-12-0  仅 x86_64，只影响 waymo 预处理
#      autoawq                       仅 x86_64，只影响 AWQ 量化推理
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 4.5 libstdc++ 加载顺序修复（conda activate 钩子）
#
#   RHEL9 系统的 /usr/lib64/libstdc++.so.6 最高 GLIBCXX_3.4.29，而 conda-forge 的
#   opencv/gdal/geos 要 3.4.30~3.4.32。env 里的 libstdc++.so.6.0.36 是够的，
#   各 .so 的 RPATH($ORIGIN/.) 也对 —— 但 `import torch` 会先把系统那份载进进程，
#   之后同 SONAME 的库直接复用它 → cv2/rasterio/fiona/shapely 全部 ImportError。
#   单独 import cv2 好的，import torch 之后就坏。纯加载顺序问题。
#   → 用 activate 钩子把 $CONDA_PREFIX/lib 放到 LD_LIBRARY_PATH 最前面。
# ---------------------------------------------------------------------------
step_ldfix() {
    log "libstdc++ 加载顺序修复 (activate.d 钩子)"
    local envdir; envdir="$($CONDA info --base)/envs/$ENV_NAME"
    mkdir -p "$envdir/etc/conda/activate.d" "$envdir/etc/conda/deactivate.d"
    cat > "$envdir/etc/conda/activate.d/zz-libstdcxx.sh" <<'EOF'
export AUTOVLA_LD_LIBRARY_PATH_BACKUP="${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
EOF
    cat > "$envdir/etc/conda/deactivate.d/zz-libstdcxx.sh" <<'EOF'
if [ -n "${AUTOVLA_LD_LIBRARY_PATH_BACKUP+x}" ]; then
    export LD_LIBRARY_PATH="$AUTOVLA_LD_LIBRARY_PATH_BACKUP"
    unset AUTOVLA_LD_LIBRARY_PATH_BACKUP
    [ -z "$LD_LIBRARY_PATH" ] && unset LD_LIBRARY_PATH
fi
EOF
    echo "  写好 activate.d/zz-libstdcxx.sh —— 之后 conda activate $ENV_NAME 自动生效"
}

step_verify() {
    log "verify"
    # verify 可能在没 activate 的 shell 里跑，这里补上同样的 LD_LIBRARY_PATH
    export LD_LIBRARY_PATH="$($CONDA info --base)/envs/$ENV_NAME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    "$PY" - <<'EOF'
import importlib, platform, sys
print("python  ", sys.version.split()[0], platform.machine())
mods = ["torch","torchvision","transformers","pytorch_lightning","numpy","cv2",
        "rasterio","fiona","geopandas","shapely","safetensors","nuplan","hydra",
        "ray","timm","av","casadi"]
bad = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"  {m:20s} {getattr(mod,'__version__','(no __version__)')}")
    except Exception as e:
        bad.append(m); print(f"  {m:20s} ** FAIL: {type(e).__name__}: {e}")
import torch
print("torch.version.cuda   :", torch.version.cuda, "(None = CPU-only wheel, 装错了)")
print("torch.cuda.is_available():", torch.cuda.is_available(), "(登录节点无 GPU，False 正常)")
print("\nFAILED:", bad if bad else "none")
EOF
}

main() {
    log "env=$ENV_NAME  conda=$CONDA  repo=$REPO"
    case "${1:-all}" in
        conda)    step_conda ;;
        torch)    step_torch ;;
        pip)      step_pip ;;
        editable) step_editable ;;
        ldfix)    step_ldfix ;;
        verify)   step_verify ;;
        all)      step_conda && step_torch && step_pip && step_editable && step_ldfix; step_verify ;;
        *)        echo "未知步骤: $1  (conda|torch|pip|editable|ldfix|verify)"; exit 1 ;;
    esac
}

main "$@"
