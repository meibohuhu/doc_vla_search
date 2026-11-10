#!/bin/bash
# 准备 NAVSIM v2：clone 官方 devkit（固定 commit），下载一个两阶段 split，接上本地的 test logs / 图 / 地图。
#
# 用法：
#   bash tools/navsim_v2/setup_data.sh warmup     # warmup_two_stage，1.2 GB，16 + 204 个 token
#   bash tools/navsim_v2/setup_data.sh navhard    # navhard_two_stage，约 32 GB，450 + 5462 个 token
# 重复跑是安全的：已经 clone / 解压过的会跳过。
# （private_test_hard_two_stage 是 2025 比赛的，目录结构不一样、比赛也结束了，这里不支持。）
set -e
source "$(dirname "$0")/env.sh"

case "${1:-warmup}" in
    warmup)  SPLIT=warmup_two_stage ;;
    navhard) SPLIT=navhard_two_stage ;;
    *) echo "用法：$0 warmup|navhard"; exit 1 ;;
esac

# 1. devkit
if [ ! -d "$V2_DEVKIT/.git" ]; then
    git clone -q https://github.com/autonomousvision/navsim.git "$V2_DEVKIT"
fi
git -C "$V2_DEVKIT" checkout -q "$V2_COMMIT"
echo "devkit：$V2_DEVKIT @ $(git -C "$V2_DEVKIT" log --oneline -1)"

# 2. 本地已有的 OpenScene test logs / 图：v2 默认去 $OPENSCENE_DATA_ROOT/{navsim_logs,sensor_blobs}/test 找第一阶段的场景
mkdir -p "$V2_DATA" "$V2_EXP"
for d in navsim_logs sensor_blobs; do
    [ -e "$V2_DATA/$d" ] || ln -s "/scratch/mh2803/vla/nuplan/$d" "$V2_DATA/$d"
done

# 3. 下载 split（用 devkit 自带的下载脚本，在 $V2_DATA 里解压）
if [ -d "$V2_DATA/$SPLIT/synthetic_scene_pickles" ]; then
    echo "$SPLIT 已经在 $V2_DATA/$SPLIT，跳过下载"
else
    cd "$V2_DATA"
    bash "$V2_DEVKIT/download/download_${SPLIT}.sh"
fi
ls "$V2_DATA/$SPLIT"
echo "合成场景 $(ls "$V2_DATA/$SPLIT/synthetic_scene_pickles" | wc -l) 个，图 $(find -L "$V2_DATA/$SPLIT/sensor_blobs" -type f | wc -l) 张"
