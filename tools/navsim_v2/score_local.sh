#!/bin/bash
# 用官方 v2 代码在本地给 submission.pkl 打 EPDMS（两阶段）。metric cache 每个 split 只建一次。
#
# 用法：
#   bash tools/navsim_v2/score_local.sh <split> <submission.pkl> [实验名]
#   例：bash tools/navsim_v2/score_local.sh warmup_two_stage $V2_EXP/warmup_two_stage/autovla/submission.pkl
# 结果：$V2_EXP/<实验名>/<时间>/*.csv，最后三行是 extended_pdm_score_stage_one / stage_two / combined，
#       combined 就是 leaderboard 上的 EPDMS。脚本最后会把这三行打印出来。
# WORKER 默认 ray_distributed_no_torch（官方默认），出问题可以设 WORKER=sequential。
set -e
source "$(dirname "$0")/env.sh"
SPLIT=${1:?split}
SUB=${2:?submission.pkl}
NAME=${3:-score_${SPLIT}_$(basename "$(dirname "$SUB")")}
CACHE=$V2_EXP/metric_cache/$SPLIT
WORKER=${WORKER:-ray_distributed_no_torch}
SYN_SCENES=$V2_DATA/$SPLIT/synthetic_scene_pickles
SYN_SENSORS=$V2_DATA/$SPLIT/sensor_blobs
SCRIPT=$V2_DEVKIT/navsim/planning/script

if [ ! -f "$CACHE/.done" ]; then
    echo "建 metric cache → $CACHE"
    v2py "$SCRIPT/run_metric_caching.py" train_test_split="$SPLIT" metric_cache_path="$CACHE" \
        synthetic_scenes_path="$SYN_SCENES" worker="$WORKER"
    touch "$CACHE/.done"
fi

v2py "$SCRIPT/run_pdm_score_from_submission.py" train_test_split="$SPLIT" submission_file_path="$SUB" \
    metric_cache_path="$CACHE" synthetic_scenes_path="$SYN_SCENES" synthetic_sensor_path="$SYN_SENSORS" \
    experiment_name="$NAME" worker="$WORKER"

CSV=$(ls -t "$V2_EXP/$NAME"/*/*.csv | head -1)
echo "结果：$CSV"
v2py - "$CSV" <<'EOF'
import sys, pandas as pd
df = pd.read_csv(sys.argv[1])
summ = df["token"].astype(str).str.startswith("extended_pdm_score")
s = df[summ].set_index("token")
print(f"有效 token：{int(df.loc[~summ, 'valid'].astype(str).eq('True').sum())}/{int((~summ).sum())}")
for k in s.index:
    print(f"{k:32s} EPDMS {100 * float(s.loc[k, 'score']):.2f}")
cols = [c for c in s.columns if c.endswith("_stage_one") or c.endswith("_stage_two")]
print(s.loc[["extended_pdm_score_combined"], cols].T.dropna().round(4).to_string())
EOF
