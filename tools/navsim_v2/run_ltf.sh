#!/bin/bash
# 参照基线：官方 Latent TransFuser（LTF，只用当前帧 F0 / L0 / R0，不用激光雷达）在两阶段 split 上的 EPDMS。
# 完全用官方 v2 代码：run_create_submission_pickle.py 出 submission.pkl，再走 score_local.sh。
# 论文（NAVSIM v2，arXiv 2506.04218）里 LTF 在 navhard 上是 25.1。
#
# 用法：
#   GPU=2 bash tools/navsim_v2/run_ltf.sh                              # warmup_two_stage
#   SPLIT=navhard_two_stage GPU=2 bash tools/navsim_v2/run_ltf.sh
# ckpt 来自官方公开的 HF autonomousvision/navsim_baselines（ltf/ltf_seed_0.ckpt，和 NavSafe model zoo 里的同一个）：
#   huggingface-cli download autonomousvision/navsim_baselines ltf/ltf_seed_0.ckpt --local-dir $V2_ROOT/checkpoints
set -e
source "$(dirname "$0")/env.sh"
SPLIT=${SPLIT:-warmup_two_stage}
LTF_CKPT=${LTF_CKPT:-$V2_ROOT/checkpoints/ltf/ltf_seed_0.ckpt}
OUT=$V2_EXP/$SPLIT/ltf
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=${GPU:-2} v2py "$V2_DEVKIT/navsim/planning/script/run_create_submission_pickle.py" \
    train_test_split="$SPLIT" agent=transfuser_agent agent.config.latent=True agent.checkpoint_path="$LTF_CKPT" \
    experiment_name=submission_ltf_$SPLIT output_dir="$OUT/run" \
    team_name="${TEAM_NAME:-MUST_SET}" authors="${AUTHORS:-MUST_SET}" email="${EMAIL:-MUST_SET}" \
    institution="${INSTITUTION:-MUST_SET}" country="${COUNTRY:-MUST_SET}" \
    synthetic_sensor_path="$V2_DATA/$SPLIT/sensor_blobs" synthetic_scenes_path="$V2_DATA/$SPLIT/synthetic_scene_pickles" \
    > "$OUT/create_submission.log" 2>&1 || { tail -30 "$OUT/create_submission.log"; exit 1; }
cp "$OUT/run/submission.pkl" "$OUT/submission.pkl"
grep -c "Agent failed for token" "$OUT/create_submission.log" | xargs echo "推理失败的 token 数："

[ -n "$SKIP_SCORE" ] || bash "$HERE/score_local.sh" "$SPLIT" "$OUT/submission.pkl" "score_${SPLIT}_ltf"
