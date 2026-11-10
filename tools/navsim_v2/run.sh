#!/bin/bash
# AutoVLA 在 NAVSIM v2 两阶段 split 上一条龙：建输入 → 多卡推理 → submission.pkl → 本地 EPDMS。
#
# 用法：
#   GPUS="0 2" bash tools/navsim_v2/run.sh                          # warmup_two_stage，默认 ckpt（env.sh）
#   SPLIT=navhard_two_stage GPUS="0 2" bash tools/navsim_v2/run.sh  # navhard（先 setup_data.sh navhard）
#   MODEL_DIR=... CKPT=... CONFIG=... TAG=xxx bash tools/navsim_v2/run.sh   # 换 ckpt，结果按 TAG 分开
#   SKIP_SCORE=1 ...                                                # 只出 submission.pkl，不打分
#   上传用：TEAM_NAME=... AUTHORS=... EMAIL=... INSTITUTION=... COUNTRY=... 一起设上，会写进 pkl
# 中断后重跑会接着推理（按 token 跳过已完成的）。
# 输出：$V2_EXP/<split>/inputs/            输入（所有 ckpt 共用）
#       $V2_EXP/<split>/<TAG>/preds/       每片一个 preds_s*.jsonl（轨迹 + 生成文本）
#       $V2_EXP/<split>/<TAG>/submission.pkl
#       $V2_EXP/<split>/<TAG>/logs/
source "$(dirname "$0")/env.sh"
SPLIT=${SPLIT:-warmup_two_stage}
TAG=${TAG:-$(basename "$MODEL_DIR")}
read -ra G <<< "${GPUS:-0 2}"
N=${#G[@]}
W=$V2_EXP/$SPLIT
OUT=$W/$TAG
mkdir -p "$OUT/logs"
cd "$REPO" || exit 1

# 1. 输入
if [ ! -f "$W/inputs/inputs.jsonl" ]; then
    v2py "$HERE/build_inputs.py" --split "$SPLIT" --out "$W/inputs" || exit 1
fi

# 2. 推理：一张卡一片
LOG=$OUT/logs/$(date +%m%d_%H%M)
pids=()
for i in "${!G[@]}"; do
    CUDA_VISIBLE_DEVICES=${G[$i]} "$PY" "$HERE/infer_autovla.py" --inputs "$W/inputs/inputs.jsonl" --out "$OUT/preds" \
        --ckpt "$CKPT" --config "$CONFIG" --base "$BASE" --shard "$i" --nshard "$N" \
        > "${LOG}_s${i}_gpu${G[$i]}.log" 2>&1 &
    pids+=($!)
    echo "片 $i/$N → GPU ${G[$i]}，pid $!，日志 ${LOG}_s${i}_gpu${G[$i]}.log"
    [ "$i" -lt $((N - 1)) ] && sleep 30   # 错开读 15 GB 的 ckpt
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ $fail = 0 ] || { echo "有推理分片失败，看 ${LOG}_s*.log"; exit 1; }

# 3. submission.pkl
v2py "$HERE/make_submission.py" --inputs "$W/inputs/inputs.jsonl" --preds "$OUT/preds" --out "$OUT/submission.pkl" \
    --team-name "${TEAM_NAME:-MUST_SET}" --authors "${AUTHORS:-MUST_SET}" --email "${EMAIL:-MUST_SET}" \
    --institution "${INSTITUTION:-MUST_SET}" --country "${COUNTRY:-MUST_SET}" || exit 1

# 4. 本地打分
[ -n "$SKIP_SCORE" ] || bash "$HERE/score_local.sh" "$SPLIT" "$OUT/submission.pkl" "score_${SPLIT}_${TAG}"
