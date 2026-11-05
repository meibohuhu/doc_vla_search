#!/bin/bash
# AutoVLA（cf SFT）在 CoLT-Drive 上输出 meta-action：每张卡跑一片，全部跑完后自动 agg。
# 说明见 tools/colt/eval_autovla.py 开头。每片约 10 GB 显存（3B，bf16）。
#
# 用法：
#   GPUS="0 2" bash scripts/1104/run_colt_autovla.sh                       # vfull，none / vlm / oracle 三种指令
#   GPUS="0 2" bash scripts/1104/run_colt_autovla.sh --sides black         # 左前 / 右前喂全黑图
#   GPUS="0" bash scripts/1104/run_colt_autovla.sh --ckpt <另一个 ckpt>    # 换 ckpt（结果按 ckpt 目录名分开存）
# 额外参数会原样传给 run 和 agg，所以 run 和 agg 的 --split / --sides / --tag 一定一致。
# 中断后重跑会接着做；换卡数（nshard）也没关系，已跑完的按（样本, 指令）跳过。
#
# 换机器：用环境变量改仓库、Python、日志目录，路径参数照常往后加，例如
#   PY=python LOGDIR=/path/logs GPUS="0 1 2 3" bash scripts/1104/run_colt_autovla.sh \
#       --data /path/CoLT-Drive/data --ckpt /path/epoch=4-loss=0.6051.ckpt \
#       --base /path/Qwen2.5-VL-3B-Instruct --vlm-labels /path/final_single_v28_single_f0_01aabc_w1280.jsonl \
#       --out /path/colt/autovla
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
cd "$REPO" || exit 1
PY=${PY:-/home/mh2803/miniconda3/envs/autovla_codeclean/bin/python}
read -ra G <<< "${GPUS:-0 2}"
N=${#G[@]}
LOG=${LOGDIR:-/scratch/mh2803/vla/colt/autovla/logs}/$(date +%m%d_%H%M)
mkdir -p "$(dirname "$LOG")"
pids=()
for i in "${!G[@]}"; do
    CUDA_VISIBLE_DEVICES=${G[$i]} $PY tools/colt/eval_autovla.py run --shard "$i" --nshard "$N" "$@" \
        > "${LOG}_s${i}_gpu${G[$i]}.log" 2>&1 &
    pids+=($!)
    echo "片 $i/$N → GPU ${G[$i]}，pid $!，日志 ${LOG}_s${i}_gpu${G[$i]}.log"
    sleep 30   # 错开读 15 GB 的 ckpt
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ $fail = 1 ] && echo "⚠️ 有分片异常退出，看上面的日志；结果先按已跑完的出表"
$PY tools/colt/eval_autovla.py agg "$@"
