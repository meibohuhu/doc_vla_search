#!/bin/bash
# 全量 navtest(12146) 评测 autovla-nocot-sft-navtrain-vit-unfreeze-adamw8bit-tigris8n 的两个 ckpt
#   epoch=4 -> GPU1, epoch=5 -> GPU2；每个 ckpt 在自己的卡上并发 3 个 job（3 分片，最后合并）
# 实际评测逻辑复用 scripts/0725/eval_ckpt_gpu2.sh（CKPT/GPU/NSHARD/LIMIT/WORK_SUBDIR 由环境变量传入）
# 用法: bash scripts/0902_gcis/eval_tigris8n_ep4_ep5.sh
set -u
REPO="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
C=/scratch/mh2803/vla/checkpoints/autovla-nocot-sft-navtrain-vit-unfreeze-adamw8bit-tigris8n
LOG="$REPO/logs/eval_0930"; mkdir -p "$LOG"
cd "$REPO"

for e in 4:1:0.6297 5:2:0.6761; do
    IFS=: read ep gpu loss <<< "$e"
    CKPT="$C/epoch=$ep-loss=$loss.ckpt" GPU=$gpu NSHARD=3 LIMIT=0 WORK_SUBDIR=eval_tigris8n_ep$ep \
        nohup bash scripts/0725/eval_ckpt_gpu2.sh > "$LOG/ep$ep.log" 2>&1 &
done
wait
