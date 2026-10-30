#!/bin/bash
# ============================================================================
# cf 开环评测（docs/1026/cf_sft_autovla_plan.md §5.3），在资源池里按节点分片跑。
#   每个节点 1 片（tools/cf_data/eval_cf_openloop.py run --shard k --nshard N），全部跑完后 agg。
#
#   tmux new -s cfeval
#   POOL_JOBID=214334 NODES=gh-a-[001-008] CKPT=<ckpt> bash tools/cf_data/run_eval_cf_openloop.sh
#   EXTRA="--report_check --limit 300" TAG=report300 ...   # 抽 300 条做 report 检查（每条多 7 次生成）
#   默认 EXTRA="--shortcut_check"（照做 + 对照 + shortcut，全部 3,385 条）
# 中断后原命令重跑会跳过已完成的条目。
# ============================================================================
set -uo pipefail
REPO=/home/mh2803/projects/doc_vla_search
POOL_JOBID="${POOL_JOBID:?设 POOL_JOBID}"
NODES="${NODES:?设 NODES，如 gh-a-[001-008]}"
CKPT="${CKPT:?设 CKPT}"
EXTRA="${EXTRA:---shortcut_check}"
RUN=$(basename "$(dirname "$CKPT")"); EP=$(basename "$CKPT" .ckpt | sed 's/[=.-]/_/g')
TAG="${TAG:-full}"
OUT="${OUT:-/shared/rc/llm-gen-agent/mhu/auto_vla/eval/cf_openloop/${RUN}_${EP}_${TAG}}"

cd "$REPO" || exit 1
source "$HOME/miniforge3/etc/profile.d/conda.sh" && conda activate auto_vla || exit 1
export PYTHONPATH="$REPO/navsim:$REPO" NAVSIM_DEVKIT_ROOT="$REPO/navsim"
export OPENSCENE_DATA_ROOT=/shared/rc/llm-gen-agent/mhu/auto_vla/nuplan
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps" NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export AUTOVLA_SAMPLE_TOP_K=1          # greedy，与 PDMS 评测口径一致

[ -f "$CKPT" ] || { echo "ERROR: 找不到 $CKPT"; exit 1; }
mapfile -t NL < <(scontrol show hostnames "$NODES")
N=${#NL[@]}
mkdir -p "$OUT"
echo "cf 开环评测  ckpt=$CKPT"
echo "  节点 $N 个: ${NL[*]}   extra: $EXTRA"
echo "  输出: $OUT"

pids=()
for k in "${!NL[@]}"; do
  srun --jobid="$POOL_JOBID" --nodelist="${NL[$k]}" -N 1 -n 1 --cpus-per-task=16 --mem=0 \
       --gres=gpu:gh200:1 --exclusive --job-name="cfeval_${k}" \
       python tools/cf_data/eval_cf_openloop.py run --out "$OUT" --ckpt "$CKPT" \
       --shard "$k" --nshard "$N" $EXTRA > "$OUT/shard${k}_${NL[$k]}.log" 2>&1 &
  pids+=($!)
done
nfail=0
for p in "${pids[@]}"; do wait "$p" || nfail=$((nfail + 1)); done
[ "$nfail" -gt 0 ] && echo "⚠️ $nfail 个分片非 0 退出，看 $OUT/shard*.log（重跑本命令可续跑）"
echo; python tools/cf_data/eval_cf_openloop.py agg --out "$OUT"
