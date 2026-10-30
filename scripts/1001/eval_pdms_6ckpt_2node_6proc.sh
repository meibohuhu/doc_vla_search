#!/bin/bash
# ============================================================================
# 0926_tiger 三个 no-CoT ViT 解冻 SFT 的 ep4 / ep5 共 6 个 ckpt，在 navtest 上评 PDMS（greedy）。
#   每个 ckpt: 2 个节点(= 2 张 GH200) x 每卡 3 进程 = navtest 切 6 片。
#   6 ckpt x 2 节点 = 12 个节点，从 16 节点资源池里取空闲的。
#
#   做法照 scripts/0901/eval_cot_sft_greedy_9job_3gpu.sh（brev）：
#     navtest token 取模切片 -> 每片一个 scene_filter -> run_pdm_score_cot.py -> 合并 CSV 算 PDMS。
#   区别：tigris 每节点 1 卡，所以每个"卡"是 pool 里的一个节点，用 srun --jobid 起到节点上
#     （每节点一个 step，step 里跑 scripts/1001/_pdms_node_worker.sh，它在本卡上并行起 3 个进程）。
#
#   greedy：AUTOVLA_SAMPLE_TOP_K=1，与 logs/eval_leaderboard.md 里 84.22 等全量结果口径一致。
#
#   tmux new -s pdms
#   POOL_JOBID=205437 bash scripts/1001/eval_pdms_6ckpt_2node_6proc.sh
#   LIMIT=1000 POOL_JOBID=205437 bash ...        # 只评 seed=42 抽样 1000（先冒烟）
#   CKPTS="a.ckpt b.ckpt" POOL_JOBID=... bash ... # 换 ckpt 列表（每个 ckpt 占 NODES_PER_CKPT 节点）
#   NSHARD=3 NODES_PER_CKPT=1 ...                 # 每个 ckpt 1 卡 x 3 进程
#   AGG_ONLY=1 bash ...                          # 不跑，只重新聚合已有结果
# ============================================================================
set -uo pipefail

export REPO=/home/mh2803/projects/doc_vla_search
DATA=/shared/rc/llm-gen-agent/mhu/auto_vla/nuplan
RUNS=/shared/rc/llm-gen-agent/mhu/auto_vla/runs/sft
OUT_ROOT="${OUT_ROOT:-/shared/rc/llm-gen-agent/mhu/auto_vla/eval/nuplan/sft_eval_1001}"

POOL_JOBID="${POOL_JOBID:-}"
NSHARD="${NSHARD:-6}"                   # 每个 ckpt 切几片（= 每 ckpt 的进程总数）
NODES_PER_CKPT="${NODES_PER_CKPT:-2}"   # 每个 ckpt 几个节点；每节点 NSHARD/NODES_PER_CKPT 个进程
LIMIT="${LIMIT:-0}"            # 0 = 全量 navtest(12146)；给数字 = seed 抽样
SEED="${SEED:-42}"
AGG_ONLY="${AGG_ONLY:-0}"

# 默认 6 个 ckpt：三跑各自 val_loss 最低的 ep4 + top1 最高的 ep5
#   2026-09-27_01-20-47  navfiltbal    (4 节点, navtrain + 166k navfilt_bal)
#   2026-09-27_01-25-35  navfiltturn40 (4 节点, navtrain + 166k navfilt_turn40)
#   2026-09-27_01-25-34  navtrain only (8 节点, 对照组)
CKPTS="${CKPTS:-\
$RUNS/2026-09-27_01-20-47/epoch=4-loss=0.6199.ckpt \
$RUNS/2026-09-27_01-20-47/epoch=5-loss=0.6740.ckpt \
$RUNS/2026-09-27_01-25-35/epoch=4-loss=0.6196.ckpt \
$RUNS/2026-09-27_01-25-35/epoch=5-loss=0.6644.ckpt \
$RUNS/2026-09-27_01-25-34/epoch=4-loss=0.6297.ckpt \
$RUNS/2026-09-27_01-25-34/epoch=5-loss=0.6761.ckpt}"
read -ra CKPT_LIST <<< "$CKPTS"

# ckpt -> 输出目录名：<run 时间戳>_<epoch_x_loss_y>
ckpt_id() { echo "$(basename "$(dirname "$1")")_$(basename "$1" .ckpt | sed 's/[=.-]/_/g')"; }

cd "$REPO" || exit 1
source "$HOME/miniforge3/etc/profile.d/conda.sh" || exit 1
conda activate auto_vla || { echo "ERROR: conda activate auto_vla 失败"; exit 1; }

# --- 环境（srun 默认 --export=ALL，worker 继承）---
export PYTHONPATH="$REPO/navsim:${PYTHONPATH:-}"
export NAVSIM_DEVKIT_ROOT="$REPO/navsim"
export OPENSCENE_DATA_ROOT="$DATA"
export NUPLAN_MAPS_ROOT="$DATA/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export AUTOVLA_SAMPLE_TOP_K=1          # ★ greedy：top_k=1 -> 只保留 argmax
# no-CoT ckpt 用 nocot config（默认）；CoT ckpt 传 EVAL_CONFIG=config/eval/1001/autovla-navtest-eval-cot-tigris.yaml。
# preflight 会拿它的 use_cot 和每个 ckpt 训练时的 hparams.yaml 对，不一致直接退出。
export EVAL_CONFIG="${EVAL_CONFIG:-$REPO/config/eval/1001/autovla-navtest-eval-nocot-tigris.yaml}"
export CACHE_PATH="$DATA/navtest_metric_cache"
export JSON_DATA_PATH="$DATA/navtest_nocot"
export SENSOR_DATA_PATH="$DATA/sensor_blobs/test"

FILTER_DIR="$REPO/navsim/navsim/planning/script/config/common/train_test_split/scene_filter"
# 分片数进 tag：scene_filter 文件名 + 输出目录都按它隔离，不同切法的结果不会混在一起聚合
SPLIT_TAG="navtest1001_${NSHARD}sh$([ "$LIMIT" -gt 0 ] && echo "_l${LIMIT}s${SEED}")"

# ============================================================================
# 聚合：每个 ckpt 合并 6 片 -> PDMS；最后一张总表
# ============================================================================
aggregate() {
  python - "$OUT_ROOT" "$SPLIT_TAG" "${CKPT_LIST[@]}" <<'PY'
import sys, glob, os
import pandas as pd
out_root, tag, ckpts = sys.argv[1], sys.argv[2], sys.argv[3:]
cols = [("score","PDMS"),("no_at_fault_collisions","NC"),("drivable_area_compliance","DAC"),
        ("ego_progress","EP"),("time_to_collision_within_bound","TTC"),("comfort","C"),
        ("driving_direction_compliance","DDC")]
rows = []
for ck in ckpts:
    cid = os.path.basename(os.path.dirname(ck)) + "_" + os.path.basename(ck)[:-5].replace("=", "_").replace(".", "_").replace("-", "_")
    work = os.path.join(out_root, tag, cid)
    csvs = glob.glob(os.path.join(work, "shard*/**/*.csv"), recursive=True)
    if not csvs:
        rows.append((cid, None, 0)); continue
    df = pd.concat([pd.read_csv(x) for x in csvs], ignore_index=True)
    df = df[df.token.notna()].drop_duplicates(subset="token")
    df.to_csv(os.path.join(work, "pdms_merged.csv"), index=False)
    rows.append((cid, df, len(csvs)))
print(f"{'ckpt':<52}{'n':>7}{'片':>4}" + "".join(f"{z:>8}" for _, z in cols))
for cid, df, ncsv in rows:
    if df is None:
        print(f"{cid:<52}{'-':>7}{0:>4}   ⚠️ 没有结果 CSV"); continue
    print(f"{cid:<52}{len(df):>7}{ncsv:>4}" + "".join(f"{df[c].mean()*100:8.2f}" for c, _ in cols))
print("\n参照 (brev, logs/eval_leaderboard.md): navtrain 103k no-CoT ViT 解冻 ep4/ep5 = 84.22 / 84.22 (n≈12126)")
print(f"合并 CSV: {out_root}/{tag}/<ckpt>/pdms_merged.csv")
PY
}

if [ "$AGG_ONLY" = "1" ]; then aggregate; exit $?; fi

# ============================================================================
# preflight
# ============================================================================
fail=0
for f in "${CKPT_LIST[@]}" "$CACHE_PATH" "$JSON_DATA_PATH" "$SENSOR_DATA_PATH" "$DATA/navsim_logs/test" \
         "$EVAL_CONFIG" "$NUPLAN_MAPS_ROOT" "$FILTER_DIR/navtest.yaml" "$REPO/scripts/1001/_pdms_node_worker.sh"; do
  [ -e "$f" ] || { echo "❌ 缺失: $f"; fail=1; }
done
[ "$fail" -eq 0 ] || { echo "前置检查未通过"; exit 1; }
# eval config 的 use_cot 必须和每个 ckpt 训练时一致（get_prompt 按它走两套 system prompt，
# 不一致不会报错、只会无端掉分）。训练时的值从 <run>/lightning_logs/version_0/hparams.yaml 读。
USECOT=$(python - "$EVAL_CONFIG" "${CKPT_LIST[@]}" <<'PY'
import os, sys, yaml
ev = bool(yaml.safe_load(open(sys.argv[1]))["model"]["use_cot"])
bad = []
for ck in sys.argv[2:]:
    hp = os.path.join(os.path.dirname(ck), "lightning_logs/version_0/hparams.yaml")
    if not os.path.exists(hp):
        print(f"  ⚠️ 找不到 {hp}，无法核对 use_cot: {ck}", file=sys.stderr); continue
    tr = bool(yaml.safe_load(open(hp))["config"]["model"]["use_cot"])
    if tr != ev: bad.append(f"{ck} (训练 use_cot={tr})")
if bad:
    sys.exit("ERROR: eval config use_cot=%s，与以下 ckpt 不一致:\n  " % ev + "\n  ".join(bad))
print(ev)
PY
) || exit 1

# --- pool 空闲节点：去掉正在被非 interactive/extern step 占用的 ---
[ -n "$POOL_JOBID" ] || { echo "ERROR: 设 POOL_JOBID=<资源池 JOBID>"; exit 1; }
[ "$(squeue -h -j "$POOL_JOBID" -o %T)" = "RUNNING" ] || { echo "ERROR: pool $POOL_JOBID 不在 RUNNING"; exit 1; }
mapfile -t POOL_NODES < <(scontrol show hostnames "$(squeue -h -j "$POOL_JOBID" -o %N)")
BUSY=" "
while IFS='|' read -r sid snodes; do
  for h in $(scontrol show hostnames "$snodes"); do BUSY+="$h "; done
done < <(squeue -h -s -j "$POOL_JOBID" -o "%i|%N" | grep -vE '^[0-9]+\.(extern|batch|interactive)\|')
FREE=()
for h in "${POOL_NODES[@]}"; do [[ "$BUSY" == *" $h "* ]] || FREE+=("$h"); done
need=$(( ${#CKPT_LIST[@]} * NODES_PER_CKPT ))
[ ${#FREE[@]} -ge "$need" ] || { echo "ERROR: 需要 $need 个空闲节点，pool 里只有 ${#FREE[@]} 个空闲 (忙: $BUSY)"; exit 1; }

# ============================================================================
# navtest 切 6 片（取模切分，难度均匀）；所有 ckpt 共用同一套分片
# ============================================================================
TAG="$SPLIT_TAG" python - "$NSHARD" "$FILTER_DIR" "$LIMIT" "$SEED" <<'PY'
import sys, yaml, os, random
n, fdir, limit, seed = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
tag = os.environ["TAG"]
cfg = yaml.safe_load(open(os.path.join(fdir, "navtest.yaml")))
toks = cfg["tokens"]
if 0 < limit < len(toks):
    random.Random(seed).shuffle(toks); toks = toks[:limit]
for i in range(n):
    shard = dict(cfg); shard["tokens"] = toks[i::n]
    with open(os.path.join(fdir, f"{tag}_shard{i}of{n}.yaml"), "w") as f:
        yaml.safe_dump(shard, f, default_flow_style=False, sort_keys=False)
print(f"  navtest 切 {n} 片: 每片 ~{len(toks)//n:,}，合计 {len(toks):,}")
PY
for i in $(seq 0 $((NSHARD - 1))); do
  printf 'defaults:\n  - scene_filter: %s_shard%dof%d\n\ndata_split: test\n' "$SPLIT_TAG" "$i" "$NSHARD" \
    > "$FILTER_DIR/../${SPLIT_TAG}_shard${i}of${NSHARD}.yaml"
done

echo "=================================================================="
echo " PDMS on navtest (greedy, use_cot=$USECOT)   pool=$POOL_JOBID"
echo " ckpt     : ${#CKPT_LIST[@]} 个，每个 $NODES_PER_CKPT 节点 x $((NSHARD / NODES_PER_CKPT)) 进程 = $NSHARD 片"
echo " 场景     : $([ "$LIMIT" -eq 0 ] && echo '全量 navtest' || echo "$LIMIT (seed=$SEED)")"
echo " eval cfg : $EVAL_CONFIG"
echo " 输出     : $OUT_ROOT/$SPLIT_TAG/<ckpt>/"
echo "=================================================================="

# ============================================================================
# 起 step：ckpt k 用 FREE[2k], FREE[2k+1]；节点 j 跑 shard s (s % 2 == j)
# ============================================================================
STEP_PIDS=()
for k in "${!CKPT_LIST[@]}"; do
  ck="${CKPT_LIST[$k]}"; cid=$(ckpt_id "$ck")
  work="$OUT_ROOT/$SPLIT_TAG/$cid"; mkdir -p "$work"
  for j in $(seq 0 $((NODES_PER_CKPT - 1))); do
    node="${FREE[$(( k * NODES_PER_CKPT + j ))]}"
    shards=(); for s in $(seq "$j" "$NODES_PER_CKPT" $((NSHARD - 1))); do shards+=("$s"); done
    echo "  $cid  ->  $node  shards ${shards[*]}"
    srun --jobid="$POOL_JOBID" --nodelist="$node" -N 1 -n 1 \
         --cpus-per-task=16 --mem=0 --gres=gpu:gh200:1 --exclusive \
         --job-name="pdms_k${k}_n${j}" \
         bash "$REPO/scripts/1001/_pdms_node_worker.sh" "$ck" "$work" "$SPLIT_TAG" "$NSHARD" "${shards[@]}" \
         > "$work/node${j}_${node}.log" 2>&1 &
    STEP_PIDS+=($!)
  done
done

echo; echo "已起 ${#STEP_PIDS[@]} 个 step，等待全部完成 ...  (进度: tail -f $OUT_ROOT/$SPLIT_TAG/*/shard*/run.log)"
nfail=0
for p in "${STEP_PIDS[@]}"; do wait "$p" || nfail=$((nfail + 1)); done
[ "$nfail" -gt 0 ] && echo "⚠️ $nfail 个 step 非 0 退出，看 $OUT_ROOT/$SPLIT_TAG/*/node*.log 和 shard*/run.log"

echo; echo "==================== 结果 ===================="
aggregate
