#!/bin/bash
# ============================================================================
# 并排评测多个 SFT checkpoint（navtest 子集）
#
# 布局：每个 ckpt 切 2 片、各占 1 张 GPU
#   ckpt1 -> GPU 4, 5
#   ckpt2 -> GPU 6, 7
#
# 用法:
#   bash scripts/0723/eval_sft_ckpts.sh                       # 默认 1000 场景，两个 ckpt
#   LIMIT=12146 bash scripts/0723/eval_sft_ckpts.sh           # 全量
#   GPULIST=0,1,2,3 bash scripts/0723/eval_sft_ckpts.sh       # 换卡
#   CKPTS="/a.ckpt /b.ckpt" bash scripts/0723/eval_sft_ckpts.sh
#
# 注意：
#   * 用 seed=42 抽样，与之前评测发布 ckpt 的那 1000 个场景【完全相同】，
#     所以可以直接跟 PDMS 89.06 对比。
#   * 用 use_cot=false 的 eval config —— 这两个 ckpt 是 no-CoT SFT 训的，
#     用 CoT 版 prompt 会分布外，分数无端偏低且不报错。
#   * 每个 ckpt 用独立的 scene_filter 文件名，避免并发跑时互相覆盖。
# ============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "$(readlink -f "$0")")" && while [ ! -f setup.py ] && [ "$PWD" != / ]; do cd ..; done; pwd)"
DATA=/data/autovla_data/nuplan
PY=/data/autovla_data/envs/autovla/bin/python

CKPTS="${CKPTS:-/data/autovla_data/checkpoints/sft/2026-07-23_04-41-59/epoch=4-loss=1.0498.ckpt /data/autovla_data/checkpoints/sft/2026-07-23_04-41-59/epoch=3-loss=1.0584.ckpt}"
LIMIT="${LIMIT:-1000}"
SEED="${SEED:-42}"          # 抽样种子；换 seed 得到不同的 1000，但结果不能直接跟 seed=42 的 89.06 比
GPULIST="${GPULIST:-4,5,6,7}"        # 每个 ckpt 吃掉其中 2 张
NPROC_PER_CKPT="${NPROC_PER_CKPT:-2}"
EVAL_CONFIG="${EVAL_CONFIG:-$REPO/config/eval/0723/autovla-navtest-eval-nocot.yaml}"

export PYTHONPATH="$REPO/navsim:${PYTHONPATH:-}"
export NAVSIM_DEVKIT_ROOT="$REPO/navsim"
export OPENSCENE_DATA_ROOT="$DATA"
export NUPLAN_MAPS_ROOT="$DATA/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export TOKENIZERS_PARALLELISM=false

CACHE_PATH="$DATA/navtest_metric_cache"
JSON_DATA_PATH="$DATA/navtest_nocot"
SENSOR_DATA_PATH="$DATA/sensor_blobs/test"
FILTER_DIR="$REPO/navsim/navsim/planning/script/config/common/train_test_split/scene_filter"
WORK="$DATA/${WORK_SUBDIR:-sft_eval}"
mkdir -p "$WORK"
cd "$REPO"

# ---- 前置检查：缺什么直接停，别跑到一半才发现 ----
fail=0
for f in "$CACHE_PATH" "$JSON_DATA_PATH" "$SENSOR_DATA_PATH" "$EVAL_CONFIG" "$NUPLAN_MAPS_ROOT"; do
    [ -e "$f" ] || { echo "❌ 缺失: $f"; fail=1; }
done
i=0; for c in $CKPTS; do
    [ -f "$c" ] || { echo "❌ ckpt 不存在: $c"; fail=1; }
    i=$((i+1))
done
[ "$fail" -eq 0 ] || { echo "前置检查未通过。"; exit 1; }
USECOT=$($PY -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['model']['use_cot'])" "$EVAL_CONFIG")
echo "eval config : $EVAL_CONFIG"
echo "use_cot     : $USECOT   (no-CoT SFT ckpt 必须为 False)"
echo "场景数      : $LIMIT   (seed=$SEED 随机抽样)"
echo

# ---- 生成分片 scene_filter（所有 ckpt 共用同一批 token，保证可比）----
NSHARD=$NPROC_PER_CKPT
TAG="${WORK_SUBDIR:-sfteval}"   # scene_filter 文件名前缀；并发跑不同 ckpt 时设 WORK_SUBDIR 避免互相覆盖
TAG="$TAG" $PY - "$NSHARD" "$FILTER_DIR" "$LIMIT" "$SEED" <<'PY'
import sys, yaml, os, random
n, fdir, limit, seed = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
tag = os.environ["TAG"]
cfg = yaml.safe_load(open(os.path.join(fdir, "navtest.yaml")))
toks = cfg["tokens"]
if limit > 0 and limit < len(toks):
    # 与 run_navtest_pdm_eval_sharded.sh 完全一致的抽样，结果才可跟 89.06 对比
    random.Random(seed).shuffle(toks)
    toks = toks[:limit]
for i in range(n):
    shard = dict(cfg); shard["tokens"] = toks[i::n]
    with open(os.path.join(fdir, f"{tag}_shard{i}of{n}.yaml"), "w") as f:
        yaml.safe_dump(shard, f, default_flow_style=False, sort_keys=False)
    print(f"  shard {i}: {len(shard['tokens']):,} tokens")
print(f"合计 {len(toks):,}")
PY
for i in $(seq 0 $((NSHARD-1))); do
    printf 'defaults:\n  - scene_filter: %s_shard%dof%d\n\ndata_split: test\n' "$TAG" "$i" "$NSHARD" \
        > "$FILTER_DIR/../${TAG}_shard${i}of${NSHARD}.yaml"
done
echo

# ---- 启动：ckpt k 用 GPULIST 里第 k 组的 NPROC_PER_CKPT 张卡 ----
# 注意 checkpoint_path 外面那层单引号：ckpt 文件名含 '='（epoch=4-loss=1.0498），
# hydra 的 override 语法会把它当成 key=value 分隔符 -> "mismatched input '='"。
k=0
for ckpt in $CKPTS; do
    name=$(basename "$ckpt" .ckpt | tr '=' '_' | tr '.' '_')
    for s in $(seq 0 $((NSHARD-1))); do
        gpu=$(echo "$GPULIST" | cut -d, -f$((k*NSHARD + s + 1)))
        d="$WORK/$name/shard$s"; mkdir -p "$d"
        echo "  GPU $gpu  <-  $name  shard$s"
        CUDA_VISIBLE_DEVICES=$gpu NAVSIM_EXP_ROOT="$d" \
        $PY "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_cot.py" \
            train_test_split="${TAG}_shard${s}of${NSHARD}" \
            agent=autovla_agent \
            agent.config_path="$EVAL_CONFIG" \
            agent.checkpoint_path="'$ckpt'" \
            agent.sensor_data_path="$SENSOR_DATA_PATH" \
            agent.lora_conf.use_lora=false \
            metric_cache_path="$CACHE_PATH" \
            +json_data_path="$JSON_DATA_PATH" \
            experiment_name="${name}_s${s}" \
            > "$d/run.log" 2>&1 &
    done
    k=$((k+1))
done

echo
echo "已启动 $((k*NSHARD)) 个进程。等待..."
wait

# ---- 汇总对比 ----
echo
echo "==================== 结果 ===================="
$PY - "$WORK" "$CKPTS" <<'PY'
import sys, glob, os
import pandas as pd
work = sys.argv[1]
names = [os.path.basename(c).replace('.ckpt','').replace('=','_').replace('.','_')
         for c in sys.argv[2].split()]
rows = []
for c, n in zip(sys.argv[2].split(), names):
    csvs = glob.glob(os.path.join(work, n, "shard*/**/*.csv"), recursive=True)
    if not csvs:
        print(f"⚠️ {n}: 没有结果"); continue
    df = pd.concat([pd.read_csv(x) for x in csvs], ignore_index=True)
    df = df[df.token.notna()].drop_duplicates(subset="token")
    df.to_csv(os.path.join(work, f"{n}_merged.csv"), index=False)
    rows.append((os.path.basename(c), len(df), df))
cols = [("score","PDMS"),("no_at_fault_collisions","NC"),("drivable_area_compliance","DAC"),
        ("ego_progress","EP"),("time_to_collision_within_bound","TTC"),("comfort","C"),
        ("driving_direction_compliance","DDC")]
print(f"{'checkpoint':<34}{'n':>6}" + "".join(f"{z:>8}" for _,z in cols))
print("-"*(40+8*len(cols)))
for name, n, df in rows:
    print(f"{name:<34}{n:>6}" + "".join(f"{df[c].mean()*100:8.2f}" for c,_ in cols))
print()
print("参照: 发布的 RFT ckpt 在同一批 1000 场景上 PDMS = 89.06 (全量 12,126 上 89.48)")
if len(rows) == 2:
    a = rows[0][2].set_index('token')
    b = rows[1][2].set_index('token')
    common = a.index.intersection(b.index)
    d = a.loc[common, 'score'].sub(b.loc[common, 'score'])
    print(f"\n逐场景配对 ({len(common)} 个):  {rows[0][0]} 更好 {(d>0.01).sum()} / "
          f"持平 {(d.abs()<=0.01).sum()} / {rows[1][0]} 更好 {(d<-0.01).sum()}")
PY
