#!/bin/bash
# ============================================================================
# navtest PDMS 评测 —— 8 卡分片并行版
#
# 背景：run_pdm_score_cot.py 的 worker 是 Sequential，单卡实测 ~2.3s/场景，
#       12,146 个场景要 ~7.8 小时。把 token 切成 N 份、每份一张卡，约 1 小时。
#
# 做法：从 navtest.yaml 派生 N 个 scene_filter（log_names 全保留，tokens 分片），
#       每片一个进程 + 一张 GPU，最后合并 CSV。
#
# 用法:
#   bash scripts/0721/run_navtest_pdm_eval_sharded.sh              # 8 卡跑全量 12,146
#   bash scripts/0721/run_navtest_pdm_eval_sharded.sh 4            # 4 卡跑全量
#   LIMIT=1000 bash .../run_navtest_pdm_eval_sharded.sh 4          # 4 卡只跑前 1000 个
#   GPUS=0,1,4,5 LIMIT=1000 bash .../..._sharded.sh 4              # 指定用哪几张卡
#   bash scripts/0721/run_navtest_pdm_eval_sharded.sh 4 merge      # 只合并已有结果
# ============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "$(readlink -f "$0")")" && while [ ! -f setup.py ] && [ "$PWD" != / ]; do cd ..; done; pwd)"
DATA=/data/autovla_data/nuplan
PY=/data/autovla_data/envs/autovla/bin/python
NSHARD="${1:-8}"
MODE="${2:-run}"
LIMIT="${LIMIT:-0}"          # 0 = 全量；否则随机抽样 N 个 token
TOKENS="${TOKENS:-}"        # 指定 token 列表文件（每行一个），优先于 LIMIT
GPUS="${GPUS:-}"            # 逗号分隔的卡号；留空则用 0..NSHARD-1

export PYTHONPATH="$REPO/navsim:${PYTHONPATH:-}"
export NAVSIM_DEVKIT_ROOT="$REPO/navsim"
export NAVSIM_EXP_ROOT="$DATA/exp"
export OPENSCENE_DATA_ROOT="$DATA"
export NUPLAN_MAPS_ROOT="$DATA/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export TOKENIZERS_PARALLELISM=false

CHECKPOINT="/data/autovla_data/checkpoints/AutoVLA_PDMS_89.ckpt"
CACHE_PATH="$DATA/navtest_metric_cache"
JSON_DATA_PATH="$DATA/navtest_nocot"
SENSOR_DATA_PATH="$DATA/sensor_blobs/test"
EVAL_CONFIG="$REPO/config/eval/0721/autovla-navtest-eval.yaml"
LORA=false          # ⚠️ 发布 ckpt 已 merge LoRA，见 issue #48
OUT="${OUT_DIR:-$DATA/pdms_shards}"
DUMP="${DUMP:-}"            # 设为目录则转储模型原始输出(JSONL)
FILTER_DIR="$REPO/navsim/navsim/planning/script/config/common/train_test_split/scene_filter"

cd "$REPO"

# ---------------------------------------------------------------- 生成分片
gen_shards() {
    mkdir -p "$OUT"
    $PY - "$NSHARD" "$FILTER_DIR" "$OUT" "$LIMIT" "$TOKENS" <<'PY'
import sys, yaml, os
n, fdir, out, limit = int(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
tokfile = sys.argv[5] if len(sys.argv) > 5 else ""
cfg = yaml.safe_load(open(os.path.join(fdir, "navtest.yaml")))
toks = cfg["tokens"]
if tokfile and os.path.isfile(tokfile):
    want = [l.strip() for l in open(tokfile) if l.strip()]
    tset = set(want)
    toks = [t for t in toks if t in tset]
    print(f"  指定 token 列表: {len(want)} 个请求 -> {len(toks)} 个在 navtest 内")
elif limit > 0:
    # ⚠️ 不能用 toks[:limit]：navtest.yaml 的 token 是按 log 聚集排列的，
    # 前 1000 个只落在 16/136 个 log 里（11.8%），子集结果与全量严重不可比。
    # 固定 seed 随机抽样，保证跨 log 覆盖且可复现。
    import random
    random.Random(42).shuffle(toks)
    toks = toks[:limit]
    print(f"  ⚠️ 随机抽样 {limit:,} 个 token（seed=42）；子集结果不能当作复现 89.11 的依据")
for i in range(n):
    shard = dict(cfg)
    shard["tokens"] = toks[i::n]        # 交错切分，各片场景分布更均匀
    p = os.path.join(fdir, f"navtest_shard{i}of{n}.yaml")
    with open(p, "w") as f:
        yaml.safe_dump(shard, f, default_flow_style=False, sort_keys=False)
    print(f"  shard {i}: {len(shard['tokens']):,} tokens -> {os.path.basename(p)}")
print(f"合计 {len(toks):,}")
PY
    # train_test_split 层也要有对应入口
    for i in $(seq 0 $((NSHARD-1))); do
        cat > "$FILTER_DIR/../navtest_shard${i}of${NSHARD}.yaml" <<EOF
defaults:
  - scene_filter: navtest_shard${i}of${NSHARD}

data_split: test
EOF
    done
}

# ---------------------------------------------------------------- 合并结果
merge() {
    $PY - "$OUT" "$DUMP" <<'PY'
import sys, glob, os, json, re
import pandas as pd
out = sys.argv[1]
dump = sys.argv[2] if len(sys.argv) > 2 else ""
csvs = sorted(glob.glob(os.path.join(out, "shard*/**/*.csv"), recursive=True))
if not csvs:
    print("没有找到分片结果"); sys.exit(1)
dfs = [pd.read_csv(c) for c in csvs]
df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset="token")
print(f"分片文件 : {len(csvs)}")
print(f"场景总数 : {len(df):,}")
print(f"valid    : {int(df['valid'].sum()):,}")
print()
cols = ["no_at_fault_collisions","drivable_area_compliance","ego_progress",
        "time_to_collision_within_bound","comfort","driving_direction_compliance","score"]
for c in cols:
    if c in df:
        tag = "★ PDMS" if c == "score" else "  "
        print(f"{tag} {c:32s} {df[c].mean()*100:6.2f}")
merged = os.path.join(out, "navtest_pdms_merged.csv")
df.to_csv(merged, index=False)
print(f"\n合并结果 -> {merged}")
print("参照: AutoVLA 论文 navtest PDMS = 89.11")

# ---- 若开了 DUMP，把各分片的模型输出 + 分数 join 成一个文件 ----
# 转储本身是「一进程一文件」(outputs_<pid>.jsonl)，PID 随机、跨次运行无法对应，
# 且只有输出没有分数。这里合并成单文件并带上 PDMS 各分项，
# 之后 `jq 'select(.score==0)'` 就能直接捞出归零场景的模型输出。
if dump and os.path.isdir(dump):
    files = sorted(glob.glob(os.path.join(dump, "outputs_*.jsonl")))
    rows = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if rows:
        smap = df.set_index("token").to_dict("index")
        FAST_RE = re.compile(r"<think>(.*?)</think>", re.S)

        def is_fast(o):
            m = FAST_RE.search(o or "")
            th = m.group(1).strip() if m else ""
            # RFT 后的 ckpt 走 fast 时 <think> 恒为这句固定模板
            return "straightforward scenario" in th and len(th) < 120

        seen, outp = set(), os.path.join(dump, "outputs_merged.jsonl")
        n_fast = 0
        with open(outp, "w", encoding="utf-8") as fh:
            for r in rows:
                t = r.get("token")
                if t in seen:
                    continue
                seen.add(t)
                rec = dict(r)
                rec["is_fast_thinking"] = is_fast(r.get("raw_output"))
                n_fast += rec["is_fast_thinking"]
                rec.update({k: v for k, v in (smap.get(t) or {}).items()
                            if k not in ("valid",) and not str(k).startswith("Unnamed")})
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\n=== 模型输出 ({len(files)} 个分片文件 -> {len(seen)} 条) ===")
        print(f"fast thinking (无实质推理): {n_fast}  ({n_fast/len(seen)*100:.1f}%)")
        print(f"slow thinking (有 CoT)   : {len(seen)-n_fast}  ({(len(seen)-n_fast)/len(seen)*100:.1f}%)")
        print(f"输出合并 -> {outp}")
        print("  用法: jq 'select(.score==0)' " + outp)
PY
}

if [ "$MODE" = "merge" ]; then merge; exit 0; fi

rm -rf "$OUT"/shard*      # 清掉上次的分片结果，避免 merge 混入
echo "生成 $NSHARD 个分片:"
gen_shards
echo

for i in $(seq 0 $((NSHARD-1))); do
    d="$OUT/shard$i"; mkdir -p "$d"
    gpu=$i
    [ -n "$GPUS" ] && gpu=$(echo "$GPUS" | cut -d, -f$((i+1)))
    echo "  GPU $gpu -> shard $i"
    CUDA_VISIBLE_DEVICES=$gpu NAVSIM_EXP_ROOT="$d" AUTOVLA_DUMP_OUTPUT="$DUMP" AUTOVLA_FORCE_COT="${AUTOVLA_FORCE_COT:-}" AUTOVLA_SAMPLE_TEMP="${AUTOVLA_SAMPLE_TEMP:-}" AUTOVLA_SAMPLE_TOP_P="${AUTOVLA_SAMPLE_TOP_P:-}" \
    $PY "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_cot.py" \
        train_test_split="navtest_shard${i}of${NSHARD}" \
        agent=autovla_agent \
        agent.config_path="$EVAL_CONFIG" \
        agent.checkpoint_path="$CHECKPOINT" \
        agent.sensor_data_path="$SENSOR_DATA_PATH" \
        agent.lora_conf.use_lora=$LORA \
        metric_cache_path="$CACHE_PATH" \
        +json_data_path="$JSON_DATA_PATH" \
        experiment_name="shard$i" \
        > "$d/run.log" 2>&1 &
done

echo
echo "$NSHARD 个分片已启动。等待全部完成..."
wait
echo
echo "=== 合并 ==="
merge
