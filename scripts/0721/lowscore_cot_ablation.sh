#!/bin/bash
# ============================================================================
# 低分场景的 CoT 消融
#
# 流程:
#   1. 从全量 fast 评测结果里挑出 PDMS <= 阈值 的场景
#   2. 用【强制 CoT】在这些场景上重跑
#   3. 逐场景配对对比 fast vs CoT，看推理能不能救回低分场景
#
# 前置: 全量 fast 评测已完成，其合并 CSV 在
#       /data/autovla_data/nuplan/pdms_shards/navtest_pdms_merged.csv
#       （即 run_navtest_pdm_eval_sharded.sh 跑完全量后的产物）
#
# 用法:
#   bash scripts/0721/lowscore_cot_ablation.sh [阈值] [GPU数]
#   THRESH=0     -> 只跑 PDMS==0 的场景（默认）
#   THRESH=0.5   -> 跑 PDMS<=50 的场景
# ============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "$(readlink -f "$0")")" && while [ ! -f setup.py ] && [ "$PWD" != / ]; do cd ..; done; pwd)"
DATA=/data/autovla_data/nuplan
PY=/data/autovla_data/envs/autovla/bin/python
THRESH="${1:-0}"
NGPU="${2:-6}"                     # 默认用 GPU 2-7（0/1 可能在跑别的）
GPU_START="${GPU_START:-2}"

FAST_CSV="${FAST_CSV:-$DATA/pdms_shards/navtest_fast_full.csv}"
WORK="$DATA/lowscore_cot"
TOKFILE="$WORK/low_tokens.txt"

[ -f "$FAST_CSV" ] || { echo "❌ 找不到全量 fast 结果: $FAST_CSV"; echo "   先跑完 run_navtest_pdm_eval_sharded.sh 全量"; exit 1; }
mkdir -p "$WORK"

# ---- 1. 提取低分 token ----
$PY - "$FAST_CSV" "$TOKFILE" "$THRESH" <<'PY'
import sys, pandas as pd
csv, out, thr = sys.argv[1], sys.argv[2], float(sys.argv[3])
df = pd.read_csv(csv)
df = df[df.token.notna()]
low = df[df.score <= thr]
low.token.to_csv(out, index=False, header=False)
print(f"全量场景 : {len(df):,}")
print(f"PDMS<={thr}: {len(low):,} 个  ({len(low)/len(df)*100:.1f}%)")
print(f"  其中 DAC=0 : {(low.drivable_area_compliance==0).sum()}")
print(f"  其中 NC<1  : {(low.no_at_fault_collisions<1).sum()}")
print(f"token 列表 -> {out}")
PY
echo

# ---- 2. 强制 CoT 重跑这些 token ----
# 用逗号拼出 "GPU_START 起，每卡 3 进程" 的 GPU 列表
NSHARD=$((NGPU*3))
GPUS=$($PY -c "print(','.join(str($GPU_START+i//3) for i in range($NSHARD)))")
echo "强制 CoT 重跑: $NSHARD 分片 on GPU $GPU_START..$((GPU_START+NGPU-1))"
echo

rm -f "$WORK"/dump/outputs_*.jsonl 2>/dev/null
TOKENS="$TOKFILE" AUTOVLA_FORCE_COT="${FORCE_COT:-1}" AUTOVLA_SAMPLE_TEMP="${SAMPLE_TEMP:-}" AUTOVLA_SAMPLE_TOP_P="${SAMPLE_TOP_P:-}" \
    DUMP="$WORK/dump" GPUS="$GPUS" OUT_DIR="$WORK/shards" \
    bash "$REPO/scripts/0721/run_navtest_pdm_eval_sharded.sh" "$NSHARD"

# 分片结果另存，避免覆盖全量 fast 的 merged csv
cp "$WORK/shards/navtest_pdms_merged.csv" "$WORK/cot_on_lowscore.csv" 2>/dev/null || true

# ---- 3. 配对对比 ----
echo
echo "=== fast vs 强制CoT 配对对比（仅低分场景）==="
$PY - "$FAST_CSV" "$WORK/cot_on_lowscore.csv" "$WORK/dump" <<'PY'
import sys, pandas as pd, glob, json, re, os
fast = pd.read_csv(sys.argv[1]); cot = pd.read_csv(sys.argv[2])
fast = fast[fast.token.notna()]; cot = cot[cot.token.notna()]
dump = sys.argv[3]
deg = set()
for f in glob.glob(os.path.join(dump, "*.jsonl")):
    for l in open(f):
        r = json.loads(l)
        if len(re.findall(r'<action_\d+>', r.get('raw_output',''))) > 10:
            deg.add(r['token'])
m = fast.merge(cot, on='token', suffixes=('_fast','_cot'))
print(f"配对场景: {len(m)}   生成退化: {len(deg & set(m.token))}")
print(f"{'指标':<8}{'fast':>8}{'强制CoT':>9}{'差值':>8}")
print('-'*34)
for c,z in [('score','PDMS'),('ego_progress','EP'),('drivable_area_compliance','DAC'),
            ('no_at_fault_collisions','NC'),('time_to_collision_within_bound','TTC'),('comfort','C')]:
    a,b = m[f'{c}_fast'].mean()*100, m[f'{c}_cot'].mean()*100
    print(f"{z:<8}{a:8.2f}{b:9.2f}{b-a:+8.2f}")
m['d'] = m.score_cot - m.score_fast
print(f"\n低分场景里 CoT 救回(>0.01): {(m.d>0.01).sum()}   持平: {(m.d.abs()<=0.01).sum()}   更差: {(m.d<-0.01).sum()}")
救 = m[m.d>0.01].sort_values('d', ascending=False)
if len(救):
    print("\nCoT 救回的场景 (token / fast->cot):")
    for _,r in 救.head(10).iterrows():
        print(f"  {r.token}  {r.score_fast*100:5.1f} -> {r.score_cot*100:5.1f}")
PY
