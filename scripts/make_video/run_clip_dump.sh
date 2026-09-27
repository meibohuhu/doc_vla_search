#!/bin/bash
# 对 video_clips.json 里的所有帧,用两个 ckpt 各跑一次带 dump 的推理。
#
#   Ours     = runs/sft/2026-09-13_18-41-37_best_cot/epoch=4-loss=0.4064.ckpt  (navtest PDMS 84.61)
#   Baseline = runs/sft/2026-09-03_20-56-46/epoch=3-loss=0.3547.ckpt           (navtest PDMS 77.00)
#
# 两次全量 eval 都没开 AUTOVLA_DUMP_OUTPUT,所以只有 PDMS 分数,没有 <think> 文本和
# 预测轨迹。视频要画三条轨迹 + 两边 reasoning,这两样必须补跑出来。
#
# 用 AUTOVLA_SAMPLE_TOP_K=1 (greedy),与全量 PDMS 评测口径一致 —— 跑出来的分数应该
# 能对上 *_merged.csv,脚本最后会逐帧核对,对不上会打印出来。
#
#   bash scripts/make_video/run_clip_dump.sh
set -uo pipefail

REPO=/home/nvidia/workspace/other_repo/AutoVLA
DATA=/data/autovla_data/nuplan
PY=/data/autovla_data/envs/autovla/bin/python

CKPT_OURS="$REPO/runs/sft/2026-09-13_18-41-37_best_cot/epoch=4-loss=0.4064.ckpt"
CKPT_BASE="$REPO/runs/sft/2026-09-03_20-56-46/epoch=3-loss=0.3547.ckpt"
GPU_OURS=${GPU_OURS:-0}
GPU_BASE=${GPU_BASE:-1}

CLIPS="$REPO/logs/0920_nvidia/video_clips.json"
TAG=${TAG:-videoclips}
WORK="/data/autovla_data/eval/nuplan/case_study/$TAG"

export PYTHONPATH="$REPO/navsim:${PYTHONPATH:-}"
export NAVSIM_DEVKIT_ROOT="$REPO/navsim"
export OPENSCENE_DATA_ROOT="$DATA"
export NUPLAN_MAPS_ROOT="$DATA/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export TOKENIZERS_PARALLELISM=false
export AUTOVLA_SAMPLE_TOP_K=1

CACHE_PATH="$DATA/navtest_metric_cache"
JSON_DATA_PATH="$DATA/navtest_nocot"
SENSOR_DATA_PATH="$DATA/sensor_blobs/test"
EVAL_CONFIG="$REPO/config/eval/0721/autovla-navtest-eval.yaml"    # use_cot: true
FILTER_DIR="$REPO/navsim/navsim/planning/script/config/common/train_test_split/scene_filter"

mkdir -p "$WORK"
cd "$REPO"
for f in "$CKPT_OURS" "$CKPT_BASE" "$CLIPS" "$CACHE_PATH" "$JSON_DATA_PATH" \
         "$SENSOR_DATA_PATH" "$EVAL_CONFIG"; do
  [ -e "$f" ] || { echo "❌ 缺失: $f"; exit 1; }
done

# --- 把所有 clip 的 token 并成一个 scene_filter(跨 log 没问题,navsim 按 token 过滤) ---
CLIPS="$CLIPS" TAG="$TAG" FDIR="$FILTER_DIR" $PY - <<'PY'
import os, json, yaml
fdir, tag = os.environ["FDIR"], os.environ["TAG"]
clips = json.load(open(os.environ["CLIPS"]))
cfg = yaml.safe_load(open(os.path.join(fdir, "navtest.yaml")))
cfg["tokens"] = [t for c in clips for t in c["tokens"]]
cfg["log_names"] = sorted({c["log"] for c in clips})
with open(os.path.join(fdir, f"{tag}.yaml"), "w") as f:
    yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
print(f"  scene_filter: {len(cfg['tokens'])} tokens / {len(cfg['log_names'])} logs")
PY
printf 'defaults:\n  - scene_filter: %s\n\ndata_split: test\n' "$TAG" > "$FILTER_DIR/../${TAG}.yaml"

run_one () {   # $1=名字 $2=ckpt $3=gpu
  local name="$1" ckpt="$2" gpu="$3"
  local d="$WORK/$name"; mkdir -p "$d/dump"
  echo "  [$name] GPU$gpu  $(basename "$ckpt")"
  CUDA_VISIBLE_DEVICES=$gpu NAVSIM_EXP_ROOT="$d" AUTOVLA_DUMP_OUTPUT="$d/dump" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_cot.py" \
      train_test_split="$TAG" \
      agent=autovla_agent \
      agent.config_path="$EVAL_CONFIG" \
      agent.checkpoint_path="'$ckpt'" \
      agent.sensor_data_path="$SENSOR_DATA_PATH" \
      agent.lora_conf.use_lora=false \
      metric_cache_path="$CACHE_PATH" \
      +json_data_path="$JSON_DATA_PATH" \
      experiment_name="vc_$name" \
      > "$d/run.log" 2>&1
  echo "  [$name] exit=$? -> $d"
}

echo "=================================================================="
echo " video clip dump  ->  $WORK"
echo " ours : $(basename "$CKPT_OURS")   GPU$GPU_OURS"
echo " base : $(basename "$CKPT_BASE")   GPU$GPU_BASE"
echo "=================================================================="

run_one ours "$CKPT_OURS" "$GPU_OURS" &
run_one base "$CKPT_BASE" "$GPU_BASE" &
wait

echo
echo "==================== 核对 ===================="
WORK="$WORK" CLIPS="$REPO/logs/0920_nvidia/video_clips.json" $PY - <<'PY'
import os, glob, json
import pandas as pd
work, clips = os.environ["WORK"], json.load(open(os.environ["CLIPS"]))
want = {t for c in clips for t in c["tokens"]}
ref = {t: (f["pdms_o"], f["pdms_b"]) for c in clips for t, f in zip(c["tokens"], c["frames"])}
for k, name in ((0, "ours"), (1, "base")):
    dumps = {}
    for f in glob.glob(os.path.join(work, name, "dump", "*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            dumps[r["token"]] = r
    csvs = glob.glob(os.path.join(work, name, "**", "*.csv"), recursive=True)
    df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True) if csvs else pd.DataFrame()
    df = df[df.token.notna()].drop_duplicates("token").set_index("token") if len(df) else df
    miss = want - set(dumps)
    print(f"\n--- {name} ---  dump {len(dumps)}/{len(want)}"
          + (f"   ❌ 缺 {len(miss)} 个: {sorted(miss)[:5]}" if miss else "   ✅"))
    if len(df):
        bad = [(t, round(df.loc[t, "score"] * 100, 1), ref[t][k])
               for t in want & set(df.index) if abs(df.loc[t, "score"] * 100 - ref[t][k]) > 0.5]
        print(f"    PDMS 与全量 eval 不一致的帧: {len(bad)}"
              + (f"  例: {bad[:3]}" if bad else "  ✅ 全部对上"))
PY
