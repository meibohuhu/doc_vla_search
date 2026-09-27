#!/bin/bash
# Case study:同一场景连续 3 帧，好/差两个 CoT checkpoint 对比推理。
#   好 ckpt = 103k CoT ep4 (PDMS 80.45, 全量 navtest)   runs/sft/2026-09-02_05-47-39_bf16_cot_103k
#   差 ckpt = 166k CoT ep2 (PDMS 74.65, 1000 子集)      runs/sft/2026-09-03_20-56-46
#   两个都是 CoT 训练 -> 都能吐 <think>，可直接对比 reasoning。
#
# 场景 2021.05.25.14.16.10_veh-35_00083_00485 的 i=656/657/658（0.5s 间隔）：
#   自车 2.0m/s 近乎停住，周围 17 车 + 17 行人 + 41 锥桶 + 施工标志，GT 5s 只前进 3.2m（该让行）。
#   已知 i=657/658 上 best=1.00 / worse=0.00（worse 撞车）。
#
# 借 agent 里的 AUTOVLA_DUMP_OUTPUT 落盘每帧的原始生成文本(含 <think>)+ 预测轨迹。
#   bash scripts/0914/run_case_study_3frames.sh
set -uo pipefail

REPO=/home/nvidia/workspace/other_repo/AutoVLA
DATA=/data/autovla_data/nuplan
PY=/data/autovla_data/envs/autovla/bin/python

# --- case 注册表:CASE=<名字> 选场景。默认 leadstop = 最早做的那个，勿改（会覆盖已有输出）---
CASE="${CASE:-leadstop}"
case "$CASE" in
  leadstop)            # 跟停慢车:差 ckpt 推理翻转成 KEEP -> 追尾
    LOG_NAME=2021.05.25.14.16.10_veh-35_00083_00485
    TOKENS=(135ae32b6edc55c5 974a70027f8f593c ed98a4566ea95092)          # i=656,657,658
    TAG=casestudy3f ;;
  leftturn_ped)        # 无保护左转 + 9 行人:差 ckpt 冲出可行驶区域
    LOG_NAME=2021.06.28.16.57.59_veh-26_00016_00484
    TOKENS=(260f5d5245015db6 abb9477dd3305951 2a94741039ad566d)          # i=130,131,132
    TAG=casestudy3f_leftturn_ped ;;
  leftturn_ped129)     # 同上场景，但往前挪一帧:129/130/131，看差 ckpt 从“不动”到“切出去”的全过程
    LOG_NAME=2021.06.28.16.57.59_veh-26_00016_00484
    TOKENS=(c672f1584cb75697 260f5d5245015db6 abb9477dd3305951)          # i=129,130,131
    TAG=casestudy3f_leftturn_ped129 ;;
  leftturn_ped129_132) # 129/130/132（跳过 131）
    LOG_NAME=2021.06.28.16.57.59_veh-26_00016_00484
    TOKENS=(c672f1584cb75697 260f5d5245015db6 2a94741039ad566d)          # i=129,130,132
    TAG=casestudy3f_leftturn_ped129_132 ;;
  leftturn_ped_seq)    # 129-134 六帧全跑，方便事后任选三帧组图(不必重复推理)
    LOG_NAME=2021.06.28.16.57.59_veh-26_00016_00484
    TOKENS=(c672f1584cb75697 260f5d5245015db6 abb9477dd3305951 \
            2a94741039ad566d adc1a3a3dd1c501e b88e5601e3055bd8)      # i=129..134
    TAG=casestudy3f_leftturn_ped_seq ;;
  rightturn_offroad)   # 路口右转:差 ckpt 转弯几何错 -> 冲出可行驶区域
    # 注意:用 549/550/551，不要往后取 —— i=552/553 上好 ckpt 自己也掉 0 分。
    LOG_NAME=2021.06.03.13.55.17_veh-35_02572_02855
    TOKENS=(c520f76d99f359f2 154d0d1b363f5501 655c3f17ee2d5683)          # i=549,550,551
    TAG=casestudy3f_rightturn_offroad ;;
  *) echo "❌ 未知 CASE=$CASE (可选: leadstop / leftturn_ped / leftturn_ped129 / leftturn_ped129_132 / leftturn_ped_seq / rightturn_offroad)"; exit 1 ;;
esac

CKPT_BEST="$REPO/runs/sft/2026-09-02_05-47-39_bf16_cot_103k/epoch=4-loss=0.4436.ckpt"
CKPT_WORSE="$REPO/runs/sft/2026-09-03_20-56-46/epoch=2-loss=0.3617.ckpt"
GPU_BEST=${GPU_BEST:-4}
GPU_WORSE=${GPU_WORSE:-5}

export PYTHONPATH="$REPO/navsim:${PYTHONPATH:-}"
export NAVSIM_DEVKIT_ROOT="$REPO/navsim"
export OPENSCENE_DATA_ROOT="$DATA"
export NUPLAN_MAPS_ROOT="$DATA/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export TOKENIZERS_PARALLELISM=false
export AUTOVLA_SAMPLE_TOP_K=1          # greedy，与 PDMS 评测口径一致

CACHE_PATH="$DATA/navtest_metric_cache"
JSON_DATA_PATH="$DATA/navtest_nocot"
SENSOR_DATA_PATH="$DATA/sensor_blobs/test"
EVAL_CONFIG="$REPO/config/eval/0721/autovla-navtest-eval.yaml"    # use_cot: true
FILTER_DIR="$REPO/navsim/navsim/planning/script/config/common/train_test_split/scene_filter"
WORK="$DATA/../eval/nuplan/case_study/${TAG}"
mkdir -p "$WORK"
cd "$REPO"

for f in "$CKPT_BEST" "$CKPT_WORSE" "$CACHE_PATH" "$JSON_DATA_PATH" "$SENSOR_DATA_PATH" "$EVAL_CONFIG"; do
  [ -e "$f" ] || { echo "❌ 缺失: $f"; exit 1; }
done

# --- 3-token scene_filter ---
TOKENS_CSV=$(IFS=,; echo "${TOKENS[*]}") LOG_NAME="$LOG_NAME" TAG="$TAG" FDIR="$FILTER_DIR" $PY - <<'PY'
import os, yaml
fdir, tag = os.environ["FDIR"], os.environ["TAG"]
cfg = yaml.safe_load(open(os.path.join(fdir, "navtest.yaml")))
cfg["tokens"] = os.environ["TOKENS_CSV"].split(",")
cfg["log_names"] = [os.environ["LOG_NAME"]]
with open(os.path.join(fdir, f"{tag}.yaml"), "w") as f:
    yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
print(f"  scene_filter: {len(cfg['tokens'])} tokens, log={cfg['log_names'][0]}")
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
      experiment_name="cs_$name" \
      > "$d/run.log" 2>&1
  echo "  [$name] exit=$? -> $d"
}

echo "=================================================================="
echo " Case study [$CASE]: $LOG_NAME  tokens=${TOKENS[*]}"
echo " best  : $(basename "$CKPT_BEST")   GPU$GPU_BEST"
echo " worse : $(basename "$CKPT_WORSE")  GPU$GPU_WORSE"
echo " 输出  : $WORK"
echo "=================================================================="

run_one best  "$CKPT_BEST"  "$GPU_BEST"  &
run_one worse "$CKPT_WORSE" "$GPU_WORSE" &
wait

echo; echo "==================== 每帧分数 ===================="
$PY - "$WORK" <<'PY'
import sys, glob, os, json
import pandas as pd
work = sys.argv[1]
for name in ("best", "worse"):
    csvs = glob.glob(os.path.join(work, name, "**", "*.csv"), recursive=True)
    if not csvs:
        print(f"{name}: ⚠️ 无 CSV，看 {work}/{name}/run.log"); continue
    df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    df = df[df.token.notna()].drop_duplicates("token")
    print(f"\n--- {name} ---")
    print(df[["token","score","no_at_fault_collisions","drivable_area_compliance",
              "ego_progress","time_to_collision_within_bound","comfort",
              "driving_direction_compliance"]].to_string(index=False))
    n = sum(1 for f in glob.glob(os.path.join(work, name, "dump", "*.jsonl")) for _ in open(f))
    print(f"    dump 行数: {n}")
PY
