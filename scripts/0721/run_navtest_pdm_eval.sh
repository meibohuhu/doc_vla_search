#!/bin/bash
# ============================================================================
# AutoVLA 在 NAVSIM navtest 上跑 PDM Score
#
# 目标：先用作者发布的 ckpt 复现 PDMS ≈ 89.11。跑通即同时验证
# 环境 / 数据 / metric cache / 评测脚本四件事都对，之后自训模型的数字才有对照。
#
# 与 navsim/scripts/evaluation/run_autovla_agent_pdm_score_evaluation.sh 的区别：
#   * 原脚本 CONFIG_PATH 写成 "$./config/..."，多了个 `$`，展开后路径是错的
#   * 原脚本用 `+agent.xxx=`；这些 key 在 common/agent/autovla_agent.yaml 里已存在，
#     hydra 的 `+` 是"新增"，对已存在的 key 会报错 -> 改成直接覆盖 `agent.xxx=`
#   * 原脚本全相对路径依赖 CWD；这里全绝对路径
#   * 原脚本指向 GRPO 训练配置；这里用专门的评测配置（见 config/eval/0721/）
#
# 用法:
#   bash scripts/0721/run_navtest_pdm_eval.sh                       # 全量 12,146 场景
#   bash scripts/0721/run_navtest_pdm_eval.sh 0,1,2,3               # 指定 GPU
#   SMOKE=50 bash scripts/0721/run_navtest_pdm_eval.sh 0            # 冒烟：只跑 50 个场景
# ============================================================================
set -euo pipefail

# 向上找到含 setup.py 的仓库根，与本脚本所在层级无关
REPO="$(cd "$(dirname "$(readlink -f "$0")")" && while [ ! -f setup.py ] && [ "$PWD" != / ]; do cd ..; done; pwd)"
DATA=/data/autovla_data/nuplan
PY=/data/autovla_data/envs/autovla/bin/python

export PYTHONPATH="$REPO/navsim:${PYTHONPATH:-}"
export NAVSIM_DEVKIT_ROOT="$REPO/navsim"
export NAVSIM_EXP_ROOT="$DATA/exp"
export OPENSCENE_DATA_ROOT="$DATA"
export NUPLAN_MAPS_ROOT="$DATA/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export TOKENIZERS_PARALLELISM=false

GPUS="${1:-0}"
TRAIN_TEST_SPLIT=navtest
CHECKPOINT="/data/autovla_data/checkpoints/AutoVLA_PDMS_89.ckpt"
CACHE_PATH="$DATA/navtest_metric_cache"
JSON_DATA_PATH="$DATA/navtest_nocot"
SENSOR_DATA_PATH="$DATA/sensor_blobs/test"
EVAL_CONFIG="$REPO/config/eval/0721/autovla-navtest-eval.yaml"

# ⚠️ issue #48：发布的 ckpt 已经把 LoRA merge 进权重了。
# 这里再套一层 LoRA 会改变模块结构，load_state_dict(strict=False) 不会报错，
# 只会静默少加载一堆权重 -> 分数莫名其妙地低（有人因此只跑到 83.69）。
LORA=false

# ---- 前置检查：缺一样就直接停，不要跑到一半才发现 ----
fail=0
chk() { if [ -e "$2" ]; then echo "  ✅ $1"; else echo "  ❌ $1 缺失: $2"; fail=1; fi; }
echo "前置检查:"
chk "checkpoint"    "$CHECKPOINT"
chk "metric cache"  "$CACHE_PATH"
chk "navtest JSON"  "$JSON_DATA_PATH"
chk "sensor blobs"  "$SENSOR_DATA_PATH"
chk "eval config"   "$EVAL_CONFIG"
chk "maps"          "$NUPLAN_MAPS_ROOT"
n_json=$(ls "$JSON_DATA_PATH"/*.json 2>/dev/null | wc -l)
n_cache=$(find "$CACHE_PATH" -name metric_cache.pkl 2>/dev/null | wc -l)
echo "  JSON  : $n_json  (期望 12146)"
echo "  cache : $n_cache (期望 12146)"
[ "$n_json" -eq 12146 ] || { echo "  ❌ JSON 数量不符"; fail=1; }
[ "$n_cache" -eq 12146 ] || { echo "  ❌ metric cache 数量不符"; fail=1; }
[ "$fail" -eq 0 ] || { echo "前置检查未通过，已中止。"; exit 1; }

EXTRA=()
if [ -n "${SMOKE:-}" ]; then
    echo "冒烟模式：只跑 $SMOKE 个场景"
    EXTRA+=("train_test_split.scene_filter.max_scenes=$SMOKE")
fi

echo
echo "GPU        : $GPUS"
echo "ckpt       : $CHECKPOINT"
echo "use_cot    : true   (见 $EVAL_CONFIG)"
echo "use_lora   : $LORA  (发布 ckpt 已 merge)"
echo "exp_root   : $NAVSIM_EXP_ROOT"
echo

cd "$REPO"
CUDA_VISIBLE_DEVICES="$GPUS" $PY "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_cot.py" \
    train_test_split="$TRAIN_TEST_SPLIT" \
    agent=autovla_agent \
    agent.config_path="$EVAL_CONFIG" \
    agent.checkpoint_path="$CHECKPOINT" \
    agent.sensor_data_path="$SENSOR_DATA_PATH" \
    agent.lora_conf.use_lora=$LORA \
    metric_cache_path="$CACHE_PATH" \
    +json_data_path="$JSON_DATA_PATH" \
    experiment_name=autovla_navtest_pdms \
    "${EXTRA[@]}"
