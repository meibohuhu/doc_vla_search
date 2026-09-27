#!/bin/bash
# GRPO on navtrain 5 万帧子集，6 卡 × G=4。
#
#   tmux new -s grpo
#   bash logs/0902/run_grpo_50k.sh                 # 普通 GRPO（PDMS + 格式闸门）
#   TEACHER=1 bash logs/0902/run_grpo_50k.sh       # 开 teacher（判决 + Δ 定价）
#
# 时间预算（按 3 卡 × G=5 实测 14.7 s/步 外推）:
#   G=5→4 生成成本 ×0.8 → ~11.8 s/步;6 卡 = 6 帧/步 → 50,000/6 = 8,333 步
#   ≈ 27 小时（6 rank 的 straggler 同步可能到 30–33h）
# 显存 ~25 GB/卡。LoRA 只有 3.68M 可训练参数，all-reduce 可忽略，基本线性扩展。
set -u

REPO=/home/nvidia/workspace/doc_drive_search/other_repo/AutoVLA
PY=/data/autovla_data/envs/autovla/bin/python
CONFIG="training/qwen2.5-vl-3B-nuplan-grpo-cot"
GPUS="${GPUS:-0,1,2,3,4,5}"
TEACHER="${TEACHER:-0}"

cd "$REPO"

# --- preflight 1: SFT ckpt 必须是【CoT】的且已填 ---
CK=$($PY -c "import yaml;print(yaml.safe_load(open('config/$CONFIG.yaml'))['model']['sft_model_path'])")
[ -f "$CK" ] || { echo "ERROR: sft_model_path 不存在: $CK"; echo "  → 填 CoT-SFT 收敛后的 ckpt"; exit 1; }

# --- preflight 2: metric cache 覆盖率 ---
#   查不到 cache 的 token 会让 rl_pdm_score 走异常分支、reward 记 0 ——
#   等于喂一条"最差"的假信号，而且不报错。必须先拦住。
$PY - "$CONFIG" << 'PYEOF' || exit 1
import os, sys, yaml
sys.path.insert(0, '.'); sys.path.insert(0, './navsim')
os.environ.setdefault("NUPLAN_MAPS_ROOT", "/data/autovla_data/nuplan/maps")
from pathlib import Path
from navsim.common.dataloader import MetricCacheLoader
c = yaml.safe_load(open(f"config/{sys.argv[1]}.yaml"))["data"]["train"]
d = c["json_dataset_path"]
if not os.path.isdir(c["metric_cache_path"]):
    print(f"ERROR: metric cache 目录不存在: {c['metric_cache_path']}")
    print("  → 先跑 logs/0902/build_navtrain_metric_cache.sh（默认建 navtrain50k）")
    sys.exit(1)
have = set(MetricCacheLoader(Path(c["metric_cache_path"])).tokens)
want = {os.path.splitext(n)[0] for n in os.listdir(d)}
miss = want - have
print(f"[preflight] 样本 {len(want)}   有 cache {len(want & have)}   缺 {len(miss)}")
if len(miss) > 0.02 * len(want):
    print(f"ERROR: 缺 {100*len(miss)/len(want):.1f}% 的 cache（>2%）。")
    print("  → 补建 cache，或把缺的 token 从样本目录里剔掉")
    sys.exit(1)
PYEOF

n_gpu=$(awk -F, '{print NF}' <<< "$GPUS")
export CUDA_VISIBLE_DEVICES="$GPUS"
export TOKENIZERS_PARALLELISM=false
export NUPLAN_MAPS_ROOT=/data/autovla_data/nuplan/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export OPENSCENE_DATA_ROOT=/data/autovla_data/nuplan
export NAVSIM_DEVKIT_ROOT="$REPO/navsim"

# teacher 开关:不改 config，临时生成一份
RUNCFG="$CONFIG"
if [ "$TEACHER" = "1" ]; then
    RUNCFG="training/_grpo_50k_teacher"
    $PY - << PYEOF
import yaml
c = yaml.safe_load(open("config/$CONFIG.yaml"))
c["rl"]["teacher"]["enable"] = True
c["training"]["devices"] = list(range($n_gpu))
yaml.safe_dump(c, open("config/$RUNCFG.yaml", "w"), allow_unicode=True, sort_keys=False)
PYEOF
else
    $PY - << PYEOF
import yaml
c = yaml.safe_load(open("config/$CONFIG.yaml"))
c["training"]["devices"] = list(range($n_gpu))
yaml.safe_dump(c, open("config/$CONFIG.yaml", "w"), allow_unicode=True, sort_keys=False)
PYEOF
fi

STAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/0902/grpo50k_${STAMP}.log"
echo "=================================================================="
echo " GRPO  50k × G=4 × ${n_gpu} 卡"
echo " config : config/${RUNCFG}.yaml"
echo " teacher: $([ "$TEACHER" = 1 ] && echo 开 || echo 关)"
echo " ckpt   : $CK"
echo " 预计   : ~8,333 步 × ~12 s ≈ 27 小时"
echo " LOG    : $LOG"
echo "=================================================================="

$PY tools/run_rft.py --config "$RUNCFG" 2>&1 | tee "$LOG"
echo "exit=${PIPESTATUS[0]}   CKPT: $(ls -dt "$REPO"/runs/grpo/*/ 2>/dev/null | head -1)"
