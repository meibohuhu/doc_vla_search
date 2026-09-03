#!/bin/bash
# CoT SFT on 全量 nuPlan trainval【166k】(train 164,282 / val 2,000)，ViT FROZEN + LLM 全参。
# AutoVLA Table S1 口径(全 1310 log, fi=4)。CoT 为 rule-based(<PLAN> 速度/路径,零 VLM)。
#
#   tmux new -s cot166k        # 断线也能活
#   bash logs/0902/run_cot_sft_166k_0902.sh
#   GPUS=0,1,2,3 bash logs/0902/run_cot_sft_166k_0902.sh      # 换卡
#
# 和 103k CoT 版(logs/0902/run_cot_sft_0902.sh)的差别只有 config 和数据目录；
# 超参逐字段对齐,好让 103k/166k 可比。数据构建/校验见 docs/0901/*、logs/0901_nvidia/。
# 采样机制见 docs/0901/sft_cot_sampling.md。
set -u

REPO=/home/nvidia/workspace/doc_drive_search/other_repo/AutoVLA
PY=/data/autovla_data/envs/autovla/bin/python
CONFIG="training/qwen2.5-vl-3B-nuplan-cot-sft-trainval166k-brev"
TAG="trainval166k_cot_sft_0902"

# 🔴 本机 GPU0 已坏? -> 实测 0 可用;CUDA 0-6 共 7 张。global batch 必须 = 32,
#    4 卡 x accum 8 = 32。默认用【0,4,5,6】(1/2/3 常被评测/GRPO 占)。
GPUS="${GPUS:-0,4,5,6}"

cd "$REPO"

# --- preflight 1: 数据 ---
TRAIN_DIR=/data/autovla_data/nuplan/trainval_cot_166k
VAL_DIR=/data/autovla_data/nuplan/trainval_cot_166k_val
# 用 find 数,不用 `ls dir/*.json`:~164k 文件会撑爆 ARG_MAX,
# "Argument list too long" 会被静默当成 0。
count_json() { find "$1" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l; }
n_train=$(count_json "$TRAIN_DIR")
n_val=$(count_json "$VAL_DIR")
[ "$n_train" -eq 0 ] && { echo "ERROR: $TRAIN_DIR 为空 -- 先跑 build_navsim_cot + merge_cot_into_samples"; exit 1; }
[ "$n_val" -eq 0 ]   && { echo "ERROR: $VAL_DIR 为空 -- 先从 166k 切 val"; exit 1; }
[ -e "$REPO/Qwen2.5-VL-3B-Instruct" ] || { echo "ERROR: 找不到 Qwen2.5-VL-3B-Instruct"; exit 1; }

# --- preflight 2: cot_output 真的填进去了 ---
# 只在 merge 漏跑时才会触发。cot_output 不是【非空字符串】的话,
# sft_dataset.py 会静默把整条样本当成 act_directly —— 训练照跑,但 CoT 一条都没学。
cov=$($PY - "$TRAIN_DIR" << 'PYEOF'
import json, os, random, sys
d = sys.argv[1]
names = os.listdir(d); random.seed(0)
names = random.sample(names, min(500, len(names)))
ok = sum(1 for n in names
         if isinstance(json.load(open(os.path.join(d, n))).get("cot_output"), str)
         and json.load(open(os.path.join(d, n)))["cot_output"])
print(int(100 * ok / len(names)))
PYEOF
)
[ "$cov" -lt 90 ] && { echo "ERROR: $TRAIN_DIR 里只有 ${cov}% 的样本带 cot_output（期望 ~99%）"; exit 1; }

# --- preflight 3: 抽一条确认 camera 图真实存在（sensor 帧级完整）---
S=$(find "$TRAIN_DIR" -maxdepth 1 -name '*.json' | head -1)
$PY -c "
import json,os
d=json.load(open('$S'));
for c in ('front_camera_paths','front_left_camera_paths','front_right_camera_paths'):
    for p in d[c][:4]:
        assert os.path.exists(p), f'camera 图不在盘: {p}'
print('  preflight: camera 图帧级完整 OK')
" || { echo "ERROR: camera 图缺失 -- 检查 sensor_blobs/trainval"; exit 1; }

# --- preflight 4: global batch 必须 = 32，和 103k / no-CoT 基线一致 ---
n_gpu=$(awk -F, '{print NF}' <<< "$GPUS")
accum=$(grep -E '^\s*accumulate_grad_batches:' "config/${CONFIG}.yaml" | awk '{print $2}')
bs=$(grep -E '^\s*batch_size:' "config/${CONFIG}.yaml" | head -1 | awk '{print $2}')
gb=$((bs * accum * n_gpu))
if [ "$gb" -ne 32 ]; then
  echo "ERROR: global batch = ${bs} x ${accum} accum x ${n_gpu} GPU = ${gb}，应为 32。"
  echo "       改 GPUS 就必须同步改 config 里的 accumulate_grad_batches。"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPUS"
export TOKENIZERS_PARALLELISM=false
# 🔴 不加这个,终端/日志会看起来"卡住不动"(进度条用 \r 刷新,行缓冲不 flush)。
#    判断死活看 nvidia-smi / ps / wandb,别看这个日志。
export PYTHONUNBUFFERED=1
export PRINT_SFT_DEBUG="${PRINT_SFT_DEBUG:-0}"
# A100-SXM4 全 NVLink,保持 NCCL 默认
export NUPLAN_MAPS_ROOT=/data/autovla_data/nuplan/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export OPENSCENE_DATA_ROOT=/data/autovla_data/nuplan
export NAVSIM_DEVKIT_ROOT="$REPO/navsim"
export WANDB_PROJECT=autovla-cot-sft-166k

# --- 采样比例从 config 反算,不写死 ---
cr=$(grep -E '^\s*cot_ratio:' "config/${CONFIG}.yaml" | awk '{print $2}')
cip=$(grep -E '^\s*cot_in_prompt_ratio:' "config/${CONFIG}.yaml" | awk '{print $2}')
read -r r_rta r_rg r_ad < <(awk -v c="$cr" -v p="$cip" 'BEGIN{printf "%.1f %.1f %.1f", 100*c*(1-p), 100*c*p, 100*(1-c)}')

STAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/0902/${TAG}_${STAMP}.log"

echo "=================================================================="
echo " CoT SFT : $TAG   (ViT frozen, LLM full-param, 166k)"
echo " config  : config/${CONFIG}.yaml"
echo " GPUs    : $GPUS  ($n_gpu 张, DDP, NVLink)"
echo " train   : $n_train json  (cot_output 覆盖率 ${cov}%)"
echo " val     : $n_val json   (166k 内部切,无泄漏)"
echo " global batch = ${bs} x ${accum} accum x ${n_gpu} GPU = ${gb}"
echo " 采样   : reason_then_act ${r_rta}% / reasoning_given ${r_rg}% / act_directly ${r_ad}%  (cot_ratio=${cr}, cot_in_prompt_ratio=${cip})"
echo " LOG     : $LOG"
echo "=================================================================="

$PY tools/run_sft.py --config "$CONFIG" 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}

echo
echo "=================================================================="
echo " exit code : $STATUS"
echo " LOG       : $LOG"
echo " CKPT dir  : $(ls -dt "$REPO"/runs/sft/*/ 2>/dev/null | head -1)"
echo "=================================================================="
exit "$STATUS"
