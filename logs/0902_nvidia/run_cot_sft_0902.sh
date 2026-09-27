#!/bin/bash
# CoT SFT on nuPlan navtrain (~101k)，ViT FROZEN + LLM 全参。
#
#   tmux new -s cotsft         # 断线也能活
#   bash logs/0902/run_cot_sft_0902.sh
#   GPUS=0,1,2,3 bash logs/0902/run_cot_sft_0902.sh      # 换卡
#
# 和 no-CoT 基线(logs/0722_nvidia/step3_run_vit_frozen_0727_100k_brev_gpu4.sh)
# 的差别只有 config 和数据目录；超参逐字段对齐，好让两次跑可比。
# 采样机制见 docs/0901/sft_cot_sampling.md。
set -u

REPO=/home/nvidia/workspace/doc_drive_search/other_repo/AutoVLA
PY=/data/autovla_data/envs/autovla/bin/python
CONFIG="training/qwen2.5-vl-3B-nuplan-cot-sft-navtrain-brev"
TAG="navtrain_cot_sft_0902"

# 🔴 本机 GPU0 已坏，CUDA 只有 0-6 共 7 张。
#    但 global batch 必须 = 32（对齐 no-CoT 基线），7 除不尽 32，
#    所以默认用 4 卡 x accum 8。要用 7 卡就得接受 gb=35 或 28，
#    那样 CoT/no-CoT 的比较里会混进一个 batch size 的混杂因子。
GPUS="${GPUS:-0,1,2,3}"

cd "$REPO"

# --- preflight 1: 数据 ---
TRAIN_DIR=/data/autovla_data/nuplan/navtrain_cot
VAL_DIR=/data/autovla_data/nuplan/navtrain_cot_val
# 用 find 数，不用 `ls dir/*.json`：~101k 文件会撑爆 ARG_MAX，
# "Argument list too long" 会被静默当成 0。
count_json() { find "$1" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l; }
n_train=$(count_json "$TRAIN_DIR")
n_val=$(count_json "$VAL_DIR")
[ "$n_train" -eq 0 ] && { echo "ERROR: $TRAIN_DIR 为空 -- 先跑 merge_cot_into_samples.py"; exit 1; }
[ "$n_val" -eq 0 ]   && { echo "ERROR: $VAL_DIR 为空 -- 先跑 merge_cot_into_samples.py"; exit 1; }
[ -e "$REPO/Qwen2.5-VL-3B-Instruct" ] || { echo "ERROR: 找不到 Qwen2.5-VL-3B-Instruct"; exit 1; }

# --- preflight 2: cot_output 真的填进去了 ---
# 只在 merge 漏跑时才会触发。cot_output 不是【非空字符串】的话，
# sft_dataset.py 会静默把整条样本当成 act_directly —— 训练照跑，但 CoT 一条都没学。
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

# --- preflight 3: global batch 必须 = 32，和 no-CoT 基线一致 ---
n_gpu=$(awk -F, '{print NF}' <<< "$GPUS")
accum=$(grep -E '^\s*accumulate_grad_batches:' "config/${CONFIG}.yaml" | awk '{print $2}')
bs=$(grep -E '^\s*batch_size:' "config/${CONFIG}.yaml" | head -1 | awk '{print $2}')
gb=$((bs * accum * n_gpu))
if [ "$gb" -ne 32 ]; then
  echo "ERROR: global batch = ${bs} x ${accum} accum x ${n_gpu} GPU = ${gb}，应为 32。"
  echo "       改 GPUS 就必须同步改 config 里的 accumulate_grad_batches，"
  echo "       否则 CoT/no-CoT 的对比里混进 batch size 这个混杂因子。"
  echo "       要故意跑别的 batch size，把这段检查注释掉并在 wandb run name 里写明。"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPUS"
export TOKENIZERS_PARALLELISM=false
# 🔴 不加这个，终端/日志会看起来"卡住不动"：
#    进度条用 \r 刷新、不带 \n，而管道里的 stderr 是行缓冲 —— 没有换行就永远不 flush，
#    于是画面停在某一步(实测停在 batch 63),但训练其实一直在跑。
#    判断死活别看这个日志，看 nvidia-smi / ps / wandb。
export PYTHONUNBUFFERED=1
# 想肉眼看喂进去的样本长什么样:PRINT_SFT_DEBUG=2 bash logs/0902/run_cot_sft_0902.sh
# 每个 dataloader worker 打印前 N 条（总量 = N x num_workers x GPU 数），只打印不改行为。
export PRINT_SFT_DEBUG="${PRINT_SFT_DEBUG:-0}"
# A100-SXM4 全 NVLink(NV12)，保持 NCCL 默认 —— 禁 P2P/SHM 反而绕开 NVLink 走慢路径
export NUPLAN_MAPS_ROOT=/data/autovla_data/nuplan/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export OPENSCENE_DATA_ROOT=/data/autovla_data/nuplan
export NAVSIM_DEVKIT_ROOT="$REPO/navsim"
export WANDB_PROJECT=autovla-cot-sft

# --- 采样比例从 config 算出来，不要写死 ---
# (之前这里是硬编码的 61.5/13.5/25，改了 config 也不会变，读 log 的人会被骗)
cr=$(grep -E '^\s*cot_ratio:' "config/${CONFIG}.yaml" | awk '{print $2}')
cip=$(grep -E '^\s*cot_in_prompt_ratio:' "config/${CONFIG}.yaml" | awk '{print $2}')
read -r r_rta r_rg r_ad < <(awk -v c="$cr" -v p="$cip" 'BEGIN{printf "%.1f %.1f %.1f", 100*c*(1-p), 100*c*p, 100*(1-c)}')

STAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/0902/${TAG}_${STAMP}.log"

echo "=================================================================="
echo " CoT SFT : $TAG   (ViT frozen, LLM full-param)"
echo " config  : config/${CONFIG}.yaml"
echo " GPUs    : $GPUS  ($n_gpu 张, DDP, NVLink)"
echo " train   : $n_train json  (cot_output 覆盖率 ${cov}%)"
echo " val     : $n_val json"
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
