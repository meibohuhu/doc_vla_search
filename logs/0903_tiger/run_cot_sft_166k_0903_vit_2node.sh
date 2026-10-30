#!/bin/bash
# ============================================================================
# CoT SFT 训练通路验证 —— **2 节点** / GH200 / 10 分钟
#
#   sbatch logs/0903_tiger/run_cot_sft_166k_0903_vit_2node.sh
#   tail -f logs/0903_tiger/slurm2n-<jobid>.out
#
# 和 1 节点版 run_cot_sft_166k_0903_vit.sh 的关系:
#   1 节点版  验环境/数据/模型加载/前向反向/checkpoint（与节点数无关的部分）
#   本脚本    在**真实训练代码**里验多节点 DDP 能不能跑通
#   test_nccl_2node.sh  只验裸 NCCL（已通过: busbw ~25GB/s，all_reduce 数值正确）
#
#   裸 NCCL 通过 ≠ 训练能跑通 —— Lightning 的 DDP 包装、find_unused_parameters、
#   梯度累积下的 no_sync、DistributedSampler 分片、checkpoint 只在 rank0 写……
#   这些都只有跑真实训练代码才验得到。所以这一步不能省。
#
# ⚠️ num_nodes 走环境变量 AUTOVLA_NUM_NODES=$SLURM_NNODES，**不改 config**。
#    config 里 num_nodes 保持 1 是给 1 节点版用的；两个脚本共用一份 config，
#    各自用环境变量声明自己的节点数，互不干扰。
#    (tools/run_sft.py: env 优先，没设才退回 config)
#
# ⚠️ global batch = 1 x 8 accum x 2 GPU = **16** ≠ 基线 32，
#    这一跑只验通路，loss 曲线与基线**不可比**。
# ============================================================================
#SBATCH --job-name=cot166k-2n
#SBATCH --partition=tigris
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1           # GH200 每节点 1 卡 -> 每节点 1 个 rank
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:10:00
#SBATCH --output=logs/0903_tiger/slurm2n-%j.out
#SBATCH --error=logs/0903_tiger/slurm2n-%j.out

set -u

REPO=/home/mh2803/projects/doc_vla_search
CONFIG="training/qwen2.5-vl-3B-nuplan-cot-sft-trainval166k-brev-vit"
TAG="trainval166k_cot_sft_0903_vit_2node"
DATA=/shared/rc/llm-gen-agent/mhu/auto_vla
NUPLAN="$DATA/nuplan"

cd "$REPO" || exit 1

# --- conda（必须 activate，不能直接调 env 里的 python，见 docs/0901/aarch64_gh200_env.md）
source "$HOME/miniforge3/etc/profile.d/conda.sh" || exit 1
conda activate auto_vla || { echo "ERROR: conda activate auto_vla 失败"; exit 1; }
PY=python

# --- num_nodes 跟着 Slurm 实际分配走，不可能对不上 ---
export AUTOVLA_NUM_NODES=$SLURM_NNODES

TRAIN_DIR="$NUPLAN/trainval_cot_166k"
VAL_DIR="$NUPLAN/trainval_cot_166k_val"

# --- preflight 1: 数据 ---
# 用 find 数，不用 `ls dir/*.json`：~164k 文件会撑爆 ARG_MAX，静默变成 0。
count_json() { find "$1" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l; }
n_train=$(count_json "$TRAIN_DIR"); n_val=$(count_json "$VAL_DIR")
[ "$n_train" -eq 0 ] && { echo "ERROR: $TRAIN_DIR 为空"; exit 1; }
[ "$n_val" -eq 0 ]   && { echo "ERROR: $VAL_DIR 为空"; exit 1; }
QWEN=$($PY -c "import yaml;print(yaml.safe_load(open('config/${CONFIG}.yaml'))['model']['pretrained_model_path'])")
[ -e "$QWEN/config.json" ] || { echo "ERROR: 找不到 Qwen: $QWEN"; exit 1; }

# --- preflight 2: camera 图真实存在（JSON 存绝对路径，换机器没改写就会挂）---
$PY - "$TRAIN_DIR" <<'PYEOF' || { echo "ERROR: camera 图缺失"; exit 1; }
import json, os, sys, glob
f = sorted(glob.glob(os.path.join(sys.argv[1], "*.json")))[0]
d = json.load(open(f)); n = 0
for c in ("front_camera_paths", "front_left_camera_paths", "front_right_camera_paths"):
    for p in d[c][:4]:
        assert os.path.exists(p), f"camera 图不在盘: {p}"; n += 1
print(f"  preflight: camera 图帧级完整 OK ({n} 张实测)")
PYEOF

# --- preflight 3: global batch ---
read -r bs accum < <($PY -c "
import yaml; c=yaml.safe_load(open('config/${CONFIG}.yaml'))['training']
print(c['batch_size'], c['accumulate_grad_batches'])")
n_gpu=$(( SLURM_NNODES * SLURM_GPUS_ON_NODE ))
gb=$(( bs * accum * n_gpu ))
echo "  preflight: global batch = ${bs} x ${accum} accum x ${n_gpu} GPU = ${gb}"
[ "$gb" -ne 32 ] && echo "  ⚠️  global batch=${gb} ≠ 32（基线），本跑只验通路，与基线不可比"

# --- 环境变量 ---
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1          # 不加的话日志会看着"卡住"（进度条用 \r 刷新）
export PRINT_SFT_DEBUG="${PRINT_SFT_DEBUG:-0}"
export NUPLAN_MAPS_ROOT="$NUPLAN/maps"
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export OPENSCENE_DATA_ROOT="$NUPLAN"
export NAVSIM_DEVKIT_ROOT="$REPO/navsim"
export NAVSIM_EXP_ROOT="$REPO/navsim/exp"
export WANDB_MODE="${WANDB_MODE:-offline}"   # 验证跑不污染 wandb 项目

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
# 出通信问题时打开:
# export NCCL_DEBUG=INFO

vit=$($PY -c "
import yaml; print(yaml.safe_load(open('config/${CONFIG}.yaml'))['model']['train_vision_backbone'])")

echo "=================================================================="
echo " CoT SFT 训练通路验证（2 节点）: $TAG"
echo " jobid   : $SLURM_JOB_ID   节点: $SLURM_JOB_NODELIST"
echo " GPU     : $SLURM_NNODES 节点 x $SLURM_GPUS_ON_NODE 卡 = $n_gpu (跨节点 DDP)"
echo " master  : $MASTER_ADDR:$MASTER_PORT   AUTOVLA_NUM_NODES=$AUTOVLA_NUM_NODES"
echo " ViT     : train_vision_backbone=$vit  (false = 冻结)"
echo " train/val : $n_train / $n_val json"
echo " global batch = $gb   (基线 32)"
echo " GPU 采样 : logs/0903_tiger/gpu-${SLURM_JOB_ID}-<host>.csv (每 5s)"
echo "=================================================================="

# --- GPU 用量采样：每节点一个 nvidia-smi 后台跑 ---
# GH200 的 HBM 是 96GB（名字里的 480GB 是 Grace CPU 侧的 LPDDR5X），余量不宽裕。
GPUCSV_DIR="$REPO/logs/0903_tiger"
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c '
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
    --format=csv -l 5 > "'"$GPUCSV_DIR"'/gpu-'"$SLURM_JOB_ID"'-$(hostname -s).csv" 2>&1 &
  echo $! > /tmp/gpusmi.'"$SLURM_JOB_ID"'.pid
' &
SAMPLER=$!

srun --kill-on-bad-exit=1 $PY tools/run_sft.py --config "$CONFIG"
STATUS=$?

kill $SAMPLER 2>/dev/null
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c \
  'kill $(cat /tmp/gpusmi.'"$SLURM_JOB_ID"'.pid 2>/dev/null) 2>/dev/null; rm -f /tmp/gpusmi.'"$SLURM_JOB_ID"'.pid' 2>/dev/null
sleep 2

echo
echo "=== GPU 用量摘要（两个节点都该有数据；只有一个 = DDP 没起来）==="
for f in "$GPUCSV_DIR"/gpu-${SLURM_JOB_ID}-*.csv; do
  [ -f "$f" ] || continue
  echo "--- $(basename "$f")"
  awk -F', *' 'NR>1 && $5 ~ /MiB/ {
      gsub(/ MiB/,"",$5); gsub(/ %/,"",$3);
      n++; u+=$3; m+=$5; if($5+0>mx)mx=$5+0; if($3+0>ux)ux=$3+0
    } END {
      if(n) printf "    采样 %d 次 | GPU 利用率 均值 %.0f%% 峰值 %.0f%% | 显存 均值 %.1f GiB 峰值 %.1f GiB (共 95 GiB)\n",
                   n, u/n, ux, m/n/1024, mx/1024
      else  print "    (没有有效采样)"
    }' "$f"
done

echo
echo "=================================================================="
echo " exit code : $STATUS  (10 分钟到点被杀 -> 143/SIGTERM，属正常)"
echo "=================================================================="
exit "$STATUS"
