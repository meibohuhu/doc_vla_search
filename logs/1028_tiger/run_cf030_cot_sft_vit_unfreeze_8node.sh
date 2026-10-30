#!/bin/bash
# ============================================================================
# 反事实指令 CoT SFT【navtrain_cot 101,288 + cf_navtrain_v1】，【ViT 解冻(全参)】，
#   tigris GH200【8 节点 x 1 卡】DDP。
#
#   config = config/training/1028_tiger/qwen2.5-vl-3B-nuplan-cf030-cot-sft-navtrain-vit-unfreeze-tigris8n
#          = run_cf_cot_sft_vit_unfreeze_8node.sh 那跑只把 cf.ratio 0.25 -> 0.30，其余逐项不变（cf018 那跑是 0.18）。
#   计划：docs/1026/cf_sft_autovla_plan.md
#
# 跑在 16 节点资源池里（logs/0926_tiger/gh200_pool_16n_runbook.md），不走 sbatch:
#   本脚本在【登录节点】上做 preflight，再用 srun --jobid=<pool> 把训练起到 pool 的一组 8 节点上。
#   训练挂了只是这个 step 结束，节点还在 pool 里，重跑本脚本即可。
#
#   tmux new -s cf030
#   POOL_JOBID=214334 bash logs/1028_tiger/run_cf030_cot_sft_vit_unfreeze_8node.sh
#
#   节点: 默认 GROUP=1 = pool 第 1-8 个节点（第 9-16 个给 cf018 那跑）;  GROUP=2 换第 9-16 个;  NODELIST=gh-a-[001-008] 显式指定
#   OVERLAP=1   srun 报 "step creation temporarily disabled" 时加 --overlap
#   SMOKE=0     跳过 dataset_utils.sft_dataset 的 cf smoke test（登录节点上要跑几分钟）
#   WANDB_MODE  默认 offline (tigris 没登录 wandb)，跑完 wandb sync <save_dir>/wandb/offline-run-*
# ============================================================================
set -u

REPO=/home/mh2803/projects/doc_vla_search
CONFIG="training/1028_tiger/qwen2.5-vl-3B-nuplan-cf030-cot-sft-navtrain-vit-unfreeze-tigris8n"
TAG="navtrain_cf030_cot_vit_unfreeze_8node"
DATA=/shared/rc/llm-gen-agent/mhu/auto_vla
NUPLAN="$DATA/nuplan"

POOL_JOBID="${POOL_JOBID:-${SLURM_JOB_ID:-}}"
NNODES="${NNODES:-8}"
GROUP="${GROUP:-1}"
NODELIST="${NODELIST:-}"
OVERLAP="${OVERLAP:-0}"
SMOKE="${SMOKE:-1}"

cd "$REPO" || exit 1

# --- conda（必须 activate，见 docs/0901/aarch64_gh200_env.md）---
#   srun 默认 --export=ALL，计算节点上的 task 继承这里激活好的环境
source "$HOME/miniforge3/etc/profile.d/conda.sh" || exit 1
conda activate auto_vla || { echo "ERROR: conda activate auto_vla 失败"; exit 1; }
PY=python

# --- preflight 0: pool 在跑 + 选出这一跑的 8 个节点 + 节点没被别的 step 占着 ---
[ -n "$POOL_JOBID" ] || { echo "ERROR: 设 POOL_JOBID=<资源池 salloc 的 JOBID>（squeue --me 查）"; exit 1; }
state=$(squeue -h -j "$POOL_JOBID" -o %T 2>/dev/null)
[ "$state" = "RUNNING" ] || { echo "ERROR: pool $POOL_JOBID 状态=${state:-不存在}，不是 RUNNING（登录节点不是 tigris 时先 export SLURM_CLUSTERS=tigris）"; exit 1; }
if [ -z "$NODELIST" ]; then
  mapfile -t POOL_NODES < <(scontrol show hostnames "$(squeue -h -j "$POOL_JOBID" -o %N)")
  first=$(( (GROUP - 1) * NNODES ))
  [ $(( first + NNODES )) -le ${#POOL_NODES[@]} ] \
    || { echo "ERROR: pool 只有 ${#POOL_NODES[@]} 个节点，GROUP=$GROUP x $NNODES 超出"; exit 1; }
  NODELIST=$(IFS=,; echo "${POOL_NODES[*]:$first:$NNODES}")
fi
mapfile -t NODES < <(scontrol show hostnames "$NODELIST")
[ ${#NODES[@]} -eq "$NNODES" ] || { echo "ERROR: NODELIST 有 ${#NODES[@]} 个节点 ≠ NNODES=$NNODES"; exit 1; }
while IFS='|' read -r sid sname snodes; do
  for h in $(scontrol show hostnames "$snodes"); do
    for n in "${NODES[@]}"; do
      [ "$h" = "$n" ] && { echo "ERROR: 节点 $n 正被 step $sid ($sname) 占用 —— 换 GROUP 或先停掉它"; exit 1; }
    done
  done
done < <(squeue -h -s -j "$POOL_JOBID" -o "%i|%j|%N" | grep -vE '^[0-9]+\.(extern|batch|interactive)\|')

# --- preflight 1: optimizer=adamw8bit 时需要 bitsandbytes ---
opt=$($PY -c "import yaml;print(str(yaml.safe_load(open('config/${CONFIG}.yaml'))['training'].get('optimizer','adamw')).lower())")
if [ "$opt" != "adamw" ]; then
  $PY -c "import importlib.util,sys; sys.exit(importlib.util.find_spec('bitsandbytes') is None)" \
    || { echo "❌ optimizer=$opt 需要 bitsandbytes:  pip install bitsandbytes"; exit 1; }
fi

# --- preflight 2: 数据 (train = navtrain_cot 101k, val = navtrain_cot_val 2k, cf 索引) ---
NAVTRAIN="$NUPLAN/navtrain_cot"
VAL_DIR="$NUPLAN/navtrain_cot_val"
SENSOR="$NUPLAN/sensor_blobs"
CF_TRAIN="$NUPLAN/cf_navtrain_v1/cf_navtrain_v1_train.npz"
CF_VAL="$NUPLAN/cf_navtrain_v1/cf_navtrain_v1_val.npz"
# 用 find 数，不用 `ls dir/*.json`：十万级文件会撑爆 ARG_MAX，静默变成 0
count_json() { find "$1" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l; }
for d in "$NAVTRAIN" "$VAL_DIR"; do
  [ "$(count_json "$d")" -eq 0 ] && { echo "ERROR: $d 为空"; exit 1; }
done
for f in "$CF_TRAIN" "$CF_VAL"; do
  [ -f "$f" ] || { echo "ERROR: 找不到 cf 索引 $f"; exit 1; }
done
n_nav=$(count_json "$NAVTRAIN"); n_val=$(count_json "$VAL_DIR")
QWEN=$($PY -c "import yaml;print(yaml.safe_load(open('config/${CONFIG}.yaml'))['model']['pretrained_model_path'])")
[ -e "$QWEN/config.json" ] || { echo "ERROR: 找不到 Qwen: $QWEN"; exit 1; }

# --- preflight 3: camera 图帧级完整（navtrain_cot 存相对路径，拼 sensor_blobs）---
for d in "$NAVTRAIN" "$VAL_DIR"; do
  S=$(find "$d" -maxdepth 1 -name '*.json' | head -1)
  $PY - "$S" "$SENSOR" <<'PYEOF' || { echo "ERROR: camera 图缺失 ($d)"; exit 1; }
import json, os, sys
d = json.load(open(sys.argv[1]))
for c in ("front_camera_paths", "front_left_camera_paths", "front_right_camera_paths"):
    for p in d.get(c, [])[:4]:
        full = p if os.path.isabs(p) else os.path.join(sys.argv[2], p)
        if not os.path.exists(full):
            sys.exit(f"  camera 图不在盘: {full}")
print(f"  preflight: camera 图帧级完整 OK  ({os.path.dirname(sys.argv[1])})")
PYEOF
done

# --- preflight 4: config 确认 ViT 解冻 + CoT + cf 打开 + 路径对 + global batch=32 ---
$PY -c "
import yaml; c=yaml.safe_load(open('config/${CONFIG}.yaml'))
m, d = c['model'], c['data']
assert m['train_vision_backbone'] is True, 'ViT 未解冻'
assert m['use_cot'] is True, 'use_cot 不是 True（cf 样本要求 CoT system prompt）'
assert float(m.get('cf', {}).get('ratio', 0)) > 0, 'model.cf.ratio 没开'
assert d['train']['json_dataset_path']=='$NAVTRAIN' and d['val']['json_dataset_path']=='$VAL_DIR', '数据路径不对'
assert d['train']['cf_index']=='$CF_TRAIN' and d['val']['cf_index']=='$CF_VAL', 'cf_index 不对'
assert d['train']['sensor_data_path']=='$SENSOR', 'sensor_data_path 不对'
print(f\"  config OK: ViT 解冻 + CoT + cf ratio={m['cf']['ratio']}  optimizer=$opt\")
" || exit 1
read -r bs accum < <($PY -c "
import yaml; c=yaml.safe_load(open('config/${CONFIG}.yaml'))['training']
print(c['batch_size'], c['accumulate_grad_batches'])")
n_gpu=$NNODES                        # GH200 每节点 1 卡
gb=$(( bs * accum * n_gpu ))
[ "$gb" -ne 32 ] && { echo "ERROR: global batch = $bs x $accum x $n_gpu = $gb ≠ 32。改 NNODES 要同步改 accum。"; exit 1; }

# --- 环境变量 ---
export AUTOVLA_NUM_NODES=$NNODES     # tools/run_sft.py: env 优先于 config 的 num_nodes
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1            # 不加的话日志会看着"卡住"（进度条用 \r 刷新）
export PRINT_SFT_DEBUG="${PRINT_SFT_DEBUG:-0}"
# 解冻 ViT 显存吃紧: 变长视觉 token 造成碎片，expandable_segments 回收，不改数值不影响速度
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NUPLAN_MAPS_ROOT="$NUPLAN/maps"
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export OPENSCENE_DATA_ROOT="$NUPLAN"
export NAVSIM_DEVKIT_ROOT="$REPO/navsim"
export NAVSIM_EXP_ROOT="$REPO/navsim/exp"
export WANDB_PROJECT=autovla-cf-sft        # 与 config / 1026 那版一致
export WANDB_MODE="${WANDB_MODE:-offline}"
export MASTER_ADDR="${NODES[0]}"
export MASTER_PORT=$(( 20000 + POOL_JOBID % 10000 + GROUP * 1000 ))
# 出通信问题时打开:
# export NCCL_DEBUG=INFO

# --- preflight 5: cf smoke test（cf prompt/答案/轨迹/解耦 + DataCollator label 屏蔽，见 docs/1026 §4 C6）---
if [ "$SMOKE" = "1" ]; then
  echo "  preflight: cf smoke test (200 条，登录节点上要几分钟；SMOKE=0 跳过)"
  $PY -m dataset_utils.sft_dataset --config "$CONFIG" --n 200 2>&1 | grep -vi warning | tail -15
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "ERROR: cf smoke test 有错，见上面"; exit 1; }
fi

STAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/1028_tiger/${TAG}_${STAMP}.log"
GPUCSV_DIR="$REPO/logs/1028_tiger/gpu"
mkdir -p "$GPUCSV_DIR"

# 每节点 1 个 task；CPU/内存要显式给，从登录节点起的 step 不会自动继承 pool 的 --cpus-per-task
SRUN=(srun --jobid="$POOL_JOBID" --nodelist="$NODELIST" -N "$NNODES" -n "$NNODES" --ntasks-per-node=1
      --cpus-per-task=16 --mem=0 --gres=gpu:gh200:1 --exclusive)
[ "$OVERLAP" = "1" ] && SRUN+=(--overlap)

# --- preflight 6: 节点自检（顺便确认 step 真能在这组节点上起来；120s 起不来多半是节点被占）---
echo "  preflight: 节点自检"
check=$(timeout 120 "${SRUN[@]}" --job-name="${TAG}_check" bash -c \
  'printf "    %-10s cpus=%-3s %s\n" "$(hostname -s)" "$(nproc)" "$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"') \
  || { echo "ERROR: srun 自检失败/超时（节点被占? 试 OVERLAP=1）"; exit 1; }
sort <<< "$check"
[ "$(grep -c . <<< "$check")" -eq "$NNODES" ] || { echo "ERROR: 自检只回来 $(grep -c . <<< "$check")/$NNODES 个节点"; exit 1; }

echo "=================================================================="
echo " cf CoT SFT : $TAG   (★ViT 解冻, navtrain_cot + cf_navtrain_v1)"
echo " config  : config/${CONFIG}.yaml   optimizer=$opt"
echo " pool    : $POOL_JOBID   GROUP=$GROUP   节点: $NODELIST"
echo " GPU     : $NNODES 节点 x 1 卡 = $n_gpu (跨节点 DDP)   master: $MASTER_ADDR:$MASTER_PORT"
echo " train   : $n_nav json (navtrain_cot) + cf $(basename "$CF_TRAIN")"
echo " val     : $n_val json (navtrain_cot_val) + cf $(basename "$CF_VAL")"
echo " global batch = $bs x $accum accum x $n_gpu GPU = $gb"
echo " wandb   : $WANDB_MODE"
echo " LOG     : $LOG"
echo " GPU 采样 : $GPUCSV_DIR/gpu-${TAG}-${STAMP}-<host>.csv (每 5s)"
echo "=================================================================="

# nvidia-smi 采样放在训练同一个 task 里（不单独起 srun step，免得和训练 step 抢资源，
#   见 logs/0903_tiger/run_cot_sft_166k_0903_vit.sh 的 117975 教训）；
#   python 退出后 step 结束，Slurm 会把残留的 nvidia-smi 一起清掉。
export CONFIG TAG STAMP GPUCSV_DIR
"${SRUN[@]}" --job-name="$TAG" --kill-on-bad-exit=1 bash -c '
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
    --format=csv -l 5 > "$GPUCSV_DIR/gpu-${TAG}-${STAMP}-$(hostname -s).csv" 2>&1 &
  exec python tools/run_sft.py --config "$CONFIG"
' 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}

echo
echo "=== GPU 用量摘要（$NNODES 个节点都该有数据；少了 = 那个节点的 rank 没起来）==="
for f in "$GPUCSV_DIR"/gpu-${TAG}-${STAMP}-*.csv; do
  [ -f "$f" ] || continue
  printf -- "--- %s  " "$(basename "$f" .csv | sed 's/.*-//')"
  awk -F', *' 'NR>1 && $5 ~ /MiB/ {
      gsub(/ MiB/,"",$5); gsub(/ %/,"",$3);
      n++; u+=$3; m+=$5; if($5+0>mx)mx=$5+0; if($3+0>ux)ux=$3+0
    } END {
      if(n) printf "采样 %d 次 | GPU 利用率 均值 %.0f%% 峰值 %.0f%% | 显存 均值 %.1f GiB 峰值 %.1f GiB (共 95 GiB)\n",
                   n, u/n, ux, m/n/1024, mx/1024
      else  print "(没有有效采样)"
    }' "$f"
done

echo
echo " exit=$STATUS  LOG=$LOG"
# 每个 rank 各自取时间戳，跨秒时会打印出不同的 save_dir；只有 rank0 那个有 lightning_logs/ckpt，其余是空目录
for d in $(grep -oE '\[output\] save_dir = \S+' "$LOG" | awk '{print $4}' | sort -u); do
  [ -d "$d/lightning_logs" ] && echo " save_dir (rank0) = $d" || echo " (空目录, 可删: $d)"
done
exit "$STATUS"
