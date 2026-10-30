#!/bin/bash
# ============================================================================
# no-CoT SFT 混训【navtrain (101,288) + trainval_166k_navfilt_bal (42,473) = 143,761】，
#   【ViT 解冻(全参) + optimizer=adamw8bit】，tigris GH200【4 节点 x 1 卡】DDP。
#
#   = logs/0926_nvidia/run_nocot_sft_navtrain_navfiltbal_vit_unfreeze_adamw8bit_4gpu.sh 的 tigris 版
#     brev 4 卡 x accum 8  ->  tigris 4 节点 x 1 卡 x accum 8，global batch 都是 32，超参全部不变。
#   多节点写法沿用 logs/0903_tiger/run_cot_sft_166k_0903_vit_2node.sh
#     (srun 每节点 1 个 task、MASTER_ADDR/PORT、AUTOVLA_NUM_NODES)。
#
# 跑在 16 节点资源池里（logs/0926_tiger/gh200_pool_16n_runbook.md），不走 sbatch:
#   本脚本在【登录节点】上做 preflight，再用 srun --jobid=<pool> 把训练起到 pool 的一组 4 节点上。
#   训练挂了只是这个 step 结束，节点还在 pool 里，重跑本脚本即可。
#
#   tmux new -s navfiltbal
#   POOL_JOBID=<JOBID> bash logs/0926_tiger/run_nocot_sft_navtrain_navfiltbal_vit_unfreeze_adamw8bit_4node.sh
#
#   节点: 默认 GROUP=1 = pool 第 1-4 个节点 (16 节点 pool 可切 4 组；navfiltturn40 默认 GROUP=2)
#         GROUP=2 换第 5-8 个, GROUP=3/4 同理;  NODELIST=gh-a-[001-004] 显式指定
#   OVERLAP=1   srun 报 "step creation temporarily disabled" 时加 --overlap
#   WANDB_MODE  默认 offline (tigris 没登录 wandb)，跑完 wandb sync <save_dir>/wandb/offline-run-*
# ============================================================================
set -u

REPO=/home/mh2803/projects/doc_vla_search
CONFIG="training/0926_tiger/qwen2.5-vl-3B-nuplan-nocot-sft-navtrain-navfiltbal-vit-unfreeze-adamw8bit-tigris4n"
TAG="navtrain_navfiltbal_nocot_vit_unfreeze_adamw8bit_4node"
DATA=/shared/rc/llm-gen-agent/mhu/auto_vla
NUPLAN="$DATA/nuplan"

POOL_JOBID="${POOL_JOBID:-${SLURM_JOB_ID:-}}"
NNODES="${NNODES:-4}"
GROUP="${GROUP:-1}"
NODELIST="${NODELIST:-}"
OVERLAP="${OVERLAP:-0}"

cd "$REPO" || exit 1

# --- conda（必须 activate，见 docs/0901/aarch64_gh200_env.md）---
#   srun 默认 --export=ALL，计算节点上的 task 继承这里激活好的环境
source "$HOME/miniforge3/etc/profile.d/conda.sh" || exit 1
conda activate auto_vla || { echo "ERROR: conda activate auto_vla 失败"; exit 1; }
PY=python

# --- preflight 0: pool 在跑 + 选出这一跑的 4 个节点 + 节点没被别的 step 占着 ---
[ -n "$POOL_JOBID" ] || { echo "ERROR: 设 POOL_JOBID=<资源池 salloc 的 JOBID>（squeue --me 查）"; exit 1; }
state=$(squeue -h -j "$POOL_JOBID" -o %T 2>/dev/null)
[ "$state" = "RUNNING" ] || { echo "ERROR: pool $POOL_JOBID 状态=${state:-不存在}，不是 RUNNING"; exit 1; }
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

# --- preflight 1: adamw8bit 依赖 bitsandbytes，没装会在 configure_optimizers 崩 ---
$PY -c "import importlib.util,sys; sys.exit(importlib.util.find_spec('bitsandbytes') is None)" || {
  echo "❌ bitsandbytes 未安装，adamw8bit 无法运行。先装:  pip install bitsandbytes"
  exit 1
}

# --- preflight 2: 数据 (navtrain 101k + navfilt_bal 42k, val = navtrain_nocot_val) ---
NAVTRAIN="$NUPLAN/navtrain_nocot"
NAVFILT="$NUPLAN/trainval_166k_navfilt_bal"
VAL_DIR="$NUPLAN/navtrain_nocot_val"
# 用 find 数，不用 `ls dir/*.json`：十万级文件会撑爆 ARG_MAX，静默变成 0
count_json() { find "$1" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l; }
for d in "$NAVTRAIN" "$NAVFILT" "$VAL_DIR"; do
  [ "$(count_json "$d")" -eq 0 ] && { echo "ERROR: $d 为空（JSON 还没从 HuggingFace 下下来?）"; exit 1; }
done
n_nav=$(count_json "$NAVTRAIN"); n_add=$(count_json "$NAVFILT"); n_val=$(count_json "$VAL_DIR")
n_broken=$(find "$NAVFILT" -maxdepth 1 -xtype l | wc -l)
[ "$n_broken" -ne 0 ] && { echo "ERROR: $NAVFILT 有 $n_broken 个断链"; exit 1; }
QWEN=$($PY -c "import yaml;print(yaml.safe_load(open('config/${CONFIG}.yaml'))['model']['pretrained_model_path'])")
[ -e "$QWEN/config.json" ] || { echo "ERROR: 找不到 Qwen: $QWEN"; exit 1; }

# --- preflight 3: camera 图帧级完整（JSON 存绝对路径，brev 来的 JSON 没改写就会挂）---
for d in "$NAVTRAIN" "$NAVFILT" "$VAL_DIR"; do
  S=$(find "$d" -maxdepth 1 -name '*.json' | head -1)
  $PY - "$S" <<'PYEOF' || { echo "ERROR: camera 图缺失 ($d)"; exit 1; }
import json, os, sys
d = json.load(open(sys.argv[1]))
for c in ("front_camera_paths", "front_left_camera_paths", "front_right_camera_paths"):
    for p in d.get(c, [])[:4]:
        if not os.path.exists(p):
            hint = ("\n  ← 还是 brev 的路径：JSON 要先把 /data/autovla_data 批量改写成 "
                    "/shared/rc/llm-gen-agent/mhu/auto_vla") if p.startswith("/data/autovla_data") else ""
            sys.exit(f"  camera 图不在盘: {p}{hint}")
print(f"  preflight: camera 图帧级完整 OK  ({os.path.dirname(sys.argv[1])})")
PYEOF
done

# --- preflight 4: config 确认 vit 解冻 + adamw8bit + no-CoT + 两个 train 路径 + global batch=32 ---
$PY -c "
import yaml; c=yaml.safe_load(open('config/${CONFIG}.yaml'))
assert c['model']['train_vision_backbone'] is True, 'ViT 未解冻'
assert c['model']['use_cot'] is False, 'use_cot 不是 False'
assert str(c['training'].get('optimizer','')).lower()=='adamw8bit', 'optimizer 不是 adamw8bit'
assert c['data']['train']['json_dataset_path']==['$NAVTRAIN','$NAVFILT'], 'train 路径不是 navtrain + navfilt_bal'
assert c['data']['val']['json_dataset_path']=='$VAL_DIR', 'val 路径不是 navtrain_nocot_val'
print('  config OK: ViT 解冻 + adamw8bit + no-CoT + navtrain/navfilt_bal 混训')
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
export WANDB_PROJECT=autovla-nocot-sft-vitunfreeze
export WANDB_MODE="${WANDB_MODE:-offline}"
export MASTER_ADDR="${NODES[0]}"
export MASTER_PORT=$(( 20000 + POOL_JOBID % 10000 + GROUP * 1000 ))
# 出通信问题时打开:
# export NCCL_DEBUG=INFO

STAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG="$REPO/logs/0926_tiger/${TAG}_${STAMP}.log"
GPUCSV_DIR="$REPO/logs/0926_tiger/gpu"
mkdir -p "$GPUCSV_DIR"

# 每节点 1 个 task；CPU/内存要显式给，从登录节点起的 step 不会自动继承 pool 的 --cpus-per-task
SRUN=(srun --jobid="$POOL_JOBID" --nodelist="$NODELIST" -N "$NNODES" -n "$NNODES" --ntasks-per-node=1
      --cpus-per-task=16 --mem=0 --gres=gpu:gh200:1 --exclusive)
[ "$OVERLAP" = "1" ] && SRUN+=(--overlap)

# --- preflight 5: 节点自检（顺便确认 step 真能在这组节点上起来；120s 起不来多半是节点被占）---
echo "  preflight: 节点自检"
check=$(timeout 120 "${SRUN[@]}" --job-name="${TAG}_check" bash -c \
  'printf "    %-10s cpus=%-3s %s\n" "$(hostname -s)" "$(nproc)" "$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"') \
  || { echo "ERROR: srun 自检失败/超时（节点被占? 试 OVERLAP=1）"; exit 1; }
sort <<< "$check"
[ "$(grep -c . <<< "$check")" -eq "$NNODES" ] || { echo "ERROR: 自检只回来 $(grep -c . <<< "$check")/$NNODES 个节点"; exit 1; }

echo "=================================================================="
echo " no-CoT SFT : $TAG   (★ViT 解冻 + adamw8bit, navtrain + 166k navfilt_bal)"
echo " config  : config/${CONFIG}.yaml"
echo " pool    : $POOL_JOBID   GROUP=$GROUP   节点: $NODELIST"
echo " GPU     : $NNODES 节点 x 1 卡 = $n_gpu (跨节点 DDP)   master: $MASTER_ADDR:$MASTER_PORT"
echo " train   : $n_nav navtrain + $n_add navfilt_bal = $((n_nav + n_add)) json"
echo " val     : $n_val json (navtrain_nocot_val)"
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
echo " $(grep -m1 -oE '\[output\] save_dir = .*' "$LOG")"
exit "$STATUS"
