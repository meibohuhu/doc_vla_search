#!/bin/bash
# ============================================================================
# CoT SFT on 全量 nuPlan trainval【166k】(train 164,282 / val 2,000)
#   ViT FROZEN + LLM 全参。AutoVLA Table S1 口径(全 1310 log, fi=4)。
#   CoT 为 rule-based(<PLAN> 速度/路径，零 VLM)。
#
# 【冒烟测试版】tigris (RIT RC) / GH200 / Slurm / 10 分钟 / 1 卡
#
#   sbatch logs/0903_tiger/run_cot_sft_166k_0903_vit.sh
#   squeue -u $USER              # 看排队
#   tail -f logs/0903_tiger/slurm-<jobid>.out
#
# ⚠️⚠️ 这台机器每个 GH200 节点**只有 1 张卡**(Gres=gpu:gh200:1，128 个 gh-a 节点)。
#      所以「N 卡」= **N 个节点各 1 卡**，是跨节点 DDP，不存在单机多卡。
#      为此 tools/run_sft.py 的 num_nodes 从写死 1 改成读 config
#      (training.num_nodes)——不改的话多节点时 Lightning 会把每个节点当成独立的
#      单卡任务，各训各的、不做 all-reduce，loss 看着正常但等于白跑。
#
#      本冒烟测试先用 **1 节点 1 卡**：2 节点在 tigris 上排队要等 ~2 天
#      (PENDING 483 / RUNNING 247，Slurm 估 9/5 21:04 才起)，而 Slurm 要同时
#      凑齐 2 个节点才能起任务，等待明显更久。1 卡能验证环境/数据/模型加载/
#      前向反向/checkpoint，**唯独验不到跨节点 NCCL 通信** —— 那个留到正式跑
#      之前单独验一次。
#
# ⚠️ 冒烟测试的 global batch = 1 x 8 accum x 1 GPU = **8**，不是基线的 32，
#    所以**这一跑的 loss 曲线和 103k / no-CoT 基线不可比**，只用来验证能不能跑通。
#    正式跑见文件末尾说明。
# ============================================================================
#SBATCH --job-name=cot166k-smoke
#SBATCH --partition=tigris
#SBATCH --nodes=1                     # GH200 每节点 1 卡 -> 1 节点 = 1 卡
#SBATCH --ntasks-per-node=1           # 每节点 1 个 rank（对应那 1 张卡）
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=16            # num_workers=8 够用；要得少更容易被 backfill
#SBATCH --mem=128G                    # 1 卡 3B 训练用不到 200G
#SBATCH --time=00:10:00               # 冒烟测试 10 分钟，到点被杀是预期行为
#SBATCH --output=logs/0903_tiger/slurm-%j.out
#SBATCH --error=logs/0903_tiger/slurm-%j.out

set -u

REPO=/home/mh2803/projects/doc_vla_search
CONFIG="training/qwen2.5-vl-3B-nuplan-cot-sft-trainval166k-brev-vit"
TAG="trainval166k_cot_sft_0903_vit_smoke"
DATA=/shared/rc/llm-gen-agent/mhu/auto_vla
NUPLAN="$DATA/nuplan"

# 冒烟测试的期望 global batch。1 卡 = 1 x 8 accum x 1 = 8。
# 正式跑请设回 32（配平方案见文件末尾）。
EXPECT_GB="${EXPECT_GB:-8}"

cd "$REPO" || exit 1

# ---------------------------------------------------------------------------
# conda
#
# ⚠️ 必须走 conda activate，**不能**直接调 envs/auto_vla/bin/python。
#    LD_LIBRARY_PATH 是 activate 钩子设的；绕过去的话 `import torch` 会先把系统
#    /usr/lib64/libstdc++.so.6(GLIBCXX 最高 3.4.29)载进进程，之后 cv2/gdal/geos
#    (要 3.4.30~3.4.32)全部 ImportError。详见 docs/0901/aarch64_gh200_env.md
# ---------------------------------------------------------------------------
source "$HOME/miniforge3/etc/profile.d/conda.sh" || exit 1
conda activate auto_vla || { echo "ERROR: conda activate auto_vla 失败"; exit 1; }
PY=python

# ---------------------------------------------------------------------------
# preflight 1: 数据
# ---------------------------------------------------------------------------
TRAIN_DIR="$NUPLAN/trainval_cot_166k"
VAL_DIR="$NUPLAN/trainval_cot_166k_val"
# 用 find 数，不用 `ls dir/*.json`：~164k 文件会撑爆 ARG_MAX，
# "Argument list too long" 会被静默当成 0。
count_json() { find "$1" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l; }
n_train=$(count_json "$TRAIN_DIR")
n_val=$(count_json "$VAL_DIR")
[ "$n_train" -eq 0 ] && { echo "ERROR: $TRAIN_DIR 为空"; exit 1; }
[ "$n_val" -eq 0 ]   && { echo "ERROR: $VAL_DIR 为空"; exit 1; }
QWEN=$($PY -c "import yaml;print(yaml.safe_load(open('config/${CONFIG}.yaml'))['model']['pretrained_model_path'])")
[ -e "$QWEN/config.json" ] || { echo "ERROR: 找不到 Qwen: $QWEN"; exit 1; }

# ---------------------------------------------------------------------------
# preflight 2: cot_output 真的填进去了
# cot_output 不是【非空字符串】的话，sft_dataset.py 会静默把整条样本当成
# act_directly —— 训练照跑，但 CoT 一条都没学。
# ---------------------------------------------------------------------------
cov=$($PY - "$TRAIN_DIR" <<'PYEOF'
import json, os, random, sys
d = sys.argv[1]
names = os.listdir(d); random.seed(0)
names = random.sample(names, min(500, len(names)))
ok = 0
for n in names:
    c = json.load(open(os.path.join(d, n))).get("cot_output")
    if isinstance(c, str) and c:
        ok += 1
print(int(100 * ok / len(names)))
PYEOF
)
[ "$cov" -lt 90 ] && { echo "ERROR: $TRAIN_DIR 只有 ${cov}% 带 cot_output（期望 ~99%）"; exit 1; }

# ---------------------------------------------------------------------------
# preflight 3: camera 图真实存在（帧级）
#   JSON 里存的是**绝对路径**，换机器后如果没批量改写，这里就会挂。
# ---------------------------------------------------------------------------
$PY - "$TRAIN_DIR" <<'PYEOF' || { echo "ERROR: camera 图缺失 -- 检查 sensor_blobs/trainval 与 JSON 里的绝对路径"; exit 1; }
import json, os, sys, glob
f = sorted(glob.glob(os.path.join(sys.argv[1], "*.json")))[0]
d = json.load(open(f))
n = 0
for c in ("front_camera_paths", "front_left_camera_paths", "front_right_camera_paths"):
    for p in d[c][:4]:
        assert os.path.exists(p), f"camera 图不在盘: {p}"
        n += 1
print(f"  preflight: camera 图帧级完整 OK ({n} 张实测)")
PYEOF

# ---------------------------------------------------------------------------
# preflight 4: global batch 一致性
#   基线是 32。冒烟测试用 2 卡凑不出 32（1 x 8 x 2 = 16），所以这里放行 16
#   但**大声警告**。正式跑必须 EXPECT_GB=32 且把 accum/节点数配平。
# ---------------------------------------------------------------------------
read -r bs accum nn < <($PY -c "
import yaml; c=yaml.safe_load(open('config/${CONFIG}.yaml'))['training']
print(c['batch_size'], c['accumulate_grad_batches'], c.get('num_nodes',1))")
n_gpu=$(( nn * SLURM_GPUS_ON_NODE ))
[ "$nn" -ne "$SLURM_NNODES" ] && {
  echo "ERROR: config 的 num_nodes=$nn 与 sbatch 的 --nodes=$SLURM_NNODES 不一致。"
  echo "       两边必须相等，否则 Lightning 的 world_size 对不上，会卡在 rendezvous。"
  exit 1; }
gb=$(( bs * accum * n_gpu ))
if [ "$gb" -ne "$EXPECT_GB" ]; then
  echo "ERROR: global batch = ${bs} x ${accum} accum x ${n_gpu} GPU = ${gb}，期望 ${EXPECT_GB}。"
  exit 1
fi
[ "$gb" -ne 32 ] && {
  echo "⚠️⚠️  global batch = ${gb} ≠ 32（基线值）。"
  echo "     这一跑只用于冒烟测试，loss 曲线与 103k / no-CoT 基线**不可比**。"; }

# ---------------------------------------------------------------------------
# 环境变量
# ---------------------------------------------------------------------------
export TOKENIZERS_PARALLELISM=false
# 🔴 不加这个，日志会看起来"卡住不动"(进度条用 \r 刷新，行缓冲不 flush)。
export PYTHONUNBUFFERED=1
export PRINT_SFT_DEBUG="${PRINT_SFT_DEBUG:-0}"
export NUPLAN_MAPS_ROOT="$NUPLAN/maps"
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export OPENSCENE_DATA_ROOT="$NUPLAN"
export NAVSIM_DEVKIT_ROOT="$REPO/navsim"
export NAVSIM_EXP_ROOT="$REPO/navsim/exp"
export WANDB_PROJECT=autovla-cot-sft-166k
# 冒烟测试不往 wandb 传，避免污染项目（要传就把这行注掉）
export WANDB_MODE="${WANDB_MODE:-offline}"

# --- 跨节点 DDP 的 rendezvous ---
#   Lightning 自己认 Slurm 环境变量，但 MASTER_ADDR 要我们给。
#   取分配到的第一个节点当 master。
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
# 用 jobid 派生端口，避免同时跑多个任务时撞端口
export MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
# GH200 跨节点：先不锁 NCCL 传输层，让它自己选（IB/RoCE 都可能）。
# 真出通信问题再打开下面两行看日志。
# export NCCL_DEBUG=INFO
# export NCCL_SOCKET_IFNAME=^lo,docker

# --- 采样比例从 config 反算，不写死 ---
read -r cr cip < <($PY -c "
import yaml; c=yaml.safe_load(open('config/${CONFIG}.yaml'))['model']
print(c['cot_ratio'], c['cot_in_prompt_ratio'])")
read -r r_rta r_rg r_ad < <(awk -v c="$cr" -v p="$cip" 'BEGIN{printf "%.1f %.1f %.1f", 100*c*(1-p), 100*c*p, 100*(1-c)}')
vit=$($PY -c "
import yaml; print(yaml.safe_load(open('config/${CONFIG}.yaml'))['model']['train_vision_backbone'])")

echo "=================================================================="
echo " CoT SFT 冒烟测试 : $TAG"
echo " jobid    : $SLURM_JOB_ID   节点: $SLURM_JOB_NODELIST"
echo " config   : config/${CONFIG}.yaml"
echo " GPU      : $SLURM_NNODES 节点 x $SLURM_GPUS_ON_NODE 卡 = $n_gpu (GH200, 跨节点 DDP)"
echo " master   : $MASTER_ADDR:$MASTER_PORT"
echo " ViT      : train_vision_backbone=$vit  (false = 冻结)"
echo " train    : $n_train json  (cot_output 覆盖率 ${cov}%)"
echo " val      : $n_val json"
echo " global batch = ${bs} x ${accum} accum x ${n_gpu} GPU = ${gb}   (基线 32)"
echo " 采样     : reason_then_act ${r_rta}% / reasoning_given ${r_rg}% / act_directly ${r_ad}%"
echo " 时限     : 10 min（到点被 Slurm 杀掉是预期的）"
echo " GPU 采样 : logs/0903_tiger/gpu-${SLURM_JOB_ID}-<host>.csv  (每 5s)"
echo "=================================================================="

# ---------------------------------------------------------------------------
# GPU 用量采样
#   每个节点起一个后台 nvidia-smi，5 秒一行写到自己的 csv。
#   GH200 的 HBM 是 **96GB**（不是名字里的 480GB —— 那是 Grace CPU 侧的
#   LPDDR5X），所以显存余量没有想象中宽裕，值得盯着。
#   用 --loop-ms 而不是 sleep 循环，采样间隔更稳。
# ---------------------------------------------------------------------------
GPUCSV_DIR="$REPO/logs/0903_tiger"
# ⚠️ 不要用单独的 `srun` 起采样器！117975 就是这么写的，结果:
#      srun: Job 117975 step creation temporarily disabled, retrying (Requested nodes are busy)
#    这个 Slurm 不允许两个 job step 并发，采样 srun 和训练 srun 互相抢，
#    最后一个 csv 都没生成。
#    1 节点时批处理脚本本身就跑在分配到的那个节点上，直接后台起 nvidia-smi
#    即可，不占 job step。
GPUCSV="$GPUCSV_DIR/gpu-${SLURM_JOB_ID}-$(hostname -s).csv"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
           --format=csv -l 5 > "$GPUCSV" 2>&1 &
SAMPLER=$!

# srun 起 N 个 task（每节点 1 个），Lightning 通过 SLURM_PROCID 认 rank
srun --kill-on-bad-exit=1 $PY tools/run_sft.py --config "$CONFIG"
STATUS=$?

# 收掉采样器
kill $SAMPLER 2>/dev/null
sleep 2
echo
echo "=== GPU 用量摘要 ==="
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
# 按实际 exit code 分情况说明。
# 117975 时这里无脑打印"到点被杀属正常"，但那次其实是 exit 1 的真失败
# (ImportError: prompt_spec)，差点被这句话盖过去。
case "$STATUS" in
  0)        verdict="正常结束（10 分钟内跑完了，样本量小的话有可能）" ;;
  143|130)  verdict="SIGTERM/SIGINT —— 10 分钟到点被 Slurm 杀掉，冒烟测试的预期结果 ✅" ;;
  1)        verdict="❌ 真失败：Python 异常，往上翻 Traceback" ;;
  *)        verdict="❌ 非预期退出码，往上翻日志" ;;
esac
echo " exit code : $STATUS   $verdict"
# 输出根目录由 config 的 training.output_root 决定（本机指向 /shared，
# 因为 /home 只剩 76G 装不下 3 个 15GB 的 ckpt）。这里从 config 读，别写死。
OUT_ROOT=$($PY -c "
import yaml; print(yaml.safe_load(open('config/${CONFIG}.yaml'))['training'].get('output_root','runs'))" 2>/dev/null)
case "$OUT_ROOT" in /*) ;; *) OUT_ROOT="$REPO/$OUT_ROOT" ;; esac
echo " CKPT dir  : $(ls -dt "$OUT_ROOT"/sft/*/ 2>/dev/null | head -1)"
echo " (output_root = $OUT_ROOT)"
echo "=================================================================="
exit "$STATUS"

# ============================================================================
# 冒烟测试通过后，正式跑要改的：
#   1. #SBATCH --time  改成真实时长（166k x 6 epoch，按需估）
#   2. global batch 回到 32。GH200 每节点 1 卡，所以：
#        --nodes=4  + config num_nodes: 4  + accumulate_grad_batches: 8  -> 1x8x4 = 32
#      或 --nodes=2 + num_nodes: 2 + accum 16 -> 1x16x2 = 32
#      改完把本脚本的 EXPECT_GB 设回 32（或删掉那行用默认）。
#   3. WANDB_MODE 去掉 offline，让曲线进 autovla-cot-sft-166k
#   4. 想断点续训的话给 Trainer 传 ckpt_path（run_sft.py 目前没接）
# ============================================================================
