#!/bin/bash
# ============================================================================
# 跨节点 NCCL 通信最小验证  —— tigris (RIT RC) / GH200 / 2 节点 x 1 卡 / 5 min
#
#   sbatch logs/0903_tiger/test_nccl_2node.sh
#   tail -f logs/0903_tiger/nccl-<jobid>.out
#
# 为什么单独搞一个而不是直接跑训练:
#   GH200 每节点只有 1 张卡(Gres=gpu:gh200:1)，多卡必然跨节点。跨节点 NCCL
#   (走 IB/RoCE)是这台机器上最容易出问题、也最容易**静默**出问题的环节 ——
#   最坏的情况不是报错，而是两个 rank 各算各的、all-reduce 没生效，
#   loss 曲线看着完全正常，但等于两个独立的单卡实验。等 6 epoch 的正式训练
#   跑完才发现就太晚了，所以先用 5 分钟把它钉死。
#
# 这个脚本查四件事:
#   1. world_size 真的是 2，两个 rank 落在**不同主机**上
#      (只有一个 host 的话就是没跨节点，白测)
#   2. all_reduce 结果**数值正确**(sum(0..N-1)，不是"没报错"就算过)
#   3. broadcast 能把 rank0 的数据真正传到 rank1
#   4. 顺带量一下 all-reduce 带宽，判断跨节点 DDP 值不值得
# ============================================================================
#SBATCH --job-name=nccl2
#SBATCH --partition=tigris
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:05:00
#SBATCH --output=logs/0903_tiger/nccl-%j.out
#SBATCH --error=logs/0903_tiger/nccl-%j.out

set -u
cd /home/mh2803/projects/doc_vla_search || exit 1

# 必须走 conda activate（LD_LIBRARY_PATH 靠 activate 钩子，见 docs/0901/aarch64_gh200_env.md）
source "$HOME/miniforge3/etc/profile.d/conda.sh" || exit 1
conda activate auto_vla || { echo "ERROR: conda activate auto_vla 失败"; exit 1; }

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
export PYTHONUNBUFFERED=1
# 出问题时打开这两行看 NCCL 选了哪条链路
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT,NET

echo "=================================================================="
echo " NCCL 跨节点验证  job=$SLURM_JOB_ID"
echo " 节点   : $SLURM_JOB_NODELIST  ($SLURM_NNODES 节点 x $SLURM_GPUS_ON_NODE 卡)"
echo " master : $MASTER_ADDR:$MASTER_PORT"
echo "=================================================================="

srun --kill-on-bad-exit=1 python - <<'PYEOF'
import os, socket, sys, time
import torch, torch.distributed as dist

rank  = int(os.environ["SLURM_PROCID"])
world = int(os.environ["SLURM_NTASKS"])
host  = socket.gethostname()

# 每节点只有 1 张卡，所以 local rank 恒为 0
torch.cuda.set_device(0)
gpu = torch.cuda.get_device_name(0)
print(f"[rank {rank}/{world}] host={host} gpu={gpu} torch={torch.__version__} "
      f"cuda={torch.version.cuda} nccl={torch.cuda.nccl.version()}", flush=True)

os.environ["RANK"] = str(rank)
os.environ["WORLD_SIZE"] = str(world)

t0 = time.time()
dist.init_process_group(backend="nccl", init_method="env://",
                        world_size=world, rank=rank,
                        timeout=__import__("datetime").timedelta(seconds=120))
if rank == 0:
    print(f"\n[init] process group 建立成功，用时 {time.time()-t0:.1f}s", flush=True)

fail = []

# --- 1. 两个 rank 必须在不同主机上 ---------------------------------------
hosts = [None] * world
dist.all_gather_object(hosts, host)
if rank == 0:
    print(f"[检查1] world_size={dist.get_world_size()}  hosts={hosts}")
    if dist.get_world_size() != 2:
        fail.append(f"world_size={dist.get_world_size()}，期望 2")
    if len(set(hosts)) < 2:
        fail.append(f"两个 rank 在同一台机器上({hosts}) —— 没有真正跨节点")
    else:
        print(f"         OK: 确实跨 {len(set(hosts))} 台主机")

# --- 2. all_reduce 数值必须正确 ------------------------------------------
# 每个 rank 贡献自己的 rank 值，SUM 后应等于 0+1+...+(world-1)
x = torch.full((1024, 1024), float(rank), device="cuda")
dist.all_reduce(x, op=dist.ReduceOp.SUM)
expect = float(world * (world - 1) / 2)
got = x[0, 0].item()
allclose = torch.allclose(x, torch.full_like(x, expect))
if rank == 0:
    print(f"[检查2] all_reduce SUM: got={got} expect={expect} 全元素一致={allclose}")
    if not (abs(got - expect) < 1e-6 and allclose):
        fail.append(f"all_reduce 结果错误 got={got} expect={expect}")
    else:
        print("         OK: all-reduce 数值正确（说明梯度真的会被聚合）")

# --- 3. broadcast: rank0 的数据要真的传到别的 rank ------------------------
b = torch.arange(8, device="cuda", dtype=torch.float32) if rank == 0 \
    else torch.zeros(8, device="cuda")
dist.broadcast(b, src=0)
ok_b = torch.allclose(b, torch.arange(8, device="cuda", dtype=torch.float32))
flags = [None] * world
dist.all_gather_object(flags, bool(ok_b))
if rank == 0:
    print(f"[检查3] broadcast 各 rank 收到正确数据: {flags}")
    if not all(flags):
        fail.append(f"broadcast 失败: {flags}")
    else:
        print("         OK")

# --- 4. all-reduce 带宽 ---------------------------------------------------
if rank == 0:
    print("[检查4] all-reduce 带宽:")
for mb in (16, 128, 512):
    n = mb * 1024 * 1024 // 4
    t = torch.ones(n, device="cuda")
    for _ in range(3):                      # warmup
        dist.all_reduce(t)
    torch.cuda.synchronize(); dist.barrier()
    t1 = time.time()
    iters = 10
    for _ in range(iters):
        dist.all_reduce(t)
    torch.cuda.synchronize()
    dt = (time.time() - t1) / iters
    # ring all-reduce 的 busbw = size * 2*(N-1)/N / t
    busbw = (mb / 1024) * 2 * (world - 1) / world / dt
    if rank == 0:
        print(f"         {mb:4d} MB : {dt*1000:7.2f} ms/次   busbw {busbw:6.2f} GB/s")
    del t
torch.cuda.empty_cache()

dist.barrier()
if rank == 0:
    print("\n==================================================================")
    if fail:
        print(" ❌ 失败：")
        for f in fail:
            print("    -", f)
    else:
        print(" ✅ 跨节点 NCCL 全部通过 —— 多节点 DDP 可用")
        print("    (梯度 all-reduce 数值已验证，不是'没报错'就算过)")
    print("==================================================================", flush=True)
dist.destroy_process_group()
sys.exit(1 if (rank == 0 and fail) else 0)
PYEOF

STATUS=$?
echo
echo "exit code: $STATUS  (0 = 通信正常)"
exit "$STATUS"
