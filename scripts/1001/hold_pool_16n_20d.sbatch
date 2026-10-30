#!/bin/bash
# ============================================================================
# 16 x GH200 资源池（20 天）—— 接 205437（10/12 00:18 到期）。
#
# account 用 ai-asl：llm-gen-agent 的 fair-share 被 205437 用到 0.785，同样请求要排到 ~10/19；
#   ai-asl fair-share 0.918，--test-only 估计 10/13 20:00（维护一结束）就能起。
#   20 天的作业跨不过 10/12-13 的 maintenance，所以不会比维护结束更早；不需要 --begin。
#
# 和 logs/0926_tiger/gh200_pool_16n_runbook.md 的 salloc 方式等价，只是改用 sbatch 占位：
#   salloc 挂在登录节点的 tmux 里，要从现在 pending 到 10/13；中间有 maintenance，
#   登录节点一重启 salloc 就没了，排队也跟着丢。sbatch 的作业在 Slurm 队列里，不依赖登录节点。
#
#   sbatch scripts/1001/hold_pool_16n_20d.sbatch
#   squeue --me --start                          # 看预计开始时间
#
# 排上以后用法不变：POOL_JOBID=<本作业 JOBID> bash logs/0926_tiger/run_*.sh / scripts/1001/eval_*.sh
#   （启动脚本的节点占用检查已经排除了 .batch step）
# 不用了: scancel <JOBID>
# ============================================================================
#SBATCH --partition=tigris
#SBATCH --account=ai-asl
#SBATCH --qos=coll_golisano_college_of_computing_and_information_sciences
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=20-00:00:00
#SBATCH --job-name=autovla-pool-16n-20d
#SBATCH --output=/shared/rc/llm-gen-agent/mhu/auto_vla/pool_logs/%x-%j.out

scontrol show hostnames "$SLURM_JOB_NODELIST" > "$HOME/pool_nodes_${SLURM_JOB_ID}.txt"
echo "[$(date)] pool $SLURM_JOB_ID 已启动: $SLURM_JOB_NODELIST"
echo "  节点列表: $HOME/pool_nodes_${SLURM_JOB_ID}.txt"
echo "  用法: POOL_JOBID=$SLURM_JOB_ID bash <启动脚本>"

# 只占着节点，不跑任何东西；到 walltime 或 scancel 时结束
sleep infinity
