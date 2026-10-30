#!/bin/bash
# ============================================================================
# 8 x GH200 资源池(7 天,ai-asl),sbatch 占位 —— 接 214334(16n,11/02 17:16 到期)。
#   同一个脚本提交两次 = 两个 8n pool(各自 POOL_JOBID)。
#   见 /home/mh2803/projects/simlingo/docs/1013_tiger/slurm_pool_sbatch_guide.md
#
# 为什么 7 天:11-10 08:00 有维护,20 天的作业要等到 11-10 20:00 才能起;
#   提交后预计 11/02 17:16(214334 到期)起,到 11-10 08:00 只剩 7 天 14 小时 → 626568、626570 按 9 天提交后 scontrol 改成了 7 天,11/09 17:16 结束。
# QOS 用 coll_golisano_...(tigris 上只有 qos_tier3 会被抢占)。
#
#   sbatch -M tigris scripts/1029/hold_pool_8n_9d.sbatch      # 提交两次
#   squeue -M tigris --me --start
#
# 排上以后:POOL_JOBID=<本作业 JOBID> OVERLAP=1 bash <8 节点启动脚本>
#   (8 节点训练一定用到第一个节点 → OVERLAP=1)
# 不用了: scancel -M tigris <JOBID>
# ============================================================================
#SBATCH --partition=tigris
#SBATCH --account=ai-asl
#SBATCH --qos=coll_golisano_college_of_computing_and_information_sciences
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=autovla-pool-8n-9d
#SBATCH --output=/shared/rc/llm-gen-agent/mhu/auto_vla/pool_logs/%x-%j.out

scontrol show hostnames "$SLURM_JOB_NODELIST" > "$HOME/pool_nodes_${SLURM_JOB_ID}.txt"
echo "[$(date)] pool $SLURM_JOB_ID 已启动: $SLURM_JOB_NODELIST"
echo "  节点列表: $HOME/pool_nodes_${SLURM_JOB_ID}.txt"
echo "  用法: POOL_JOBID=$SLURM_JOB_ID OVERLAP=1 bash <启动脚本>"

# 只占着节点,不跑任何东西;到 walltime 或 scancel 时结束
sleep infinity
