# GH200 16 节点资源池 runbook（tigris）

日期 2026-09-26 · 集群 tigris (RIT) · 登录节点 `tigris.rc.rit.edu`
参考:Tigris_GH200_MultiNode_Experiment_Guide.md（salloc 资源池 + srun 多实验）

> 目标:用 `salloc` 一次性拿 **16 × GH200**（1 节点 = 1 卡）并保留 **14 天**。
> 在这批节点上反复启动/重启训练;单个实验挂了只是那个 srun step 结束,节点还在,不用重新排队。

---

## 0. 选型结论（account / QoS / walltime）

2026-09-26 22:xx 用 `sbatch --test-only` 实测的预计开始时间（**只是估计**,会随队列变化）:

| account | QoS | 规模 | walltime | 结果 |
|---|---|---|---|---|
| llm-gen-agent | `proj_llm-gen-agent` | 8n | 任意 | ❌ `QOSGrpBillingMinutes` —— 该 QoS 的 billing 额度是 **0**,用不了 |
| llm-gen-agent | `qos_tier3` | 8n | 20d | ❌ `Invalid qos specification` |
| llm-gen-agent | `coll_golisano_college_of_computing_and_information_sciences` | 8n | 7 / 10 / 14 / 15d | 09-27 08:22 |
| llm-gen-agent | 同上 | 8n | 20d | 10-13 20:00（中间有 maintenance,20 天跨不过去,只能排到维护结束后） |
| **llm-gen-agent** | **同上** | **16n** | **14 / 15d** | **09-27 08:47** ← 采用 |
| rc-onboard | `coll_other` | 8n | 1 / 3 / 7d | 10-01 15:07 |
| rc-onboard | `coll_other` | 8n | 20d | 10-13 20:00 |

为什么用 `llm-gen-agent` 而不是 0903 脚本的账户:

- `logs/0903_tiger/*.sh` 没写 `-A`,`sacct` 显示实际跑在 **`rc-onboard` / `coll_other`** 上。
  默认账户 `rc-default` 的 MaxJobs=0,不能跑作业。
- 优先级权重（`sprio -w`）:FAIRSHARE = QOS = 60,480,000,AGE 只有 604,800 → **fair-share 决定排队**。
- fair-share（`sshare -U`）:`llm-gen-agent` **0.92**,`rc-onboard` **0.006**（0903 那几次已经用掉了）。
  → 同样的请求,rc-onboard 排在队尾,llm-gen-agent 排在前面。

---

## 1. 申请

```bash
tmux new -s gh200_pool

salloc -p tigris \
  -A llm-gen-agent \
  --qos=coll_golisano_college_of_computing_and_information_sciences \
  -N 16 --ntasks-per-node=1 \
  --gres=gpu:gh200:1 --cpus-per-task=16 --mem=200G \
  --time=14-00:00:00 \
  --job-name=autovla-pool-16n
```

- 出现 `Pending job allocation <JOBID>` 即已入队。**记下 JOBID**,`Ctrl-b d` 脱离 tmux,排队不受影响。
- `--mem=200G` 是**每节点** CPU 内存（GH200 节点 72 核 / 550G）。内层 `srun` 申请的 CPU/内存不能超过这里。
- 改 15 天:`--time=15-00:00:00`,实测预计开始时间一样。

## 2. 排队中

```bash
squeue --me --start -o "%.10i %.18j %.2t %.6D %.20S %R"
```

## 3. 排上以后

本集群 `LaunchParameters=use_interactive_step` → 排上后 tmux 里直接进入**第一个计算节点**的 shell。

```bash
scontrol show hostnames "$SLURM_JOB_NODELIST" > ~/pool_nodes.txt     # 存节点列表
scontrol show job "$SLURM_JOB_ID" | grep -E 'AllocTRES|TimeLimit'     # 确认 16 卡 / 14 天
```

然后 `Ctrl-b d` 脱离。

> ⚠️ **不要在这个 shell 里 `exit`,也不要 `scancel <JOBID>`** —— 都会立刻释放全部 16 个节点。
> ⚠️ tmux 只活在**这台登录节点**上。登录节点重启 → tmux 里的 salloc 死 → allocation 被释放。

回到这个 shell:`tmux attach -t gh200_pool`（先确认 `hostname` 是同一台登录节点）。

## 4. 在 pool 里跑实验

从登录节点直接往 pool 里起 step（不用进 tmux）:

```bash
JOBID=<JOBID>
GROUP1=$(sed -n '1,8p'  ~/pool_nodes.txt | paste -sd,)
GROUP2=$(sed -n '9,16p' ~/pool_nodes.txt | paste -sd,)

srun --jobid=$JOBID --exclusive --job-name=exp1 \
  -N 8 -n 8 --ntasks-per-node=1 --nodelist="$GROUP1" \
  bash exp1.sh > exp1.log 2>&1 &

srun --jobid=$JOBID --exclusive --job-name=exp2 \
  -N 8 -n 8 --ntasks-per-node=1 --nodelist="$GROUP2" \
  bash exp2.sh > exp2.log 2>&1 &
```

- 实验挂了 / 跑完:只是那个 step 结束,同一组节点直接再 `srun` 下一个。
- **不要**在 pool 里用 `sbatch` —— 会新建一个作业重新排队,不复用这批节点。
- 报 `step creation temporarily disabled (Requested nodes are busy)`:给 srun 加 `--overlap`（0903 碰到过这个问题）。
- 现成的 pool 版启动脚本(4 节点 x 1 卡,accum 8,global batch 32,与 brev 4 卡一致):
  `run_nocot_sft_navtrain_navfilt{bal,turn40}_vit_unfreeze_adamw8bit_4node.sh`,
  `POOL_JOBID=<JOBID> [GROUP=1..4] bash <脚本>`;GROUP k = pool 第 4k-3 ~ 4k 个节点。

## 5. 释放 & 成本

```bash
scancel <JOBID>      # 释放全部 16 个节点
```

- 占着不跑也照常计费:每节点 billing ≈ 779 / 分钟
  （`TRESBillingWeights`:GH200 570.78 + 16 CPU × 3.96 + 200G × 0.725）,
  16 节点 × 14 天 ≈ **2.5 亿 billing-min**,全部记到 `llm-gen-agent` 的 fair-share 上。
- 空占的卡别人用不了。不用了就尽早 `scancel`。

---

## 速查

| 目标 | 命令 |
|---|---|
| 看排队 / 预计开始 | `squeue --me --start` |
| 回到 pool shell | `tmux attach -t gh200_pool` |
| 看节点 | `cat ~/pool_nodes.txt` |
| 往 pool 里起实验 | `srun --jobid=<JOBID> --exclusive -N 8 -n 8 --ntasks-per-node=1 --nodelist="$GROUP1" bash exp.sh` |
| 看 step | `squeue --me -s` |
| 释放 | `scancel <JOBID>` |
