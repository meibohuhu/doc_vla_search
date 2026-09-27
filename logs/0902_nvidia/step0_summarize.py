"""汇总 step0_rollout.py 的 jsonl → 第 0 步的三个数。"""
import json, sys, glob
import numpy as np
import collections

rows = []
for f in sys.argv[1:]:
    for line in open(f):
        rows.append(json.loads(line))
R = [r for r in rows if r["rollouts"]]
alls = [x for r in R for x in r["rollouts"]]
n_f, n_r = len(R), len(alls)
print(f"帧 {n_f}   rollout {n_r}   (G≈{n_r/max(n_f,1):.1f})\n")

inf = [x for x in alls if not x["feasible"]]
print("① 触发器 —— 学生轨迹不可行(NC=0 或 DAC=0)")
print(f"   逐 rollout : {100*len(inf)/n_r:5.2f}%   ({len(inf)}/{n_r})")
byf = [any(not x['feasible'] for x in r['rollouts']) for r in R]
allf = [all(not x['feasible'] for x in r['rollouts']) for r in R]
print(f"   至少一条不可行的帧: {100*np.mean(byf):5.1f}%     全部不可行的帧: {100*np.mean(allf):5.1f}%")
nc0 = sum(1 for x in alls if x['nc'] < 1); dac0 = sum(1 for x in alls if x['dac'] < 1)
print(f"   其中 碰撞 NC=0 {100*nc0/n_r:.2f}%   出可行驶区 DAC=0 {100*dac0/n_r:.2f}%")

print("\n② chg —— 不可行【且】说法也不在 D 里（才会真正产生代价）")
have = [x for x in inf if x["say_in_D"] is not None]
out = [x for x in have if not x["say_in_D"]]
if have:
    print(f"   P(说法 ∉ D | 不可行) = {100*len(out)/len(have):5.1f}%   ({len(out)}/{len(have)})")
    print(f"   → chg 占全部 rollout  = {100*len(out)/n_r:5.2f}%")
emptyD = sum(1 for x in inf if not x["D"])
print(f"   ⚠️ 不可行 rollout 里 D 为空(teacher 只能弃权): {100*emptyD/max(len(inf),1):5.1f}%")

print("\n②b 按失效原因拆开 —— 🔴 关键:横向失效用改速度救不了")
for name, sel in (("NC=0（碰撞，速度可能有救）", lambda x: x['nc'] < 1),
                  ("DAC=0（出可行驶区，横向）", lambda x: x['dac'] < 1),
                  ("只 DAC=0、NC=1", lambda x: x['dac'] < 1 and x['nc'] >= 1)):
    sub = [x for x in inf if sel(x)]
    if not sub: continue
    h = [x for x in sub if x["say_in_D"] is not None]
    o = [x for x in h if not x["say_in_D"]]
    e = sum(1 for x in sub if not x["D"])
    print(f"   {name:26s} n={len(sub):4d}  说法∉D {100*len(o)/max(len(h),1):5.1f}%  D为空 {100*e/len(sub):5.1f}%")

print("\n③ 组内 PDMS 跨度（GRPO 的 advantage 有没有信号）")
sp = [max(x['pdms'] for x in r['rollouts']) - min(x['pdms'] for x in r['rollouts']) for r in R]
sd = [float(np.std([x['pdms'] for x in r['rollouts']])) for r in R]
sp = np.array(sp); sd = np.array(sd)
print(f"   跨度 p25={np.percentile(sp,25):.3f}  p50={np.percentile(sp,50):.3f}  p75={np.percentile(sp,75):.3f}  p90={np.percentile(sp,90):.3f}")
print(f"   组内 std p50={np.percentile(sd,50):.3f}")
print(f"   ⚠️ 组内全同分(std<1e-4，advantage 恒 0、整组无梯度): {100*np.mean(sd<1e-4):5.1f}%")

print("\n④ 顺带:格式与自洽")
print(f"   off-format 率      {100*np.mean([x['fmt_viol'] for x in alls]):5.2f}%")
cons = np.array([x['consistency'] for x in alls])
print(f"   m_consistency 均值 {cons.mean():+.3f}   (+1 {100*np.mean(cons>0):.1f}% / 0 {100*np.mean(cons==0):.1f}% / −1 {100*np.mean(cons<0):.1f}%)")
c = collections.Counter(tuple(x['plan']) if x['plan'] else None for x in alls)
print("   <PLAN> 分布 top6:", c.most_common(6))
print(f"   PDMS 均值 {np.mean([x['pdms'] for x in alls]):.4f}")
