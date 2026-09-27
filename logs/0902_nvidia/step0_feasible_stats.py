"""第 0 步 go/no-go 的【不需要模型】那一半:可行集 D 长什么样。

base = GT 轨迹。回答三件事:
  ① |D| 的分布 —— D 要是恒等于 4 档或恒为空，teacher 就没话可说，方法直接判死
  ② GT 轨迹本身可不可行 —— 门1"GT 有时就是错的"的直接证据
  ③ GT 的 <PLAN> 说法在不在 D 里 —— 说法与可行集的错位率

⚠️ 这不是完整的第 0 步。P(学生轨迹不可行) 和 P(说法∉D | 不可行) 要等
   CoT-SFT 的 ckpt 能 rollout 才算得了。
"""
import json, os, random, sys, time
sys.path.insert(0, '.'); sys.path.insert(0, './navsim')
os.environ.setdefault("NUPLAN_MAPS_ROOT", "/data/autovla_data/nuplan/maps")
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
os.environ.setdefault("OPENSCENE_DATA_ROOT", "/data/autovla_data/nuplan")
import collections
from pathlib import Path
import numpy as np
from navsim.common.dataclasses import Trajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from models.utils.feasible import FeasibleSet, PLAN_OF_PROFILE

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
PROF_OF_PLAN = {v: k for k, v in PLAN_OF_PROFILE.items()}
SLOWER = ["HARD_BRAKE", "BRAKE"]; FASTER = ["ACCEL"]

fs = FeasibleSet(Path("/data/autovla_data/nuplan/navtest_metric_cache"))
cot = json.load(open('/data/autovla_data/nuplan/cot_navtest.json'))
d = '/data/autovla_data/nuplan/navtest_cot'
names = os.listdir(d); random.seed(17); names = random.sample(names, N)
samp = TrajectorySampling(num_poses=10, interval_length=0.5)

sizeD = collections.Counter(); n = 0
gt_infeasible = 0; say_out = 0; say_out_given_inf = 0; n_inf = 0
both_sides = 0; empty = 0; unck_any = 0
plan_vs_D = collections.Counter()
t0 = time.time()
for k, nm in enumerate(names, 1):
    tok = os.path.splitext(nm)[0]
    if tok not in cot: continue
    s = json.load(open(os.path.join(d, nm)))
    poses = np.array(s['gt_trajectory'], dtype=np.float32)
    v0 = float(np.hypot(*s['velocity'][:2]))
    out = fs.feasible_set(tok, poses, v0)
    D = out['D']
    m_gt = fs.score(tok, Trajectory(poses, samp))
    if m_gt is None: continue
    n += 1
    sizeD[len(D)] += 1
    if out['uncheckable']: unck_any += 1
    if not D: empty += 1
    if any(x in D for x in SLOWER) and any(x in D for x in FASTER): both_sides += 1
    inf = not fs.is_feasible(m_gt)
    gt_infeasible += inf; n_inf += inf
    say = PROF_OF_PLAN.get(cot[tok]['speed'])
    if say is not None:
        outside = say not in D
        say_out += outside
        if inf: say_out_given_inf += outside
        plan_vs_D[(cot[tok]['speed'], 'in D' if not outside else 'NOT in D')] += 1
    if k % 300 == 0:
        print(f"  {k}/{len(names)}  {time.time()-t0:.0f}s", flush=True)

print(f"\n{'='*66}\nn = {n}   用时 {time.time()-t0:.0f}s")
print(f"\n① |D| 分布（4 档中有几档可行）")
for sz in range(5):
    print(f"   |D|={sz}: {sizeD[sz]:5d}  {100*sizeD[sz]/n:5.1f}%")
print(f"   D 为空（→弃权）           {100*empty/n:5.1f}%")
print(f"   快慢两侧都可行（→弃权）    {100*both_sides/n:5.1f}%")
print(f"   有档不可评（路径不够长）    {100*unck_any/n:5.1f}%")
print(f"\n② GT 轨迹本身不可行(NC=0 或 DAC=0):  {100*gt_infeasible/n:5.2f}%   (n={gt_infeasible})")
print(f"\n③ GT 的 <PLAN> 说法不在 D 里:        {100*say_out/n:5.1f}%")
if n_inf:
    print(f"   其中「GT 轨迹不可行」时说法也不在 D 里: {100*say_out_given_inf/n_inf:5.1f}%  (n={n_inf})")
print(f"\n   逐档:")
for sp in ("STOP","DECELERATE","KEEP","ACCELERATE"):
    a=plan_vs_D[(sp,'in D')]; b=plan_vs_D[(sp,'NOT in D')]
    if a+b: print(f"     {sp:11s} n={a+b:5d}   在 D 里 {100*a/(a+b):5.1f}%")
