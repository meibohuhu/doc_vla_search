"""DAC=0 到底有多严重 —— 量【出界深度】和【出界时刻】。

背景:PDMScorer 的 DAC 是二值的，而且在 PDMS 里是**乘性**的 ——
车角蹭出路沿 12cm 和撞进建筑物，得到的分数一样(0.000)。
`566bd5e6d09153e0` 就是这样:直行保速完全正确，只因为 50m 外右后角蹭出 12cm，
整帧 PDMS 归零、KEEP 被判出可行集，于是落进 chg，teacher 会教它减速。

    python logs/0902/dac_depth.py logs/0902/navtest600_greedy_G1_ep1_poses.jsonl
"""
import json, math, os, sys
sys.path.insert(0, '.'); sys.path.insert(0, './navsim')
os.environ.setdefault("NUPLAN_MAPS_ROOT", "/data/autovla_data/nuplan/maps")
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
os.environ.setdefault("OPENSCENE_DATA_ROOT", "/data/autovla_data/nuplan")
import numpy as np
from pathlib import Path
from shapely.geometry import Point
from shapely.ops import unary_union
from models.utils.feasible import FeasibleSet

EGO_L, EGO_W, RAC = 5.176, 2.297, 1.461
DT = 0.5
PDM_HORIZON = 4.0          # PDMScorer 只评前 4s


def corners(x, y, h, ex, ey, eh):
    c, s = math.cos(eh), math.sin(eh)
    gx, gy = ex + x * c - y * s, ey + x * s + y * c
    gh = eh + h
    cx, cy = gx + RAC * math.cos(gh), gy + RAC * math.sin(gh)
    hl, hw = EGO_L / 2, EGO_W / 2
    cc, ss = math.cos(gh), math.sin(gh)
    return [(cx + cc * dx - ss * dy, cy + ss * dx + cc * dy)
            for dx, dy in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]


def main():
    fs = FeasibleSet(Path("/data/autovla_data/nuplan/navtest_metric_cache"))
    rows = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
    depth, t_first, depth4 = [], [], []
    n_dac0 = 0
    for r in rows:
        x = r["rollouts"][0]
        if x["dac"] >= 1 or "poses" not in x:
            continue
        n_dac0 += 1
        mc = fs.metric_cache(r["token"])
        ego = mc.ego_state.rear_axle
        ex, ey, eh = float(ego.x), float(ego.y), float(ego.heading)
        U = unary_union(list(mc.drivable_area_map._geometries))
        P = np.asarray(x["poses"])
        dmax, tf, dmax4 = 0.0, None, 0.0
        for k in range(len(P)):
            t = DT * (k + 1)
            d = max(U.distance(Point(*c)) for c in corners(P[k, 0], P[k, 1], P[k, 2], ex, ey, eh))
            if d > 1e-6:
                dmax = max(dmax, d)
                if tf is None:
                    tf = t
                if t <= PDM_HORIZON:
                    dmax4 = max(dmax4, d)
        depth.append(dmax); depth4.append(dmax4); t_first.append(tf if tf else 9.9)

    depth = np.array(depth); depth4 = np.array(depth4); t_first = np.array(t_first)
    print(f"DAC=0 的 rollout: {n_dac0}\n")
    print("① 最大出界深度（整条 5s 轨迹）")
    for q in (10, 25, 50, 75, 90):
        print(f"   p{q:<3} = {np.percentile(depth, q):.2f} m")
    for th in (0.1, 0.3, 0.5, 1.0, 2.0):
        print(f"   ≤{th:>4} m 的占 {100*np.mean(depth <= th):5.1f}%")
    print("\n②【只看 PDM 实际评的前 4s】的最大出界深度")
    print(f"   4s 内根本没出界（=出界发生在 4~5s，PDM 评不到那里）: {100*np.mean(depth4 <= 1e-6):5.1f}%")
    d4 = depth4[depth4 > 1e-6]
    if len(d4):
        for q in (25, 50, 75, 90):
            print(f"   p{q:<3} = {np.percentile(d4, q):.2f} m")
        for th in (0.1, 0.3, 0.5, 1.0):
            print(f"   ≤{th:>4} m 的占 {100*np.mean(d4 <= th):5.1f}%  （占全部 DAC=0 的 {100*np.sum(d4<=th)/n_dac0:.1f}%）")
    print("\n③ 首次出界时刻")
    for t in (1.0, 2.0, 3.0, 4.0):
        print(f"   t ≤ {t:.0f}s 就出界: {100*np.mean(t_first <= t):5.1f}%")
    print(f"   t ≥ 4.0s 才出界（预测尾部）: {100*np.mean(t_first >= 4.0):5.1f}%")
    print("\n④ 如果给触发器加闸门,DAC=0 里还剩多少算数")
    for th, tmax in ((0.3, 3.0), (0.3, 4.0), (0.5, 3.0), (0.5, 4.0)):
        keep = np.mean((depth4 > th) & (t_first <= tmax))
        print(f"   深度>{th}m 且 首次出界≤{tmax}s : {100*keep:5.1f}%  (n={int(keep*n_dac0)})")


if __name__ == "__main__":
    main()
