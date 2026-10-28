"""画 D6 导出索引（cf_navtrain_v1_{train,val}.npz）里的单条样本：前视图 | BEV | 速度曲线。

    python tools/cf_data/render_index.py --out docs/1026/cf_export_examples name:行号 [name:行号 ...]
    python tools/cf_data/render_index.py --out ... --pick 2      # 每种 (类别, 方向) 随机挑 2 条真反事实

标题第二行是训练时 user prompt 里的 suggestion，第三行是答案的决策句。
"""
import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lane_change import DT, N, clearance, ego_box, to_global, to_local  # noqa: E402
from speed import SpeedSynth                     # noqa: E402
from nuplan.common.maps.maps_datatypes import SemanticMapLayer  # noqa: E402

INDEX = "/scratch/mh2803/vla/nuplan/cf_navtrain_v1/cf_navtrain_v1_{}.npz"
COLORS = {"STOP": "tab:red", "DECELERATE": "tab:orange", "KEEP": "tab:green", "ACCELERATE": "tab:blue",
          "LANE_LEFT": "tab:purple", "LANE_RIGHT": "tab:brown"}


def render(z, i: int, synth: SpeedSynth, out_png: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.image import imread

    tok, cls = str(z["token"][i]), str(z["cls"][i])
    d = json.load(open(f"{synth.json_dir}/{tok}.json"))
    mc = synth.fs.metric_cache(tok)
    e = mc.ego_state.rear_axle
    ex, ey, eh = float(e.x), float(e.y), float(e.heading)
    gt, tr = np.asarray(d["gt_trajectory"]), z["traj"][i].astype(np.float64)
    _, dec = synth.tokenize(tr)

    fig = plt.figure(figsize=(19, 6.8))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.5, 0.75, 0.55, 0.9])
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(imread(d["front_camera_paths"][-1]))
    ax.axis("off")

    dm = mc.drivable_area_map
    uo = mc.observation.unique_objects

    def lanes(ax):
        for geom, typ in zip(dm._geometries, dm._map_types):
            if typ in (SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR):
                loc = to_local(np.asarray(geom.exterior.coords), ex, ey, eh)
                ax.plot(-loc[:, 1], loc[:, 0], color="0.6" if typ == SemanticMapLayer.LANE else "0.85", lw=0.7)

    def agents(ax, k):
        """observation 第 k 帧（0.1 s 一帧）。车 / 自行车紫色，其余（行人、杆子、花盆、锥桶）灰色。"""
        obs = mc.observation[k]
        for t, geom in zip(obs.tokens, obs._geometries):
            o = uo.get(t)
            mover = o is not None and o.tracked_object_type.name in ("VEHICLE", "BICYCLE")
            loc = to_local(np.asarray(geom.exterior.coords), ex, ey, eh)
            ax.fill(-loc[:, 1], loc[:, 0], color="tab:purple" if mover else "0.6", alpha=0.55 if mover else 0.4, lw=0)

    def box(ax, p, color, ls="-"):
        g = to_global(p[None, :2], ex, ey, eh)[0]
        loc = to_local(np.asarray(ego_box(g[0], g[1], eh + p[2]).exterior.coords), ex, ey, eh)
        ax.plot(-loc[:, 1], loc[:, 0], color=color, lw=1.6, ls=ls)

    # t = 0 快照 + 两条 5 s 轨迹。他车只画 t = 0：轨迹上的点是 0.5–5 s，不能和别的时刻的他车框叠在一起看
    ax = fig.add_subplot(gs[:, 1])
    lanes(ax)
    agents(ax, 0)
    # 在动的车 / 自行车：中心点 0–5 s 的走向（细线，末端圆点 = 5 s），免得把 t = 0 的框当成它一直在那里
    tracks = {}
    for k in range(0, 51, 5):
        obs = mc.observation[k]
        for t, geom in zip(obs.tokens, obs._geometries):
            if t in uo and uo[t].tracked_object_type.name in ("VEHICLE", "BICYCLE"):
                tracks.setdefault(t, []).append((geom.centroid.x, geom.centroid.y))
    for t, pts in tracks.items():
        loc = to_local(np.asarray(pts), ex, ey, eh)
        if np.linalg.norm(loc[-1] - loc[0]) > 1.0:
            ax.plot(-loc[:, 1], loc[:, 0], "-", color="tab:purple", lw=0.8, alpha=0.7)
            ax.plot(-loc[-1, 1], loc[-1, 0], "o", color="tab:purple", ms=2.5, alpha=0.7)
    ax.plot(-gt[:, 1], gt[:, 0], "o-", color="0.45", ms=3, lw=1, label=f"expert ({z['expert_speed'][i]})")
    ax.plot(-tr[:, 1], tr[:, 0], "o-", color=COLORS[cls], ms=4, lw=1.2, label=f"cf ({cls})")
    ax.plot(-dec[:, 1], dec[:, 0], "x", color="k", ms=5, label=f"decoded (err {z['token_err'][i]:.2f} m)")
    box(ax, np.zeros(3), "k")
    far = max(gt[-1, 0], tr[-1, 0])
    side = max(15.0, np.abs(np.r_[gt[:, 1], tr[:, 1]]).max() + 5)
    ax.set_xlim(-side, side)
    ax.set_ylim(-12, max(40, far + 8))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title("boxes: agents at t = 0; purple lines: where they go in 5 s\n(check gaps in the snapshots on the right)", fontsize=7)

    # 同一时刻的快照：cf 车框（彩色）、专家车框（灰虚线）、这一刻的他车
    for row, k in ((0, 4), (1, 9)):
        ax = fig.add_subplot(gs[row, 2])
        lanes(ax)
        agents(ax, 5 * (k + 1))
        box(ax, gt[k], "0.3", "--")
        box(ax, tr[k], COLORS[cls])
        gap = min((ego_box(*to_global(tr[k:k + 1, :2], ex, ey, eh)[0], eh + tr[k, 2]).distance(geom)
                   for t, geom in zip(mc.observation[5 * (k + 1)].tokens, mc.observation[5 * (k + 1)]._geometries)
                   if t in uo and uo[t].tracked_object_type.name in ("VEHICLE", "BICYCLE")), default=np.inf)
        cx, cy = -tr[k, 1], tr[k, 0]
        ax.set_xlim(cx - 11, cx + 11)
        ax.set_ylim(cy - 14, cy + 18)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"t = {0.5 * (k + 1):g} s   cf gap to vehicles {gap:.1f} m", fontsize=8)

    ax = fig.add_subplot(gs[:, 3])
    t = np.arange(N + 1) * DT
    if cls.startswith("LANE"):
        # 换道纵向照抄专家，看横向：cf 相对专家往哪边偏了多少（左为正）
        ax.plot(t[1:], tr[:, 1] - gt[:, 1], "o-", color=COLORS[cls], lw=2, label="cf − expert (lateral)")
        ax.axhline(0, color="0.45", lw=1)
        for y in (-3.5, 3.5):
            ax.axhline(y, color="0.7", ls=":", lw=0.8)
        ax.set_ylabel("lateral offset from expert (m, left +)")
        ax.set_ylim(-5, 5)
    else:
        def spd(p):
            p = np.vstack([[0.0, 0.0], np.asarray(p)[:, :2]])
            return np.linalg.norm(np.diff(p, axis=0), axis=1) / DT
        ax.step(t[1:], spd(gt), where="pre", color="0.45", lw=1.5, label="expert")
        ax.step(t[1:], spd(tr), where="pre", color=COLORS[cls], lw=2, label="cf")
        ax.axhline(2.0, color="0.7", ls=":", lw=0.8)
        ax.set_ylabel("speed (m/s, 0.5 s average)")
        ax.set_ylim(bottom=0)
    ax.set_xlabel("t (s)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    sug = str(z["suggestion"][i])
    c_cf, c_ex = clearance(mc, tr, ex, ey, eh), clearance(mc, gt, ex, ey, eh)
    fig.suptitle(f"token {tok}   v0 {z['v0'][i]:.1f} m/s   cmd '{d['instruction']}'   expert speed word: {z['expert_speed'][i]}"
                 f"   min gap to vehicles in 5 s: cf {c_cf[0]:.1f} m @{c_cf[1]:g}s, expert {c_ex[0]:.1f} m @{c_ex[1]:g}s\n"
                 f"prompt:  ... Suggested action: {sug if sug else '(none: KEEP is never a suggestion)'}.\n"
                 f"answer:  {z['decision'][i]}", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=85)
    plt.close(fig)


def pick(z, k: int, seed: int = 0):
    """每种 (类别, 方向) 随机挑 k 条真反事实。"""
    rng = random.Random(seed)
    out = []
    for cls in COLORS:
        for path in sorted(set(z["plan_path"][z["cls"] == cls].tolist())):
            idx = np.where((z["cls"] == cls) & (z["plan_path"] == path) & z["differs_from_expert"])[0].tolist()
            if len(idx) >= 50:                   # 太少的组合不挑
                for n, i in enumerate(rng.sample(idx, k)):
                    out.append((f"{cls.lower()}_{path.lower()}_{n}", i))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pick", type=int, default=0)
    ap.add_argument("items", nargs="*", help="name:行号")
    args = ap.parse_args()
    z = np.load(INDEX.format(args.split))
    synth = SpeedSynth()
    os.makedirs(args.out, exist_ok=True)
    items = pick(z, args.pick) if args.pick else [(a.split(":")[0], int(a.split(":")[1])) for a in args.items]
    for name, i in items:
        render(z, i, synth, f"{args.out}/{name}_{z['token'][i]}.png")
        print(f"{name:28s} {z['token'][i]}  v0 {z['v0'][i]:5.1f}  sug '{z['suggestion'][i]}'  ->  {z['decision'][i]}"
              f"   (expert {z['expert_speed'][i]})")


if __name__ == "__main__":
    main()
