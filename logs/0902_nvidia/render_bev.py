"""把 rollout 画成 BEV —— 用来肉眼确认"不可行"到底是速度问题还是转向问题。

    python logs/0902/render_bev.py --jsonl logs/0902/navtest600_greedy_G1_ep1.jsonl \
        --out logs/0902/bev_greedy --n 8 --pick infeasible_dac

图里画什么（全部在 ego 局部系，x 向前、y 向左）:
    浅灰面     可行驶区（metric_cache.drivable_area_map，就是判 DAC 用的那份）
    灰框       其他 agent 在 t=0 的框；淡框 = t=2s 的位置
    蓝虚线     centerline（route）
    绿线       GT 轨迹
    红/绿粗线  模型预测轨迹（红 = 判不可行）
    细虚线     4 条反事实速度剖面，绿=在 D 里、红=不在

⚠️ 数据源是 metric_cache 本身 —— 也就是 PDMScorer 判分时**实际看到的**那份，
   不是从 log 里另取的。所以图上看到的就是判定依据，不会出现"图和分数对不上"。
"""
import argparse, json, lzma, math, os, pickle, sys, textwrap
sys.path.insert(0, '.'); sys.path.insert(0, './navsim')
os.environ.setdefault("NUPLAN_MAPS_ROOT", "/data/autovla_data/nuplan/maps")
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
os.environ.setdefault("OPENSCENE_DATA_ROOT", "/data/autovla_data/nuplan")

import numpy as np
import matplotlib
matplotlib.use("Agg")
# ⚠️ 机器上没有 CJK 字体，图里一律用英文，否则中文会渲染成方框
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly
from pathlib import Path
from navsim.common.dataloader import MetricCacheLoader
from models.utils.feasible import FeasibleSet, PROFILES

COL = {"HARD_BRAKE": "#8e44ad", "BRAKE": "#2980b9", "KEEP": "#16a085", "ACCEL": "#d35400"}

# Pacifica。⚠️ DAC 判的是【车身四角】在不在可行驶区里，不是轨迹中心线 ——
#    只画中心线会看到"线明明在路上却 DAC=0"。所以必须把车框画出来。
EGO_L, EGO_W, RAC = 5.176, 2.297, 1.461


def ego_box(x, y, h):
    """rear-axle 位姿 → 车身四角（ego 局部系）。"""
    cx, cy = x + RAC * math.cos(h), y + RAC * math.sin(h)
    hl, hw = EGO_L / 2, EGO_W / 2
    c, s = math.cos(h), math.sin(h)
    return np.array([[cx + c * dx - s * dy, cy + s * dx + c * dy]
                     for dx, dy in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))])


def to_local(pts, ex, ey, eh):
    c, s = math.cos(eh), math.sin(eh)
    dx, dy = np.asarray(pts)[:, 0] - ex, np.asarray(pts)[:, 1] - ey
    return np.stack([dx * c + dy * s, -dx * s + dy * c], axis=1)


def draw(ax, mc, rec, roll, fs, tok):
    ego = mc.ego_state.rear_axle
    ex, ey, eh = float(ego.x), float(ego.y), float(ego.heading)

    # 可行驶区
    dm = mc.drivable_area_map
    for g in dm._geometries:
        try:
            xy = to_local(np.asarray(g.exterior.coords), ex, ey, eh)
        except Exception:
            continue
        ax.add_patch(MplPoly(xy, closed=True, fc="#e8e8e8", ec="#cccccc", lw=0.4, zorder=0))

    # ⚠️ agent 是【随时间移动】的，而 ego 轨迹跨 5s ——
    #    只画 t=0 会让"轨迹穿过某个框"看起来像碰撞，其实那辆车早开走了。
    #    所以画三帧 + 在轨迹上标时间刻度，让读者能对上时间。
    for t_idx, alpha, lw, ls in ((0, 0.95, 1.3, "-"), (20, 0.5, 1.0, "-"), (40, 0.25, 1.0, "--")):
        try:
            om = mc.observation[t_idx]
        except Exception:
            continue
        for k, g in enumerate(om._geometries):
            if om._tokens[k] == "red_light":
                continue
            try:
                xy = to_local(np.asarray(g.exterior.coords), ex, ey, eh)
            except Exception:
                continue
            ax.add_patch(MplPoly(xy, closed=True, fc="none", ec="#555555", ls=ls,
                                 lw=lw, alpha=alpha, zorder=3))
    for t_idx, alpha, lw, ls, lb in ((0, 0.95, 1.3, "-", "agents t=0s"),
                                     (20, 0.5, 1.0, "-", "agents t=2s"),
                                     (40, 0.25, 1.0, "--", "agents t=4s")):
        ax.plot([], [], linestyle=ls, color="#555555", lw=lw, alpha=alpha, label=lb)

    # centerline
    try:
        cl = mc.centerline
        q = np.arange(0, min(float(cl.length), 120.0), 2.0)
        arr = np.asarray(cl.interpolate(q, as_array=True))[:, :2]
        p = to_local(arr, ex, ey, eh)
        ax.plot(p[:, 0], p[:, 1], "--", color="#3498db", lw=1.0, alpha=0.7, zorder=2, label="centerline")
    except Exception:
        pass

    # 反事实剖面
    v0 = roll.get("v0", None)
    if v0 is not None:
        for mode in PROFILES:
            tr = fs.synth(tok, np.asarray(roll["poses"]), v0, mode)
            if tr is None:
                continue
            p = np.asarray(tr.poses)[:, :2]
            ok = mode in roll["D"]
            # ⚠️ 可行的反事实必须【压在 pred 之上】并标出终点 ——
            #    刹车剖面很短，不然会被 pred 的粗线整条盖住，图就说明不了"存在可行解"。
            ax.plot(p[:, 0], p[:, 1], linestyle=":" if ok else (0, (1, 3)),
                    color=COL[mode] if ok else "#999999", lw=2.6 if ok else 1.1,
                    alpha=1.0 if ok else 0.45, zorder=9 if ok else 4,
                    label=f"{mode}{' (feasible)' if ok else ' ✗'}")
            if ok:
                ax.plot(p[-1, 0], p[-1, 1], "s", ms=5, mfc="white", mew=1.6,
                        color=COL[mode], zorder=10)
                ax.add_patch(MplPoly(ego_box(p[-1, 0], p[-1, 1], float(tr.poses[-1][2])),
                                     closed=True, fc="none", ec=COL[mode],
                                     lw=1.0, ls=":", alpha=0.9, zorder=9))

    # GT + 预测
    gt = np.asarray(roll["gt"])[:, :2]
    ax.plot(np.r_[0, gt[:, 0]], np.r_[0, gt[:, 1]], "-", color="#27ae60", lw=2.2, zorder=5, label="GT")
    P = np.asarray(roll["poses"])
    pr = P[:, :2]
    c = "#27ae60" if roll["feasible"] else "#e74c3c"
    ax.plot(np.r_[0, pr[:, 0]], np.r_[0, pr[:, 1]], "-", color=c, lw=2.6, zorder=6, label="pred")
    # ★ 车身框:DAC 判的是四角，不是中心线
    for k in range(0, len(P), 2):
        ax.add_patch(MplPoly(ego_box(P[k, 0], P[k, 1], P[k, 2]), closed=True,
                             fc="none", ec=c, lw=0.8, alpha=0.55, zorder=5))
    ax.add_patch(MplPoly(ego_box(0, 0, 0), closed=True, fc="none", ec="k", lw=1.4, zorder=7))
    ax.plot(0, 0, "k^", ms=7, zorder=8)

    # 时间刻度:让 agent 快照和 ego 位置能对上时间
    for k in (1, 3, 5, 7, 9):                       # t = 1,2,3,4,5 s
        ax.plot(pr[k, 0], pr[k, 1], "o", ms=3.5, color=c, zorder=8)
        ax.annotate(f"{0.5*(k+1):.0f}s", (pr[k, 0], pr[k, 1]), fontsize=6,
                    color=c, xytext=(2, 4), textcoords="offset points", zorder=8)

    ax.set_xlim(-12, 55); ax.set_ylim(-22, 22); ax.set_aspect("equal")
    ax.grid(alpha=0.15); ax.set_xlabel("x forward [m]"); ax.set_ylabel("y left [m]")
    plan = "/".join(roll["plan"]) if roll["plan"] else "—"
    ax.set_title(f"{tok}   PDMS={roll['pdms']:.3f}   NC={roll['nc']:.0f}  DAC={roll['dac']:.0f}"
                 f"   v0={v0:.1f} m/s\n"
                 f"said <PLAN>{plan}   |   D = {roll['D'] or 'EMPTY'}   |   "
                 f"said-in-D = {roll['say_in_D']}", fontsize=9)
    ax.legend(fontsize=6, loc="upper left", ncol=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--pick", default="infeasible_dac",
                    choices=["infeasible_dac", "infeasible_nc", "feasible", "any"])
    ap.add_argument("--tokens", default=None,
                    help="只画这些 token（每行一个）。给跨 ckpt 对照用 —— 同一批帧才好比。")
    ap.add_argument("--samples", default="/data/autovla_data/nuplan/navtest_cot")
    ap.add_argument("--cache", default="/data/autovla_data/nuplan/navtest_metric_cache")
    args = ap.parse_args()

    fs = FeasibleSet(Path(args.cache))
    rows = [json.loads(l) for l in open(args.jsonl)]
    if args.tokens:
        want = [l.strip() for l in open(args.tokens) if l.strip()]
        by = {r["token"]: r for r in rows}
        rows = [by[t] for t in want if t in by]
    cand = []
    for r in rows:
        for g, x in enumerate(r["rollouts"]):
            ok = {"infeasible_dac": (not x["feasible"]) and x["dac"] < 1 and x["nc"] >= 1,
                  "infeasible_nc": (not x["feasible"]) and x["nc"] < 1,
                  "feasible": x["feasible"], "any": True}[args.pick]
            if ok:
                cand.append((r["token"], g, x))
    print(f"候选 {len(cand)} 条 ({args.pick})，画前 {args.n} 条")
    os.makedirs(args.out, exist_ok=True)

    for i, (tok, g, x) in enumerate(cand[: args.n]):
        s = json.load(open(os.path.join(args.samples, tok + ".json")))
        mc = fs.metric_cache(tok)
        # rollout 的 jsonl 里没存 poses，这里用 D/metrics 重算不了 —— 直接从 text 拿不到，
        # 所以退而用 GT + 剖面；若 jsonl 里带 poses 就用真的
        roll = dict(x)
        roll["gt"] = s["gt_trajectory"]
        roll["v0"] = float(np.hypot(*s["velocity"][:2]))
        if "poses" not in roll:
            print(f"  ⚠️ {tok} 的 jsonl 里没有 poses，跳过（需要重跑 rollout 时加 --dump-poses）")
            continue
        fig, ax = plt.subplots(figsize=(9, 7))
        draw(ax, mc, None, roll, fs, tok)
        txt = x["text"].split("</think>")[0].replace("<think>", "").strip()
        fig.text(0.01, 0.005, textwrap.fill(txt, 110), fontsize=7, color="#333333")
        fig.tight_layout(rect=[0, 0.06, 1, 1])
        fp = os.path.join(args.out, f"{i:02d}_{tok}_g{g}.png")
        fig.savefig(fp, dpi=130); plt.close(fig)
        print("  →", fp)


if __name__ == "__main__":
    main()
