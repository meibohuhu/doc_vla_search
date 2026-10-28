"""
D2 速度反事实（docs/1026/cf_sft_autovla_plan.md §3 D2）。

画面不动，路径用专家的（不够长时沿 PDM centerline 续），只把快慢换掉：

    STOP / DECELERATE / KEEP / ACCELERATE  各一条，恒定加速度，ACCELERATE 封顶到限速

标签用 plan_from_track 在合成轨迹上重新算。同一类准备几档幅度，取第一档标签对得上的 ——
例如 −1.0 m/s² 在 v0 < 5 m/s 时 3 s 后已低于 2 m/s，会被标成 STOP，那就换 −0.5 再试。
都对不上就丢掉，让每个词对应的轨迹干净。

然后逐关检查，任何一关不过就记下原因、丢掉:
    弯道     横向加速度 ≤ A_LAT_MAX（快的档会开过专家路径终点、沿路线高速拐弯）
    PDM      NC / DAC / DDC 全为 1（PDM 只查前 4 s）
    红灯     metric cache 里没有红绿灯，PDM 从来不管红灯。这里从 navsim_logs 取红绿灯自己查：
             往路线上一段红灯 lane connector 里开得比 t = 0 时更深就算闯。t = 0 已经进去了的不算
    限速     ACCELERATE 加到地图限速就不再加；v0 已经到限速、加不上去的，标签对不上，自然丢掉
    tokenizer 编码成 10 个 action token 再解码，解码后的标签也要对得上

🔴 其他车按日志回放，不会对自车做出反应。STOP / DECELERATE 时后车照样按日志开过来，
   PDM 不算自车的责任（后车追尾不归责），这里也不查。

用法:
    python tools/cf_data/speed.py --n_tokens 300 --out /scratch/mh2803/vla/nuplan/cf_navtrain_v1/proto_speed --viz 12
    python tools/cf_data/speed.py --workers 48 --out /scratch/mh2803/vla/nuplan/cf_navtrain_v1/speed/navtrain
"""

import argparse
import json
import math
import os
import random
from typing import Dict, List, Optional

import numpy as np
import torch
from shapely.geometry import Point

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataclasses import Trajectory
from models.utils.feasible import _headings, _interp_xy
from models.utils.rl_rewards import plan_from_track
from lane_change import (A_LAT_MAX, DT, HEAD_MISMATCH, JSON_DIR, N, SUB, LaneChangeSynth, max_lat_acc,
                         to_global, to_local, upsample)

# 每类按顺序试的加速度（m/s²），取第一档标签对得上的
PROFILES: Dict[str, List[float]] = {
    "STOP": [-2.5],                       # 减速侧 p99.9（feasible.py 的重标）。v0 > 9.5 时 3 s 停不下来 → 丢
    "DECELERATE": [-1.0, -0.5],           # v0 < ~5 m/s 时 −1.0 会被标成 STOP，退到 −0.5
    "KEEP": [0.0],
    "ACCELERATE": [1.0],                  # navtest |a| p90 1.29；+1.5 持续 5 s 太猛（logs/0902_nvidia/step0_gonogo.md §3）
}
V_MAX = 16.0
# 地图没有限速时的默认值：波士顿城区 25 mph，新加坡 50 km/h。拉斯维加斯 / 匹兹堡地图里全有
DEFAULT_LIMIT = {"us-ma-boston": 11.2, "sg-one-north": 13.9}


def schedule(v0: float, a: float, v_cap: float) -> np.ndarray:
    """恒定加速度，0.1 s 一步积分 → 0.1 s .. 5.0 s 的累积弧长（50 个）。速度夹在 [0, max(v_cap, v0)]。"""
    h, v, s, out = DT / SUB, float(v0), 0.0, []
    cap = max(v_cap, v)                   # v0 已经超过封顶时保持 v0，不强行减速
    for _ in range(N * SUB):
        vn = min(max(v + a * h, 0.0), cap)
        s += 0.5 * (v + vn) * h
        v = vn
        out.append(s)
    return np.asarray(out)


def forward_only(gt: np.ndarray) -> np.ndarray:
    """专家停着时 GT 会前后抖几厘米。只留沿专家朝向往前走了 > 5 cm 的点，
    否则合成的停车轨迹会沿倒退的那一段取点、朝向翻 180°（全量里 2 条，解码后侧移 1.5 m）。"""
    keep, last = [], np.zeros(2)
    for p in gt:
        if (p[:2] - last) @ (math.cos(p[2]), math.sin(p[2])) > 0.05:
            keep.append(p)
            last = p[:2]
    return np.asarray(keep, dtype=np.float64).reshape(-1, gt.shape[1])


class SpeedSynth(LaneChangeSynth):
    """复用 LaneChangeSynth 的 metric cache / PDM / 地图 / tokenizer。"""

    def city(self, log: str) -> str:
        self.map_api(log)
        return self._log_map[log]

    def speed_limit(self, m, city: str, pts_g: np.ndarray, heads_g: np.ndarray) -> float:
        """路径上各点所在 lane 的限速取最小（一个点上有多条重叠的，取朝向对得上的里面最大的）。

        只看 lane，不看路口里的 lane connector：拉斯维加斯 connector 上多是 4.5 / 2.2 m/s 的转弯限速，
        混进来的话 300 帧里有 106 帧被压到 ≤ 4.5，56 帧专家自己都比"限速"快 1 m/s 以上。
        转弯该多快由横向加速度管。整条路径都在路口里时才退回用 connector；都没有 → 城市默认值。"""
        for layer in (SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR):
            lims = []
            for (x, y), h in zip(pts_g, heads_g):
                here = []
                for ln in m.get_all_map_objects(Point2D(x, y), layer):
                    ls = ln.baseline_path.linestring
                    s = ls.project(Point(x, y))
                    p0, p1 = ls.interpolate(max(0.0, s - 0.5)), ls.interpolate(min(ls.length, s + 0.5))
                    dh = abs((math.atan2(p1.y - p0.y, p1.x - p0.x) - h + math.pi) % (2 * math.pi) - math.pi)
                    lim = ln.speed_limit_mps
                    if dh < HEAD_MISMATCH and lim is not None and lim > 0:
                        here.append(float(lim))
                if here:
                    lims.append(max(here))
            if lims:
                return min(lims)
        return DEFAULT_LIMIT.get(city, 13.4)

    def run(self, token: str, rng: random.Random = None) -> List[dict]:
        """一帧 → 每类一条候选（不论成败，都带 status / reason）。"""
        d = json.load(open(f"{self.json_dir}/{token}.json"))
        log = d["front_camera_paths"][-1].split("/trainval/")[1].split("/")[0]
        mc = self.fs.metric_cache(token)
        e = mc.ego_state.rear_axle
        ex, ey, eh = float(e.x), float(e.y), float(e.heading)
        gt = np.asarray(d["gt_trajectory"], dtype=np.float64)
        v0 = float(np.hypot(*d["velocity"]))
        exp = plan_from_track([(0.0, 0.0)] + [tuple(p) for p in gt[:, :2]])
        m, city = self.map_api(log), self.city(log)

        # 专家自己闯不闯红灯 —— 检查器的误报率
        base = {"token": token, "cmd": d["instruction"], "v0": v0, "log": log, "city": city,
                "expert": list(exp[1:]) if exp else None,
                "expert_red": self.red_light(mc, m, log, token, self.boxes(upsample(gt), ex, ey, eh)),
                "red_lights": sum(r for _, r in self._log_tl[log][1][self._log_tl[log][0][token]])}

        xy, s = self.fs._path(token, forward_only(gt))
        out = []
        for cls, accs in PROFILES.items():
            rec = dict(base, cls=cls, status="fail", reason="")
            v_cap = V_MAX
            if cls == "ACCELERATE":
                q = schedule(v0, accs[0], V_MAX)[SUB - 1::SUB]
                p = _interp_xy(xy, s, np.minimum(q, s[-1]))
                g = to_global(p, ex, ey, eh)
                hs = _headings(p, (0.0, 0.0)) + eh
                v_cap = self.speed_limit(m, city, g, hs)
                rec["v_limit"] = v_cap
            got, traj, fine = [], None, None
            for a in accs:
                q = schedule(v0, a, v_cap)
                pts = _interp_xy(xy, s, q)
                if pts is None:
                    got.append("path_short"); continue
                tr = np.concatenate([pts[SUB - 1::SUB], _headings(pts[SUB - 1::SUB], (0.0, 0.0))[:, None]], axis=1)
                lab = plan_from_track([(0.0, 0.0)] + [tuple(p) for p in tr[:, :2]])
                got.append(lab[1] if lab else None)
                if lab and lab[1] == cls:
                    traj, fine, rec["a"], rec["plan"] = tr, np.vstack([[0.0, 0.0], pts]), a, list(lab[1:])
                    break
            if traj is None:
                reason = "path_short" if all(g == "path_short" for g in got) else \
                    ("at_speed_limit" if cls == "ACCELERATE" and v0 >= v_cap - 1.2 else "label_mismatch")
                out.append(dict(rec, reason=reason, got=got)); continue
            rec["traj"] = traj.round(4).tolist()
            rec["v_end"] = float(np.linalg.norm(fine[-1] - fine[-2]) / (DT / SUB))
            # 专家减速转弯时，KEEP / ACCELERATE 会开过专家路径的终点、沿路线中心线高速拐进路口
            # （554714b4c7f15a4d：11.2 m/s 拐 90°）。PDM 不拦（路径在车道里），tokenizer 也表示不了
            rec["a_lat"] = max_lat_acc(traj)
            if rec["a_lat"] > A_LAT_MAX:
                out.append(dict(rec, reason="too_fast_for_curve")); continue

            tj = Trajectory(traj.astype(np.float32), TrajectorySampling(num_poses=N, interval_length=DT))
            met = self.fs.score(token, tj)
            if met is None:
                out.append(dict(rec, reason="pdm_error")); continue
            rec["pdm"] = met

            idx, dec = self.tokenize(traj)
            rec["token_ids"] = idx.tolist()
            rec["decoded"] = dec.round(4).tolist()
            rec["token_err"] = float(np.linalg.norm(dec[:, :2] - traj[:, :2], axis=1).max())
            dlab = plan_from_track([(0.0, 0.0)] + [tuple(p) for p in dec[:, :2]])
            rec["plan_decoded"] = list(dlab[1:]) if dlab else None

            if met["nc"] < 1 or met["dac"] < 1 or met["ddc"] < 1:
                out.append(dict(rec, reason="pdm_" + "_".join(
                    k for k in ("nc", "dac", "ddc") if met[k] < 1))); continue
            red = self.red_light(mc, m, log, token, self.boxes(fine, ex, ey, eh))
            rec["red"] = red
            if red is not None:
                out.append(dict(rec, reason="red_light")); continue
            if rec["plan_decoded"] != rec["plan"]:
                out.append(dict(rec, reason="token_label")); continue
            rec["start_from_stop"] = cls == "ACCELERATE" and v0 < 0.5
            out.append(dict(rec, status="ok"))
        return out


# ── 画图 ────────────────────────────────────────────────────────────────

COLORS = {"STOP": "tab:red", "DECELERATE": "tab:orange", "KEEP": "tab:blue", "ACCELERATE": "tab:green"}


def render(recs: List[dict], synth: SpeedSynth, out_png: str):
    """一帧四类画在一张图上：前视图 | BEV | 速度曲线。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.image import imread

    r0 = recs[0]
    d = json.load(open(f"{synth.json_dir}/{r0['token']}.json"))
    mc = synth.fs.metric_cache(r0["token"])
    e = mc.ego_state.rear_axle
    ex, ey, eh = float(e.x), float(e.y), float(e.heading)
    gt = np.asarray(d["gt_trajectory"])

    fig = plt.figure(figsize=(17, 6.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 0.8, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(imread(d["front_camera_paths"][-1]))
    ax.set_title("front camera, t = 0", fontsize=10)
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    dm = mc.drivable_area_map
    for geom, typ in zip(dm._geometries, dm._map_types):
        if typ in (SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR):
            loc = to_local(np.asarray(geom.exterior.coords), ex, ey, eh)
            ax.plot(-loc[:, 1], loc[:, 0], color="0.6" if typ == SemanticMapLayer.LANE else "0.85", lw=0.7)
    for i, alpha in ((0, 0.9), (25, 0.35)):
        for tok, geom in zip(mc.observation[i].tokens, mc.observation[i]._geometries):
            loc = to_local(np.asarray(geom.exterior.coords), ex, ey, eh)
            ax.fill(-loc[:, 1], loc[:, 0], color="tab:purple", alpha=alpha * 0.5, lw=0)
    # 路线上 5 s 内亮过红灯的 connector（日志里的红绿灯）
    m = synth.map_api(r0["log"])
    idx, tls = synth._log_tl[r0["log"]]
    route = {str(x) for x in mc.route_lane_ids}
    i0 = idx[r0["token"]]
    reds = {str(lc) for k in range(i0, min(i0 + N + 1, len(tls))) for lc, is_red in tls[k] if is_red and str(lc) in route}
    for lc in reds:
        obj = m.get_map_object(lc, SemanticMapLayer.LANE_CONNECTOR)
        if obj is not None:
            loc = to_local(np.asarray(obj.polygon.exterior.coords), ex, ey, eh)
            ax.fill(-loc[:, 1], loc[:, 0], color="red", alpha=0.25, lw=0)
    ax.plot(-gt[:, 1], gt[:, 0], "o-", color="0.45", ms=3, lw=1, label="expert")
    for r in recs:
        if "traj" in r:
            tr = np.asarray(r["traj"])
            ax.plot(-tr[:, 1], tr[:, 0], "o", color=COLORS[r["cls"]], ms=4 if r["status"] == "ok" else 2,
                    mfc="none" if r["status"] != "ok" else COLORS[r["cls"]], label=r["cls"])
    ax.plot([0], [0], "s", color="k", ms=6)
    far = max([gt[-1, 0]] + [np.asarray(r["traj"])[-1, 0] for r in recs if "traj" in r])
    ax.set_xlim(-15, 15)
    ax.set_ylim(-12, max(40, far + 8))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.legend(loc="lower right", fontsize=6)
    ax.set_title("red fill = on-route connector that is red within 5 s", fontsize=8)

    ax = fig.add_subplot(gs[0, 2])
    t = np.arange(N + 1) * DT
    def spd(tr):
        p = np.vstack([[0.0, 0.0], np.asarray(tr)[:, :2]])
        return np.linalg.norm(np.diff(p, axis=0), axis=1) / DT
    ax.step(t[1:], spd(gt), where="pre", color="0.45", lw=1.5, label=f"expert {r0['expert']}")
    lines = []
    for r in recs:
        if "traj" in r:
            ax.step(t[1:], spd(r["traj"]), where="pre", color=COLORS[r["cls"]],
                    lw=2 if r["status"] == "ok" else 1, ls="-" if r["status"] == "ok" else ":",
                    label=f"{r['cls']} a={r['a']:+.1f}")
        tag = "ok" if r["status"] == "ok" else r["reason"]
        extra = f" got={r['got']}" if "got" in r else ""
        extra += f" red@{r['red']['t']}s" if r.get("red") else ""
        lines.append(f"{r['cls']}: {tag}{extra}")
    if any("v_limit" in r for r in recs):
        vl = [r["v_limit"] for r in recs if "v_limit" in r][0]
        ax.axhline(vl, color="k", ls="--", lw=0.8, label=f"speed limit {vl:.1f}")
    ax.axhline(2.0, color="0.7", ls=":", lw=0.8)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("speed (m/s, 0.5 s average)")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)

    fig.suptitle(f"token {r0['token']}   v0 {r0['v0']:.1f} m/s   cmd '{r0['cmd']}'   {r0['city']}"
                 f"   expert red-light: {r0['expert_red']}\n" + "   |   ".join(lines), fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=85)
    plt.close(fig)


# ── 批量 ────────────────────────────────────────────────────────────────

_SYNTH: Optional[SpeedSynth] = None
_HEAVY = ("traj", "decoded", "token_ids")


def _init_worker(json_dir: str):
    global _SYNTH
    torch.set_num_threads(1)
    _SYNTH = SpeedSynth(json_dir=json_dir)


def _run_one(tok: str) -> List[dict]:
    try:
        recs = _SYNTH.run(tok)
    except Exception as ex:  # noqa: BLE001 —— 记下来，不中断
        recs = [{"token": tok, "status": "error", "reason": repr(ex)[:200]}]
    _SYNTH.fs._cache.clear()
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", default=JSON_DIR, help="navtrain_nocot 或 navtrain_nocot_val")
    ap.add_argument("--n_tokens", type=int, default=0, help="随机抽多少帧；0 = 全部")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--viz", type=int, default=0, help="画几帧（每帧四类一张图），按有无失败各取一半")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tokens = sorted(f[:-5] for f in os.listdir(args.json_dir))
    if args.n_tokens:
        tokens = random.Random(args.seed).sample(tokens, args.n_tokens)

    recs = []
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers, initializer=_init_worker, initargs=(args.json_dir,)) as pool:
            for i, rs in enumerate(pool.imap(_run_one, tokens, chunksize=16), 1):
                recs += rs
                if i % 5000 == 0:
                    print(f"  {i}/{len(tokens)}", flush=True)
    else:
        _init_worker(args.json_dir)
        for tok in tokens:
            recs += _run_one(tok)

    ok = [r for r in recs if r["status"] == "ok"]
    with open(f"{args.out}/candidates.jsonl", "w") as f:
        for r in recs:
            if r["status"] != "ok":
                r = {k: v for k, v in r.items() if k not in _HEAVY}
            f.write(json.dumps(r) + "\n")
    with open(f"{args.out}/ok.jsonl", "w") as f:
        for r in ok:
            f.write(json.dumps(r) + "\n")

    from collections import Counter
    print(f"{len(tokens)} tokens -> {len(recs)} candidates, ok {len(ok)}")
    for cls in PROFILES:
        rs = [r for r in recs if r.get("cls") == cls]
        print(f"  {cls:11s}", Counter(r["status"] if r["status"] == "ok" else r["reason"] for r in rs).most_common())
    errs = [r for r in recs if r["status"] == "error"]
    if errs:
        print("errors:", len(errs), errs[0]["reason"])
    first = {}
    for r in recs:
        if "expert_red" in r:
            first.setdefault(r["token"], r)
    print("expert red-light flagged:", sum(r["expert_red"] is not None for r in first.values()), "/", len(first))
    if ok:
        print("ok token_err p50/p90/max:", np.percentile([r["token_err"] for r in ok], [50, 90, 100]).round(2))
        print("ok same as expert speed word:", Counter((r["cls"], r["expert"][0] == r["cls"]) for r in ok if r["expert"]))
        print("ok start_from_stop:", sum(r.get("start_from_stop", False) for r in ok))
        print("ok a used:", Counter((r["cls"], r["a"]) for r in ok))

    if args.viz:
        synth = _SYNTH or SpeedSynth(json_dir=args.json_dir)
        by_tok: Dict[str, List[dict]] = {}
        for r in recs:
            if "cls" in r:
                by_tok.setdefault(r["token"], []).append(r)
        mixed = [t for t, rs in by_tok.items() if any(r["status"] != "ok" and r["reason"] not in ("label_mismatch", "path_short") for r in rs)]
        clean = [t for t, rs in by_tok.items() if t not in mixed]
        pick = mixed[:args.viz // 2] + clean[:args.viz - min(len(mixed), args.viz // 2)]
        for j, t in enumerate(pick):
            render(by_tok[t], synth, f"{args.out}/{'mixed' if t in mixed else 'clean'}_{j:02d}_{t}.png")
            synth.fs._cache.clear()


if __name__ == "__main__":
    main()
