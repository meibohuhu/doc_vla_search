"""For lane-change-like navtrain frames: centerline offset stats + example figures.

usage: lc_render.py offsets <scan.json> <out.json>
       lc_render.py render  <offsets.json> <out_dir> <n_left> <n_right> <seed>
"""
import json
import lzma
import math
import os
import pickle
import random
import sys
from multiprocessing import Pool

import numpy as np

MC_META = "/scratch/mh2803/vla/nuplan/navtrain_metric_cache/metadata/navtrain_metric_cache_metadata_node_0.csv"
JSON_DIR = "/scratch/mh2803/vla/nuplan/navtrain_nocot"
LOG_DIR = "/scratch/mh2803/vla/nuplan/navsim_logs/trainval"
SENSOR = "/scratch/mh2803/vla/nuplan/sensor_blobs/trainval"


def mc_paths():
    with open(MC_META) as f:
        rows = f.read().splitlines()[1:]
    return {p.split("/")[-2]: p for p in rows}


def load_mc(path):
    with lzma.open(path, "rb") as f:
        return pickle.load(f)


def to_global(mc, xy):
    e = mc.ego_state.rear_axle
    c, s = math.cos(e.heading), math.sin(e.heading)
    xy = np.asarray(xy, dtype=np.float64)
    return np.stack([e.x + xy[:, 0] * c - xy[:, 1] * s, e.y + xy[:, 0] * s + xy[:, 1] * c], axis=1)


def to_local(mc, xy):
    e = mc.ego_state.rear_axle
    c, s = math.cos(e.heading), math.sin(e.heading)
    xy = np.asarray(xy, dtype=np.float64)
    dx, dy = xy[:, 0] - e.x, xy[:, 1] - e.y
    return np.stack([dx * c + dy * s, -dx * s + dy * c], axis=1)


def signed_offset(cl, pt):
    """Signed lateral distance of a global point from the centerline (left positive)."""
    from shapely.geometry import Point
    s = float(cl.project(Point(*pt)))
    L = float(cl.length)
    s_a = max(0.0, min(s, L - 0.5))           # keep a non-zero tangent at the ends of the path
    a = np.asarray(cl.interpolate(np.array([s_a, s_a + 0.5, s]), as_array=True))[:, :2]
    t = a[1] - a[0]
    d = np.asarray(pt) - a[2]
    sign = 1.0 if (t[0] * d[1] - t[1] * d[0]) >= 0 else -1.0
    return sign * float(np.hypot(*d)), s, float(cl.length)


def offsets_one(args):
    r, path = args
    try:
        mc = load_mc(path)
        g = np.asarray(json.load(open(f"{JSON_DIR}/{r['token']}.json"))["gt_trajectory"])
        pts = to_global(mc, np.vstack([[0, 0], g[:, :2]]))
        o0, _, _ = signed_offset(mc.centerline, pts[0])
        oe, se, L = signed_offset(mc.centerline, pts[-1])
        r = dict(r, off0=o0, off_end=oe, cl_beyond=se >= L - 0.5)
    except Exception as ex:  # noqa: BLE001
        r = dict(r, err=repr(ex))
    return r


def cmd_offsets(scan, out):
    R = json.load(open(scan))
    lc = [r for r in R if r["cmd"] == "keep forward" and abs(r["lat_end"]) >= 2.5 and abs(r["head_end"]) < 10]
    P = mc_paths()
    jobs = [(r, P[r["token"]]) for r in lc if r["token"] in P]
    with Pool(32) as pool:
        rows = pool.map(offsets_one, jobs, chunksize=16)
    json.dump(rows, open(out, "w"))
    print(len(lc), "candidates,", len(jobs), "with metric cache ->", out)


def future_front(token, img_now, k):
    """Front camera k frames (2 Hz) after `token`, via navsim_logs."""
    log = img_now.split("/trainval/")[1].split("/")[0]
    frames = pickle.load(open(f"{LOG_DIR}/{log}.pkl", "rb"))
    idx = next(i for i, fr in enumerate(frames) if fr["token"] == token)
    j = min(idx + k, len(frames) - 1)
    return f"{SENSOR}/{frames[j]['cams']['CAM_F0']['data_path']}", (j - idx) * 0.5


def render_one(r, mc, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.image import imread
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    d = json.load(open(f"{JSON_DIR}/{r['token']}.json"))
    g = np.asarray(d["gt_trajectory"])
    h = np.asarray(d["his_trajectory"])
    img_now = d["front_camera_paths"][-1]
    img_fut, dt = future_front(r["token"], img_now, 5)

    fig = plt.figure(figsize=(18, 6.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1.6, 1.0])
    for k, (p, t) in enumerate([(img_now, "front camera, t = 0"), (img_fut, f"front camera, t = +{dt:.1f} s")]):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(imread(p))
        ax.set_title(t, fontsize=11)
        ax.axis("off")

    ax = fig.add_subplot(gs[0, 2])
    dm = mc.drivable_area_map
    for geom, typ in zip(dm._geometries, dm._map_types):
        if typ not in (SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR):
            continue
        xs, ys = geom.exterior.xy
        loc = to_local(mc, np.stack([xs, ys], axis=1))
        lane = typ == SemanticMapLayer.LANE
        # plot with x forward pointing up: horizontal = -y (left on the left), vertical = x
        ax.plot(-loc[:, 1], loc[:, 0], color="0.55" if lane else "0.8",
                lw=0.8 if lane else 0.6, ls="-" if lane else ":")
    cl = np.asarray(mc.centerline.interpolate(
        np.arange(0, float(mc.centerline.length), 1.0), as_array=True))[:, :2]
    cl = to_local(mc, cl)
    ax.plot(-cl[:, 1], cl[:, 0], color="tab:blue", lw=1.2, ls="--", label="PDM route centerline")
    ax.plot(-h[:, 1], h[:, 0], "o", color="0.4", ms=3, label="history")
    ax.plot(-g[:, 1], g[:, 0], "o-", color="tab:red", ms=3.5, lw=1.4, label="expert GT (0.5 s)")
    ax.plot([0], [0], "s", color="k", ms=6)
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, max(55, g[-1, 0] + 5))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.legend(loc="lower right", fontsize=7)
    side = "LEFT" if r["lat_end"] > 0 else "RIGHT"
    ax.set_title(f"BEV (ego frame, up = forward)\noff route centerline: t0 {r['off0']:+.1f} m -> end {r['off_end']:+.1f} m",
                 fontsize=9)
    fig.suptitle(
        f"token {r['token']} | cmd '{r['cmd']}' | v0 {r['v0']:.1f} m/s | "
        f"end lateral {r['lat_end']:+.1f} m, heading {r['head_end']:+.1f} deg (looks like {side})  ||  "
        f"current label <PLAN>{r['speed']},{r['path']}</PLAN>",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=90)
    plt.close(fig)


def cmd_render(offsets, out_dir, n_left, n_right, seed, lo, hi, prefix):
    rows = [r for r in json.load(open(offsets)) if "err" not in r and r["path"] == "STRAIGHT"
            and lo <= abs(r["off_end"] - r["off0"]) < hi]
    rnd = random.Random(seed)
    L = [r for r in rows if r["lat_end"] > 0]
    R = [r for r in rows if r["lat_end"] < 0]
    pick = [("left", r) for r in rnd.sample(L, n_left)] + [("right", r) for r in rnd.sample(R, n_right)]
    P = mc_paths()
    os.makedirs(out_dir, exist_ok=True)
    for i, (side, r) in enumerate(pick):
        out = f"{out_dir}/{prefix}_{side}_{i:02d}_{r['token']}.png"
        render_one(r, load_mc(P[r["token"]]), out)
        print(out)


if __name__ == "__main__":
    if sys.argv[1] == "offsets":
        cmd_offsets(sys.argv[2], sys.argv[3])
    else:
        cmd_render(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]),
                   float(sys.argv[7]), float(sys.argv[8]), sys.argv[9])
