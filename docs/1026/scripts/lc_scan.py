"""Scan navtrain_nocot for lane-change-like GT trajectories and their plan_from_track labels."""
import glob
import importlib.util
import json
import math
import sys
from multiprocessing import Pool

import numpy as np

REPO = "/home/mh2803/vla/doc_vla_search"
spec = importlib.util.spec_from_file_location("rr", f"{REPO}/models/utils/rl_rewards.py")
rr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rr)


def one(f):
    d = json.load(open(f))
    g = np.asarray(d["gt_trajectory"])
    p = rr.trajectory_plan(g)
    return {
        "token": d["token"], "cmd": d["instruction"],
        "v0": float(np.hypot(*d["velocity"])),
        "lat_end": float(g[-1, 1]), "lon_end": float(g[-1, 0]),
        "head_end": float(math.degrees(g[-1, 2])),
        "speed": p[0] if p else None, "path": p[1] if p else None,
    }


if __name__ == "__main__":
    split, out = sys.argv[1], sys.argv[2]
    fs = sorted(glob.glob(f"/scratch/mh2803/vla/nuplan/{split}/*.json"))
    with Pool(32) as pool:
        rows = pool.map(one, fs, chunksize=256)
    json.dump(rows, open(out, "w"))
    print(len(rows), "rows ->", out)
