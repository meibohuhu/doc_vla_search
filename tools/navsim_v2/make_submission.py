"""infer_autovla.py 的输出 → 官方格式的 submission.pkl（和 v2 run_create_submission_pickle.py 写出来的一样）。

要在 v2 devkit 的 PYTHONPATH 下跑（Trajectory 要用 v2 的类 pickle，服务器才能读）：
    PYTHONPATH=$V2_DEVKIT python tools/navsim_v2/make_submission.py --inputs <dir>/inputs.jsonl --preds <dir>/preds \
        --out <dir>/submission.pkl [--team-name ... --authors ... --email ... --institution ... --country ...]

格式：{"team_name", "authors", "email", "institution", "country / region",
       "first_stage_predictions": [{token: Trajectory}], "second_stage_predictions": [{token: Trajectory}]}
Trajectory 取 AutoVLA 10 个点（0.5 s 间隔，5 s）的前 8 个 = 官方默认的 4 s / 0.5 s。
打分只用到 4 s 以内，所以和 navtest 评测时给 10 个点结果一样。

推理失败（action token 不够等）或没跑到的 token：默认用匀速直行轨迹顶上（--fallback cv），数量会打印出来并写进
<out 同名>.json；--fallback none 则不放进 submission（本地打分会直接少这些 token）。
"""
import argparse
import json
import os
import pickle

import numpy as np
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataclasses import Trajectory

SAMPLING = TrajectorySampling(time_horizon=4, interval_length=0.5)


def cv_traj(velocity):
    """匀速直行（和官方 ConstantVelocityAgent 一样：用当前纵向速度，横向为 0）。"""
    v = float(np.hypot(*velocity))
    t = np.arange(1, SAMPLING.num_poses + 1) * SAMPLING.interval_length
    return np.stack([v * t, np.zeros_like(t), np.zeros_like(t)], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--preds", required=True, help="infer_autovla.py 的 --out 目录")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fallback", default="cv", choices=["cv", "none"])
    for k in ("team-name", "authors", "email", "institution", "country"):
        ap.add_argument(f"--{k}", default="MUST_SET")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.inputs)]
    pred = {}
    for f in sorted(os.listdir(a.preds)):
        if f.startswith("preds_s") and f.endswith(".jsonl"):
            for l in open(os.path.join(a.preds, f)):
                r = json.loads(l)
                if r.get("traj") and len(r["traj"]) >= SAMPLING.num_poses:
                    pred[r["token"]] = r["traj"]

    out = {1: {}, 2: {}}
    fell = {1: [], 2: []}
    for r in rows:
        if r["token"] in pred:
            poses = np.asarray(pred[r["token"]][:SAMPLING.num_poses], dtype=np.float32)
        elif a.fallback == "cv":
            poses = cv_traj(r["velocity"]).astype(np.float32)
            fell[r["stage"]].append(r["token"])
        else:
            fell[r["stage"]].append(r["token"])
            continue
        out[r["stage"]][r["token"]] = Trajectory(poses, SAMPLING)

    sub = {"team_name": a.team_name, "authors": a.authors, "email": a.email,
           "institution": a.institution, "country / region": a.country,
           "first_stage_predictions": [out[1]], "second_stage_predictions": [out[2]]}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump(sub, f)
    info = {"n_stage1": len(out[1]), "n_stage2": len(out[2]), "fallback": a.fallback,
            "fallback_stage1": fell[1], "fallback_stage2": fell[2]}
    json.dump(info, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
    print(f"写出 {a.out}：第一阶段 {len(out[1])}，第二阶段 {len(out[2])}；"
          f"没有模型输出的 token {len(fell[1])} + {len(fell[2])}（{'匀速直行顶上' if a.fallback == 'cv' else '没放进去'}）")
    if "MUST_SET" in (a.team_name, a.authors, a.email, a.institution, a.country):
        print("注意：队名 / 作者等还是 MUST_SET，本地打分没关系，上传前要填。")


if __name__ == "__main__":
    main()
