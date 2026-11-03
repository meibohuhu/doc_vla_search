"""把 build_navsim_cot.py 的输出灌进样本 JSON 的 cot_output。2026-10-27 重建。

    python tools/cot_data/merge_cot_into_samples.py --src .../navtrain_nocot --cot cot_navtrain.json --dst .../navtrain_cot \
        --sensor_root /scratch/mh2803/vla/nuplan/sensor_blobs

--sensor_root 给了的话，8 路相机路径改成相对它的路径（trainval/<log>/CAM_F0/xxx.jpg），
训练 config 里 sensor_data_path 设成各机器自己的 sensor_blobs；不给就保留原来的绝对路径。

检查：src 的每个 token 都要在 cot 里（弃权的是空串）；cot 的 <PLAN> 要等于 plan_from_track(gt_trajectory)。
"""
import argparse
import json
import os
import sys
from collections import Counter

VLA_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # 仓库根目录
sys.path.insert(0, VLA_REPO)
from models.utils.rl_rewards import plan_from_track  # noqa: E402

CAMS = ["front", "front_left", "front_right", "back", "back_left", "back_right", "left", "right"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--cot", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--sensor_root", default=None)
    args = ap.parse_args()

    cot = json.load(open(args.cot))
    os.makedirs(args.dst, exist_ok=True)
    root = os.path.normpath(args.sensor_root) + "/" if args.sensor_root else None
    c = Counter()
    for f in sorted(os.listdir(args.src)):
        d = json.load(open(os.path.join(args.src, f)))
        c["n"] += 1
        text = cot.get(d["token"])
        if text is None:
            c["no_cot"] += 1
            text = ""
        if text:
            plan = text.split("<PLAN>")[1].split("</PLAN>")[0]
            lab = plan_from_track([(0.0, 0.0)] + [tuple(p[:2]) for p in d["gt_trajectory"]])
            c["plan_mismatch"] += plan != f"{lab[1]},{lab[2]}"
        else:
            c["empty"] += 1
        d["cot_output"] = text
        if root:
            for cam in CAMS:
                ps = d.get(f"{cam}_camera_paths", [])
                if any(p and not p.startswith(root) for p in ps):
                    c["path_outside_root"] += 1
                d[f"{cam}_camera_paths"] = [p[len(root):] if p and p.startswith(root) else p for p in ps]
        with open(os.path.join(args.dst, f), "w") as g:
            json.dump(d, g, indent=2, ensure_ascii=False)
    print(f"写出 {args.dst}: {c['n']} 条   有 CoT {c['n'] - c['empty']}（{100 * (c['n'] - c['empty']) / c['n']:.1f}%）"
          f"   空 {c['empty']}（其中 cot 文件里没有的 {c['no_cot']}）")
    print(f"  <PLAN> 和 gt_trajectory 对不上 {c['plan_mismatch']}   相机路径不在 sensor_root 下 {c['path_outside_root']}")
    if c["no_cot"] or c["plan_mismatch"] or c["path_outside_root"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
