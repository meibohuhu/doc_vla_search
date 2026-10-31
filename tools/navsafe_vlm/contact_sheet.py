"""把一个场景的若干帧拼成缩略图，用来人工标注。

    python tools/navsafe_vlm/contact_sheet.py <run> <token> <out.jpg> [--step 20 | --frames 10,15,20] [--cols 4]
    python tools/navsafe_vlm/contact_sheet.py <run> <token> <out.jpg> --frames 10,15 --views l0,f0,r0   # 每帧一行三路相机

有 frames/<f>/cam_*.jpg（--vis-cameras 跑的）就用它，否则用 cam/cam_f0_*.jpg；前视图都要去掉叠加。
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clean_overlay import clean  # noqa: E402

RUNS = "/scratch/mh2803/vla/navsafe/navsafe_runs"


def load(run, tok, f, cam):
    raw = f"{RUNS}/{run}/scenarios/{tok}/frames/{f:05d}/{cam}.jpg"
    if not os.path.exists(raw):
        raw = f"{RUNS}/{run}/scenarios/{tok}/cam/{cam}_{f:05d}.jpg"
    # 只有前视图叠了走廊；l0、r0 是原图
    return clean(cv2.imread(raw)) if cam == "cam_f0" else cv2.imread(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("token")
    ap.add_argument("out")
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--frames", default="")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--w", type=int, default=480)
    ap.add_argument("--views", default="", help="例如 l0,f0,r0：每帧一行，按这个顺序放几路相机（忽略 --cols）")
    a = ap.parse_args()
    d = f"{RUNS}/{a.run}/scenarios/{a.token}"
    n = len(os.listdir(f"{d}/frames")) if os.path.isdir(f"{d}/frames") else len(glob.glob(f"{d}/cam/cam_f0_*.jpg"))
    frames = [int(x) for x in a.frames.split(",")] if a.frames else list(range(0, n, a.step))
    cams = [f"cam_{v}" for v in a.views.split(",")] if a.views else ["cam_f0"]
    cols = len(cams) if a.views else a.cols
    h = a.w * 9 // 16
    tiles = []
    for f in frames:
        for cam in cams:
            im = cv2.resize(load(a.run, a.token, f, cam), (a.w, h))
            cv2.rectangle(im, (0, 0), (150 if a.views else 110, 26), (0, 0, 0), -1)
            cv2.putText(im, f"f{f} {cam[4:]}" if a.views else f"f{f}", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
            tiles.append(im)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(a.out, np.vstack(rows))
    print(a.out, f"{len(frames)} frames of {n}")


if __name__ == "__main__":
    main()
