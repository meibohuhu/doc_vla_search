#!/usr/bin/env python3
"""
真正检查 sensor 数据是否够用：按 scene_filter 抽样场景，逐张核对图片文件存在。

为什么需要这个
--------------
`download_nuplan_autovla.sh verify` 只比对 **log 目录名**是否齐全，
而 navtrain 的 `current_*` 分片就会为全部 1192 个 log 建好目录 ——
即使 `history_*` 分片完全缺失（历史帧一张图都没有），那个 verify 也会报 ✅。
这是一次**假通过**：目录在 ≠ 图片在。

本脚本按真实训练需求核对：每个场景取 num_history_frames 帧 × 8 个相机，
逐个 os.path.exists。这正是 nuplan_dataset.py 预处理时会读的那些文件。

用法:
    python scripts/0721/verify_sensor_coverage.py navtrain [抽样数]
    python scripts/0721/verify_sensor_coverage.py navtest  [抽样数]
"""
import os
import pickle
import random
import sys
from collections import Counter
from pathlib import Path

import yaml
from tqdm import tqdm

REPO = next(p for p in Path(__file__).resolve().parents if (p / "setup.py").exists())
FILTER_DIR = REPO / "navsim/navsim/planning/script/config/common/train_test_split/scene_filter"
DATA = Path("/data/autovla_data/nuplan")

NUM_HISTORY, NUM_FUTURE = 4, 10
NUM_FRAMES = NUM_HISTORY + NUM_FUTURE
SPLIT_OF = {"navtrain": "trainval", "navtest": "test", "navmini": "mini"}


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "navtrain"
    n_sample = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    split = SPLIT_OF[name]

    cfg = yaml.safe_load(open(FILTER_DIR / f"{name}.yaml"))
    tokens = set(cfg["tokens"])
    lognames = set(cfg["log_names"])
    log_dir, sensor_dir = DATA / "navsim_logs" / split, DATA / "sensor_blobs" / split

    print(f"filter    : {name}  ({len(tokens):,} scenes, {len(lognames)} logs)")
    print(f"sensor    : {sensor_dir}")

    # 先随机挑若干 log，再从中收集场景，避免把 1192 个 pkl 全读一遍
    logs = [p for p in sorted(log_dir.glob("*.pkl")) if p.stem in lognames]
    random.seed(0)
    random.shuffle(logs)

    scenes = []          # (log_name, [该场景 4 帧的 cams dict])
    for p in logs:
        if len(scenes) >= n_sample:
            break
        frames = pickle.load(open(p, "rb"))
        for i in range(len(frames)):
            fl = frames[i:i + NUM_FRAMES]
            if len(fl) < NUM_FRAMES:
                break
            if fl[NUM_HISTORY - 1]["token"] not in tokens:
                continue
            scenes.append((p.stem, [f["cams"] for f in fl[:NUM_HISTORY]]))
            if len(scenes) >= n_sample:
                break

    print(f"抽样      : {len(scenes)} 个场景 × {NUM_HISTORY} 帧 × 8 相机\n")

    missing_by_cam, missing_by_frame = Counter(), Counter()
    checked = missing = 0
    bad_scenes = set()
    for logname, framecams in tqdm(scenes, desc="核对图片", leave=False):
        for fi, cams in enumerate(framecams):     # fi=0..2 历史, fi=3 当前
            for cam, meta in cams.items():
                checked += 1
                if not (sensor_dir / meta["data_path"]).exists():
                    missing += 1
                    missing_by_cam[cam] += 1
                    missing_by_frame[fi] += 1
                    bad_scenes.add(logname)

    print(f"核对图片  : {checked:,}")
    print(f"缺失      : {missing:,}  ({missing/checked*100:.2f}%)")
    print(f"受影响场景: {len(bad_scenes)} 个 log\n")

    if missing:
        print("按帧位置（0-2=历史帧, 3=当前帧）:")
        for fi in sorted(missing_by_frame):
            tag = "当前帧" if fi == NUM_HISTORY - 1 else f"历史帧 t-{NUM_HISTORY-1-fi}"
            print(f"  frame[{fi}] {tag:10s}: {missing_by_frame[fi]:,}")
        print("\n按相机:")
        for cam, c in missing_by_cam.most_common():
            print(f"  {cam}: {c:,}")
        print("\n❌ sensor 数据不完整。若缺失集中在历史帧，说明 navtrain_history_* 分片未下全。")
        sys.exit(1)
    else:
        print("✅ 抽样范围内图片全部存在")


if __name__ == "__main__":
    main()
