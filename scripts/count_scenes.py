#!/usr/bin/env python3
"""
数 nuPlan/OpenScene 各种 SceneFilter 配置下能切出多少个 scene。

关键前提（已核实）：navsim 的 filter_scenes() 只读 log .pkl 元数据，
sensor_blobs_path 在枚举 token 时完全不参与。
=> 只要 7GB 的 metadata 就能把场景数定死，不需要任何 sensor 数据。

逻辑逐行复刻 navsim/navsim/common/dataloader.py::filter_scenes，
区别只是不保存 frame_list（省内存），用 set 统计 distinct token
（与原实现的 dict[token] = frame_list 语义一致）。

用法:
    python scripts/count_scenes.py --logs ./dataset/nuplan/navsim_logs/trainval
    python scripts/count_scenes.py --logs ... --only navtrain   # 只跑校验那一项
"""
import argparse
import pickle
from pathlib import Path

import yaml
from tqdm import tqdm

FILTER_DIR = Path("navsim/navsim/planning/script/config/common/train_test_split/scene_filter")


def count(log_files, num_history=4, num_future=10, frame_interval=None,
          has_route=True, tokens=None, desc=""):
    """复刻 filter_scenes 的计数；返回 distinct token 数。"""
    num_frames = num_history + num_future
    if frame_interval is None:
        frame_interval = num_frames          # SceneFilter.__post_init__ 的默认行为
    token_set = set(tokens) if tokens is not None else None

    found = set()
    for p in tqdm(log_files, desc=desc, leave=False):
        frames = pickle.load(open(p, "rb"))
        for i in range(0, len(frames), frame_interval):
            fl = frames[i:i + num_frames]
            if len(fl) < num_frames:
                continue
            center = fl[num_history - 1]
            if has_route and len(center["roadblock_ids"]) == 0:
                continue
            tok = center["token"]
            if token_set is not None and tok not in token_set:
                continue
            found.add(tok)
    return len(found)


def load_filter(name):
    d = yaml.safe_load(open(FILTER_DIR / f"{name}.yaml"))
    return d.get("log_names"), d.get("tokens"), d.get("frame_interval")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help="navsim_logs/<split> 目录")
    ap.add_argument("--only", default=None, help="只跑某一项 (navtrain|default|fi4|fi1|navtrain_logs_fi4|navtrain_logs_fi1)")
    args = ap.parse_args()

    log_dir = Path(args.logs)
    all_logs = sorted(log_dir.iterdir())
    print(f"log 目录: {log_dir}   共 {len(all_logs)} 个 log\n")

    nt_logs, nt_tokens, nt_fi = load_filter("navtrain")
    nt_log_files = [f for f in all_logs if f.name.replace(".pkl", "") in set(nt_logs)]

    # (key, 说明, log 子集, frame_interval, token 白名单)
    cases = [
        ("navtrain",
         "navtrain.yaml 原样【校验：应 = 103,288】",
         nt_log_files, nt_fi, nt_tokens),
        ("default",
         "默认 SceneFilter (frame_interval=14, 不重叠) over 全部 log",
         all_logs, None, None),
        ("fi4",
         "frame_interval=4 over 全部 log",
         all_logs, 4, None),
        ("fi1",
         "frame_interval=1 (全重叠) over 全部 log  ← 上界",
         all_logs, 1, None),
        ("navtrain_logs_fi4",
         "frame_interval=4，但只用 navtrain 的 1192 个 log",
         nt_log_files, 4, None),
        ("navtrain_logs_fi1",
         "frame_interval=1，但只用 navtrain 的 1192 个 log  ← 决定性：445GB 能不能到 166k",
         nt_log_files, 1, None),
    ]

    print(f"{'配置':<20} {'log数':>7} {'scene数':>12}   说明")
    print("-" * 100)
    for key, desc, logs, fi, toks in cases:
        if args.only and args.only != key:
            continue
        n = count(logs, frame_interval=fi, tokens=toks, desc=key)
        print(f"{key:<20} {len(logs):>7} {n:>12,}   {desc}")


if __name__ == "__main__":
    main()
