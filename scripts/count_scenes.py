#!/usr/bin/env python3
"""
数 nuPlan/OpenScene 在各种 SceneFilter 配置下能切出多少个 scene。

关键前提（已核实）：navsim 的 filter_scenes() 只读 log .pkl 元数据，
sensor_blobs_path 在枚举 token 时完全不参与
(navsim/common/dataloader.py:87-90)。
=> 只要 metadata 就能把场景数定死，不需要任何 sensor 数据。

计数逻辑逐行复刻 navsim/navsim/common/dataloader.py::filter_scenes，
区别只有两点：
  1. 不保存 frame_list（省内存）
  2. 单遍扫描 —— 每个 log pkl 只 load 一次，同时评估所有配置
     （否则 14GB pickle 要扫 6 遍）
用 set 统计 distinct token，与原实现 dict[token] = frame_list 语义一致。

用法:
    python scripts/count_scenes.py --logs ./dataset/nuplan/navsim_logs/trainval
"""
import argparse
import pickle
from pathlib import Path

import yaml
from tqdm import tqdm

FILTER_DIR = Path("navsim/navsim/planning/script/config/common/train_test_split/scene_filter")
NUM_HISTORY = 4
NUM_FUTURE = 10
NUM_FRAMES = NUM_HISTORY + NUM_FUTURE  # 14


def scan_log(frames, frame_interval):
    """复刻 filter_scenes 对单个 log 的切片；yield 通过 has_route 的 center token。"""
    for i in range(0, len(frames), frame_interval):
        fl = frames[i:i + NUM_FRAMES]
        if len(fl) < NUM_FRAMES:
            continue
        center = fl[NUM_HISTORY - 1]
        if len(center["roadblock_ids"]) == 0:      # has_route=True
            continue
        yield center["token"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help="navsim_logs/<split> 目录")
    args = ap.parse_args()

    log_dir = Path(args.logs)
    all_logs = sorted(p for p in log_dir.iterdir() if p.suffix == ".pkl")

    nt = yaml.safe_load(open(FILTER_DIR / "navtrain.yaml"))
    nt_lognames = set(nt["log_names"])
    nt_tokens = set(nt["tokens"])
    nt_log_files = {p for p in all_logs if p.name.replace(".pkl", "") in nt_lognames}

    print(f"log 目录 : {log_dir}")
    print(f"总 log 数: {len(all_logs)}    其中属于 navtrain 的: {len(nt_log_files)}\n")

    # key -> (说明, 是否只用 navtrain 的 log, frame_interval, token 白名单)
    CASES = {
        "navtrain":          ("navtrain.yaml 原样【自校验：应 = 103,288】", True,  1,          nt_tokens),
        "default_fi14":      ("默认 SceneFilter (fi=14, 不重叠), 全部 log", False, NUM_FRAMES, None),
        "all_fi4":           ("fi=4, 全部 log",                            False, 4,          None),
        "all_fi1":           ("fi=1 (全重叠), 全部 log  ← 上界",            False, 1,          None),
        "ntlogs_fi4":        ("fi=4, 只用 navtrain 的 log",                 True,  4,          None),
        "ntlogs_fi1":        ("fi=1, 只用 navtrain 的 log  ← 445GB 上界",   True,  1,          None),
    }

    found = {k: set() for k in CASES}

    # ---- 单遍扫描：每个 pkl 只 load 一次 ----
    for p in tqdm(all_logs, desc="扫描 log"):
        frames = pickle.load(open(p, "rb"))
        in_nt = p in nt_log_files
        # 同一 frame_interval 只切一次，多个 case 共用
        by_fi = {}
        for key, (_, nt_only, fi, _) in CASES.items():
            if nt_only and not in_nt:
                continue
            if fi not in by_fi:
                by_fi[fi] = list(scan_log(frames, fi))
        for key, (_, nt_only, fi, toks) in CASES.items():
            if nt_only and not in_nt:
                continue
            s = found[key]
            if toks is None:
                s.update(by_fi[fi])
            else:
                s.update(t for t in by_fi[fi] if t in toks)

    print(f"\n{'配置':<16} {'scene 数':>12}   说明")
    print("-" * 78)
    for key, (desc, *_ ) in CASES.items():
        print(f"{key:<16} {len(found[key]):>12,}   {desc}")

    print("\n参照: AutoVLA Table S1  nuPlan train = 166.3k / test = 12.1k")


if __name__ == "__main__":
    main()
