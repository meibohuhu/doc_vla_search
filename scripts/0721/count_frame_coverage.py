#!/usr/bin/env python3
"""
估算：navtrain 的 445GB sensor 包，能覆盖多少个「fi=4 密集切片」出来的场景？

背景
----
AutoVLA (Table S1) 的 nuPlan 166.3k = 全部 1310 个 trainval log + frame_interval=4
（已由 count_scenes.py 实测确认 = 166,282）。
但 navtrain 的 445GB sensor 包只含它自己那 103,288 个场景所需的图像帧。

训练每个场景只需要 history+current 共 4 帧的图
（未来 10 帧只用于 GT 轨迹，取自 log 元数据，不需要图像）
=> 覆盖集 = 所有 navtrain 场景的 [center-3 .. center] 这 4 帧的并集。

于是可以纯靠 metadata 算出：fi=4 切出的场景里，有多少个的 4 帧图全都在覆盖集内。

注意：这是**上界估计**。真实包内容以下载后实测为准
（navtrain_current_* = current 帧，navtrain_history_* = 历史帧，
 命名与此假设一致，但没有官方文件清单可核对）。
"""
import pickle
from pathlib import Path

import yaml
from tqdm import tqdm

FILTER_DIR = Path("navsim/navsim/planning/script/config/common/train_test_split/scene_filter")
LOG_DIR = Path("/data/autovla_data/nuplan/navsim_logs/trainval")
NUM_HISTORY, NUM_FUTURE = 4, 10
NUM_FRAMES = NUM_HISTORY + NUM_FUTURE


def main():
    nt = yaml.safe_load(open(FILTER_DIR / "navtrain.yaml"))
    nt_lognames = set(nt["log_names"])
    nt_tokens = set(nt["tokens"])

    all_logs = sorted(p for p in LOG_DIR.iterdir() if p.suffix == ".pkl")
    nt_log_files = [p for p in all_logs if p.name.replace(".pkl", "") in nt_lognames]

    covered = set()      # navtrain 包里应有图的帧 token
    fi4_scenes = []      # (center_token, [4 帧 token]) —— 只统计 navtrain log 内的

    for p in tqdm(nt_log_files, desc="扫描 navtrain log"):
        frames = pickle.load(open(p, "rb"))
        toks = [f["token"] for f in frames]

        # (a) navtrain 场景的覆盖帧：fi=1 生成 + token 白名单过滤
        for i in range(0, len(frames), 1):
            fl = frames[i:i + NUM_FRAMES]
            if len(fl) < NUM_FRAMES:
                continue
            c = fl[NUM_HISTORY - 1]
            if len(c["roadblock_ids"]) == 0:
                continue
            if c["token"] not in nt_tokens:
                continue
            covered.update(toks[i:i + NUM_HISTORY])      # history+current 共 4 帧

        # (b) fi=4 密集切片的候选场景
        for i in range(0, len(frames), 4):
            fl = frames[i:i + NUM_FRAMES]
            if len(fl) < NUM_FRAMES:
                continue
            c = fl[NUM_HISTORY - 1]
            if len(c["roadblock_ids"]) == 0:
                continue
            fi4_scenes.append((c["token"], toks[i:i + NUM_HISTORY]))

    usable = sum(1 for _, fr in fi4_scenes if all(t in covered for t in fr))
    total = len(fi4_scenes)

    print(f"\n覆盖帧集合大小            : {len(covered):,}")
    print(f"fi=4 候选场景 (navtrain log): {total:,}")
    print(f"其中 4 帧图全在覆盖集内   : {usable:,}  ({usable/total*100:.1f}%)")
    print(f"\n对比:")
    print(f"  navtrain.yaml 标准切法  : 103,288")
    print(f"  AutoVLA Table S1        : 166,282 (全部 1310 log, fi=4)")
    print(f"  => 445GB 实际可用上界   : {usable:,}")


if __name__ == "__main__":
    main()
