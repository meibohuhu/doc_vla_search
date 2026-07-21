#!/usr/bin/env python3
"""
fast_nocot_patch 的 A/B 实测：先验正确性，再看提速。

做两件事：
  1. 同一批 N 个 navtest 场景，补丁前 / 后各跑一遍 nocot 提取，分别计时
  2. 逐字节 diff 两边产出的 JSON —— 不一致就说明补丁改变了语义，提速再多也不能用

计时公平性说明
--------------
baseline 先跑（冷 page cache，要真读 JPEG），patched 后跑（根本不读 JPEG）。
如果反过来测 baseline 会因页缓存变快，所以这个顺序对补丁**不利**、结论保守。
而真实场景是 103k 场景 × 32 张图 = 330 万次读，远超页缓存，冷读才是常态。

用法:
    python scripts/bench_fast_nocot.py [N]
"""
import json
import sys
import time
from pathlib import Path

# 向上找到含 setup.py 的仓库根，与本脚本所在层级无关
REPO = next(p for p in Path(__file__).resolve().parents if (p / "setup.py").exists())
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "navsim"))

from transformers import AutoProcessor

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
FILTER = "./navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"

CONFIG = {
    "pretrained_model_path": "./Qwen2.5-VL-3B-Instruct",
    "dataset_name": "nuplan",
    "dataset_path": "./dataset/nuplan/placeholder/test",
    "scene_filter": FILTER,
}


def extract(dataset, idxs):
    """跑 __getitem__ + process_sample，返回 {token: json_str} 和耗时。"""
    from tools.preprocessing.nocot_sample_generation import process_sample
    import torch

    out = {}
    t0 = time.time()
    for i in idxs:
        s = dataset[i]
        # process_sample 期望的是 collate 之后的单样本 dict，这里直接喂原始 dict
        token, res = process_sample(s, "nuplan")
        out[token] = json.dumps(res, sort_keys=True, ensure_ascii=False)
    return out, time.time() - t0


def main():
    print(f"=== fast_nocot_patch A/B (N={N}) ===\n")
    processor = AutoProcessor.from_pretrained(CONFIG["pretrained_model_path"], use_fast=True)

    from dataset_utils.preprocessing.nuplan_dataset import NuplanCoTAnnotationDataset

    print("构建 dataset (加载 147 个 test log)...")
    ds = NuplanCoTAnnotationDataset(CONFIG, processor)
    idxs = list(range(min(N, len(ds))))

    # ---------- baseline ----------
    print(f"\n[1/2] baseline（原始路径，会解码 32 张图/场景）...")
    base_out, base_t = extract(ds, idxs)
    print(f"      {base_t:.1f}s  =  {base_t/len(idxs)*1000:.0f} ms/场景")

    # ---------- patched ----------
    # 模块名不能以数字开头 -> importlib 按字符串导入
    import importlib
    apply = importlib.import_module("dataset_utils.preprocessing.0721.fast_nocot_patch").apply
    apply()
    print(f"\n[2/2] patched（跳过图像解码 / base64 / vision 预处理）...")
    fast_out, fast_t = extract(ds, idxs)
    print(f"      {fast_t:.1f}s  =  {fast_t/len(idxs)*1000:.0f} ms/场景")

    # ---------- 正确性 ----------
    print(f"\n=== 正确性 ===")
    same = base_out.keys() == fast_out.keys()
    print(f"token 集合一致 : {'✅' if same else '❌'}")
    diffs = [k for k in base_out if base_out[k] != fast_out.get(k)]
    if not diffs:
        print(f"JSON 逐字节一致: ✅  ({len(base_out)} 个样本全部相同)")
    else:
        print(f"JSON 逐字节一致: ❌  {len(diffs)}/{len(base_out)} 个不同")
        k = diffs[0]
        a, b = json.loads(base_out[k]), json.loads(fast_out[k])
        for key in a:
            if a[key] != b.get(key):
                print(f"   字段 '{key}' 不同:")
                print(f"     baseline: {str(a[key])[:150]}")
                print(f"     patched : {str(b.get(key))[:150]}")

    # ---------- 提速 ----------
    print(f"\n=== 提速 ===")
    sp = base_t / fast_t if fast_t > 0 else float("inf")
    print(f"加速比: {sp:.1f}×")
    for n, label in ((103288, "navtrain 103,288 场景"), (12146, "navtest 12,146 场景")):
        print(f"{label}:  baseline {base_t/len(idxs)*n/3600:.1f}h  ->  patched {fast_t/len(idxs)*n/3600:.1f}h")
    print("\n注: 单进程计时；实际预处理可用 --num_workers 并行，但本机 8 核已被训练任务占用。")


if __name__ == "__main__":
    main()
