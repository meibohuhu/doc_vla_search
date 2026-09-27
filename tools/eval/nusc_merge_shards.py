#!/usr/bin/env python3
"""
合并 nusc_eval.py 分片评测的原始累加和 (--dump_raw 存的 .pt)，产出与单跑完全一致的
L2 / Collision 结果表（STP3 累积 + UniAD 瞬时两种口径）。

原理：PlanningMetric 的 obj_col/obj_box_col/L2 是逐 timestep 的【求和】，total 是样本【计数】。
各片是对不相交样本子集的求和 → 直接把 4 个量按片相加，再除以合计 total，
数学上等价于对全集单跑。

用法：
    python tools/eval/nusc_merge_shards.py \
        --shards a.pt b.pt c.pt \
        --output final_table.txt
"""
import argparse
from pathlib import Path

import torch
from prettytable import PrettyTable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True, help="各片 dump_raw 的 .pt")
    ap.add_argument("--output", default=None, help="结果表输出（默认只打印）")
    ap.add_argument("--tag", default="", help="标签（写进表头，如 epoch0）")
    args = ap.parse_args()

    raws = [torch.load(p, map_location="cpu") for p in args.shards]

    # 健壮性：确认这些片来自同一个 checkpoint、且 shard_id 互不重复、覆盖齐 num_shards
    ckpts = {r.get("checkpoint") for r in raws}
    assert len(ckpts) == 1, f"⚠️ 混入了不同 ckpt 的片: {ckpts}"
    num_shards = raws[0].get("num_shards")
    shard_ids = sorted(r.get("shard_id") for r in raws)
    if num_shards is not None:
        expected = list(range(num_shards))
        if shard_ids != expected:
            print(f"⚠️ 片不完整/有重复：拿到 shard_id={shard_ids}，期望 {expected}。"
                  f"结果仅覆盖这些片。")

    obj_col = sum(r["obj_col"] for r in raws)
    obj_box_col = sum(r["obj_box_col"] for r in raws)
    L2 = sum(r["L2"] for r in raws)
    total = sum(int(r["total"]) for r in raws)
    assert total > 0, "合计 total=0，没有有效样本"

    eval_result = {
        "obj_col": obj_col / total,
        "obj_box_col": obj_box_col / total,
        "L2": L2 / total,
    }

    hdr = f" [{args.tag}]" if args.tag else ""
    print(f"合并 {len(raws)} 片{hdr}  ckpt={list(ckpts)[0]}  样本合计 total={total}")

    def build(title, per_timestep):
        tab = PrettyTable()
        tab.title = title
        tab.field_names = ["metrics", "0.5s", "1.0s", "1.5s", "2.0s", "2.5s", "3.0s"]
        for key, value in eval_result.items():
            row = [key]
            for i in range(min(len(value), 6)):
                v = float(value[i]) if per_timestep else float(value[:i + 1].mean())
                row.append("%.4f" % v)
            tab.add_row(row)
        return tab

    t_stp3 = build("STP3's Definition Planning Metrics (Cumulative Average)", per_timestep=False)
    t_uniad = build("UniAD's Definition Planning Metrics (Per-Timestep)", per_timestep=True)
    print(t_stp3)
    print(t_uniad)

    if args.output:
        with open(args.output, "a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Merged Evaluation Results{hdr} - {total} samples ({len(raws)} shards)\n")
            f.write(f"Checkpoint: {list(ckpts)[0]}\n")
            f.write(f"{'='*60}\n\n")
            f.write(str(t_stp3) + "\n\n")
            f.write(str(t_uniad) + "\n")
        print(f"\n结果已存: {args.output}")


if __name__ == "__main__":
    main()
