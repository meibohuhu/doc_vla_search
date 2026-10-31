"""改了标注以后，不重跑模型、直接用已有的回答重新打分。

    python tools/navsafe_vlm/rescore.py --labels labels_v2 --task vru <results/*.jsonl> ...

--task vru 时回答里的 BLOCKED 也当 NONE（五类 prompt 的结果拿来和只判人车动物的 prompt 比）。
"""
import argparse
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe import TASKS, report  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--labels", default="labels_v2")
    ap.add_argument("--task", default="all", choices=list(TASKS))
    a = ap.parse_args()
    gt = {(r, t, f): frozenset("NONE" if x in TASKS[a.task] else x for x in ok)
          for r, t, f, ok in importlib.import_module(a.labels).frames()}
    for p in a.results:
        rows = []
        for r in map(json.loads, open(p)):
            k = (r["run"], r["token"], r["frame"])
            if k not in gt:
                continue
            pred = "NONE" if r["pred"] in TASKS[a.task] else r["pred"]
            rows.append({**r, "gt": "|".join(sorted(gt[k])), "pred": pred, "ok": pred in gt[k]})
        miss = len(gt) - len(rows)
        report(os.path.basename(p) + (f"  （{miss} 帧没有回答）" if miss else ""), rows)


if __name__ == "__main__":
    main()
