"""画 cf 开环评测（tools/cf_data/eval_cf_openloop.py）的例子：前视图 | 轨迹（ego 系）| 速度曲线。

每张图 4 条轨迹：
  default  不给 suggestion（普通 reason_then_act prompt）时模型的轨迹
  follow   给 Scene report + Suggested action 后模型的轨迹
  cf 目标  cf 索引里这条反事实的轨迹（训练时的答案）
  专家     原 JSON 的 gt_trajectory

    python tools/cf_data/render_eval_examples.py --eval_dir <eval_cf_openloop 的输出目录> --out <png 目录> \
        --per_cls 3 --cls STOP DECELERATE ACCELERATE
每类挑 per_cls 条"因建议改变"的（默认不这样、给了建议照做），再挑 1 条"默认就这样"的作对照。
"""
import argparse
import glob
import json
import os
import random
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_cf_openloop import follows  # noqa: E402

JSON_DIR = "/shared/rc/llm-gen-agent/mhu/auto_vla/nuplan/navtrain_cot_val"
SENSOR = "/shared/rc/llm-gen-agent/mhu/auto_vla/nuplan/sensor_blobs"
# 图里全用英文：本机没有同时含中英文字形的字体
STYLE = {"default": ("tab:gray", "-", "default (no suggestion)"),
         "follow": ("tab:red", "-", "follow (with suggestion)"),
         "cf": ("tab:green", "--", "cf target"),
         "expert": ("tab:blue", ":", "expert")}


def speeds(traj):
    xy = np.vstack([[0.0, 0.0], np.asarray(traj)[:, :2]])
    return np.linalg.norm(np.diff(xy, axis=0), axis=1) * 2.0     # 0.5 s 一个点


def short(text, n=150):
    t = " ".join((text or "").replace("<answer>", "").split())
    t = t.split("The final output action is")[0].strip()
    return t[:n] + ("…" if len(t) > n else "")


def render(r, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.image import imread

    d = json.load(open(os.path.join(JSON_DIR, f"{r['token']}.json")))
    img = d["front_camera_paths"][-1]
    img = img if os.path.isabs(img) else os.path.join(SENSOR, img)
    trajs = {"default": r["default"]["traj"], "follow": r["follow"]["traj"],
             "cf": r["gt_traj"], "expert": r.get("expert_traj") or d.get("gt_trajectory")}

    fig = plt.figure(figsize=(18, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 0.8, 1.0])
    ax = fig.add_subplot(gs[0])
    ax.imshow(imread(img)); ax.axis("off")

    ax = fig.add_subplot(gs[1])                       # ego 系：x 向前、y 向左 → 画成 (-y, x)
    for k, tr in trajs.items():
        if tr is None:
            continue
        c, ls, lab = STYLE[k]
        p = np.vstack([[0, 0], np.asarray(tr)[:, :2]])
        ax.plot(-p[:, 1], p[:, 0], ls, color=c, lw=2, marker="o", ms=3, label=lab)
    ax.plot(0, 0, "ks", ms=8)
    ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=0.3)
    ax.set_xlabel("lateral (m, right +)"); ax.set_ylabel("longitudinal (m)"); ax.legend(fontsize=8, loc="lower right")

    ax = fig.add_subplot(gs[2])
    t = np.arange(1, 11) * 0.5
    for k, tr in trajs.items():
        if tr is None:
            continue
        c, ls, lab = STYLE[k]
        ax.plot(t, speeds(tr), ls, color=c, lw=2, marker="o", ms=3, label=lab)
    ax.axhline(r["v0"], color="k", lw=0.8, alpha=0.4)
    ax.set_xlabel("t (s)"); ax.set_ylabel("speed (m/s)"); ax.grid(alpha=0.3); ax.set_ylim(bottom=0)

    fp, fk = follows(r["cls"], r["follow"]["plan"], r["follow"]["traj"])
    dp, _ = follows(r["cls"], r["default"]["plan"], r["default"]["traj"])
    fig.suptitle(
        f"[{r['cls']}]  token {r['token']}   v0={r['v0']:.1f} m/s   "
        f"follow: PLAN {'OK' if fp else 'NO'}, traj {'OK' if fk else 'NO'}   default already does it: {'yes' if dp else 'no'}\n"
        f"Scene report: {r['report']}   Suggested action: {r['suggestion']}\n"
        f"follow  → {short(r['follow']['text'])}\n"
        f"default → {short(r['default']['text'])}",
        fontsize=10, x=0.01, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.80))
    fig.savefig(out_png, dpi=80)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cls", nargs="+", default=["STOP", "DECELERATE", "ACCELERATE"])
    ap.add_argument("--per_cls", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    recs = []
    for f in sorted(glob.glob(os.path.join(args.eval_dir, "rows_shard*.jsonl"))):
        recs += [json.loads(l) for l in open(f)]
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    made = []
    for cls in args.cls:
        rs = [r for r in recs if r["cls"] == cls]
        changed = [r for r in rs if follows(cls, *[r["follow"][k] for k in ("plan", "traj")])[1]
                   and not follows(cls, *[r["default"][k] for k in ("plan", "traj")])[0]]
        same = [r for r in rs if follows(cls, *[r["default"][k] for k in ("plan", "traj")])[0]]
        pick = [("changed", r) for r in rng.sample(changed, min(args.per_cls, len(changed)))]
        pick += [("same", r) for r in rng.sample(same, min(1, len(same)))]
        for j, (kind, r) in enumerate(pick):
            png = os.path.join(args.out, f"{cls}_{j}_{kind}_{r['token']}.png")
            render(r, png)
            made.append(png)
            print(png)
    print(f"共 {len(made)} 张 -> {args.out}")


if __name__ == "__main__":
    main()
