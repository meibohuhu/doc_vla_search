"""CoLT 上 AutoVLA 的示例图（docs/1104/colt_examples/）。读 eval_autovla.py 已跑完的结果，不跑模型。

每张图：左边是前视图，下面写 GT 可接受的动作和不接 VLM 时模型的 reasoning；
右边是 ego 系俯视轨迹（向上为前、向右为右），三条线分别是不接 VLM / 接 VLM（v28）/ GT 指令。
图例里写标签、给的 suggestion、模型的 <PLAN>、读成的 CoLT 动作和对错（--lane-readout avoid）。

    python tools/colt/render_examples.py --ids NP12_O30_center_v_full R6_O5_left_biased_v_full …
"""
import argparse
import json
import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_autovla as E  # noqa: E402

STYLE = {"none": ("#888888", "-", "no VLM"), "vlm": ("#d62728", "-", "VLM v28"),
         "oracle": ("#2ca02c", "--", "GT instr.")}
SUG = {k: E.prompt_spec.CF_SUGGESTIONS[v[1]] for k, v in E.INSTR.items()}


def reasoning(text):
    t = (text or "").split("<answer>")[0]
    t = t.replace("<think>", "").replace("</think>", "").strip()
    return t[:220] + ("…" if len(t) > 220 else "")


def render(sid, root, m, acc, labs, gen, mode, out):
    fig = plt.figure(figsize=(15, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.25, 1], wspace=0.16)
    ax = fig.add_subplot(gs[0])
    ax.imshow(Image.open(f"{root}/{sid}/front_camera.jpg"))
    ax.axis("off")
    v0 = m["history"][-1]["ego_speed"]
    ax.set_title(f"{sid}\n{m['obstacle_type']}  ({E.group(m)}, nav {m['navigation']}, v0 {v0:.1f} m/s)",
                 fontsize=11, loc="left")
    gt = ", ".join(sorted(f"{a}+{b}" for a, b in acc))
    none_txt = gen[(sid, "none")]["text"]
    ax.text(0, 1.0, "\n".join(textwrap.wrap(f"GT ok: {gt}", 150)
                              + textwrap.wrap(f"AutoVLA reasoning (no VLM): {reasoning(none_txt)}", 150)),
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))

    bx = fig.add_subplot(gs[1])
    for y in (-1.85, 1.85):
        bx.axvline(y, color="#bbbbbb", lw=1, ls=":")
    lines = []
    for src in ("none", "vlm", "oracle"):
        lab = labs[src][sid]
        r = gen[(sid, E.key_of(lab))]
        pp = E.plan_pair(r["plan"], mode)
        ok = pp in acc
        color, ls, name = STYLE[src]
        if r.get("traj"):
            t = np.array([[0, 0]] + [p[:2] for p in r["traj"]])
            bx.plot(-t[:, 1], t[:, 0], ls, color=color, lw=2.2 if src == "vlm" else 1.8, marker="o", ms=3)
        instr = "no instruction" if lab == "NONE" else f"{lab} → \"{SUG[lab]}\""
        plan = ",".join(r["plan"]) if r["plan"] else "no PLAN"
        act = "+".join(pp) if pp else "—"
        lines.append((color, f"{name}: {instr}\n   PLAN {plan} → {act}  {'✓' if ok else '✗'}"))
    bx.set_xlim(-8, 8)
    bx.set_ylim(-1, max(10, bx.get_ylim()[1]))
    bx.set_xlabel("lateral (m, right +)")
    bx.set_ylabel("forward (m)")
    bx.grid(alpha=0.3)
    for i, (color, s) in enumerate(lines):
        bx.text(-0.25, -0.13 - i * 0.12, s, transform=bx.transAxes, va="top", fontsize=8.5, color=color)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--out", default=os.path.join(E.REPO, "docs/1104/colt_examples"))
    ap.add_argument("--split", default="vfull")
    ap.add_argument("--data", default=E.DATA)
    ap.add_argument("--out-root", default=E.OUT, help="eval_autovla.py 的 --out")
    ap.add_argument("--vlm-labels", default=E.VLM_LABELS)
    ap.add_argument("--sides", default="front")
    ap.add_argument("--tag", default="autovla_cf018")
    ap.add_argument("--lane-readout", default="avoid", choices=["avoid", "plan"])
    a = ap.parse_args()
    a.every = 1
    root, ids, meta, acc = E.load_split(a.split, a.data)
    labs = E.labels_for(["none", "vlm", "oracle"], a.ids, meta, acc, a)
    run_dir = os.path.join(a.out_root, a.split, f"{a.tag}_sides-{a.sides}")
    gen = {(r["sample_id"], r["key"]): r for r in E.read_rows(run_dir)}
    os.makedirs(a.out, exist_ok=True)
    for i, sid in enumerate(a.ids):
        fn = os.path.join(a.out, f"{i:02d}_{sid}.png")
        render(sid, root, meta[sid], acc[sid], labs, gen, a.lane_readout, fn)
        print(fn)


if __name__ == "__main__":
    main()
