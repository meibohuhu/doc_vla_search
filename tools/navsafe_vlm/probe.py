"""用 NavSafe 测试帧调 VLM 的危险分类 prompt。模型只加载一次，可以一次比较多个 prompt。

    /scratch/mh2803/vla/simlingo/conda_envs/qwen3vl/bin/python tools/navsafe_vlm/probe.py \
        --prompts tools/navsafe_vlm/prompts/v1.txt [v2.txt ...] --device cuda:1

输入图：scenarios/<token>/cam/cam_f0_XXXXX.jpg 去掉叠加（clean_overlay），缩到 --width，缓存在 CACHE。
输出：每个 prompt 一份 <out>/<prompt名>.jsonl（每帧回答）和终端里的混淆矩阵、错帧列表。
"""
import argparse
import collections
import json
import os
import re
import sys
import time

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RUNS = "/scratch/mh2803/vla/navsafe/navsafe_runs"
CACHE = "/scratch/mh2803/vla/navsafe/vlm_probe/images"
LABELS = ("PEDESTRIAN", "CYCLIST", "ANIMAL", "BLOCKED", "NONE")
# 是 / 否问题（v6_vehicle）：YES 记成 BLOCKED、NO 记成 NONE，只看车这一类时用
MODELS = {"qwen3vl-8b": "Qwen/Qwen3-VL-8B-Instruct", "qwen3vl-32b": "/scratch/mh2803/vla/checkpoints/Qwen3-VL-32B-Instruct"}


def image(run, tok, f, width, cam="cam_f0"):
    """cam_f0 / cam_l0 / cam_r0。
    --vis-cameras CAM_L0,CAM_R0 跑的结果在 frames/<f>/ 下存了三路图：l0、r0 是干净的，f0 仍然叠了走廊
    （除非跑的时候设了 NAVSAFE_NO_OVERLAY=1，这时设 NAVSAFE_RAW_F0=1 直接用）。旧的结果只有 cam/cam_f0_*.jpg。"""
    p = f"{CACHE}/w{width}/{run}/{tok}/{cam}_f{f:05d}.jpg"
    if not os.path.exists(p):
        # 要 cv2；qwen3vl 环境没有，先在 autovla_codeclean 里跑一遍 --prepare 把缓存建好
        import cv2
        raw = f"{RUNS}/{run}/scenarios/{tok}/frames/{f:05d}/{cam}.jpg"
        if cam != "cam_f0":
            assert os.path.exists(raw), f"{run}/{tok} f{f} 没有存 {cam}（要用 --vis-cameras CAM_L0,CAM_R0 重跑）"
            im = cv2.imread(raw)
        elif os.environ.get("NAVSAFE_RAW_F0"):
            im = cv2.imread(raw)
        else:
            from clean_overlay import clean
            if not os.path.exists(raw):
                raw = f"{RUNS}/{run}/scenarios/{tok}/cam/cam_f0_{f:05d}.jpg"
            im = clean(cv2.imread(raw))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        h = round(im.shape[0] * width / im.shape[1])
        cv2.imwrite(p, cv2.resize(im, (width, h), interpolation=cv2.INTER_AREA))
    return Image.open(p).convert("RGB")


# 分任务判时，标注里不归这个 prompt 管的类别改成 NONE
TASKS = {"all": (), "vru": ("BLOCKED",), "blocked": ("PEDESTRIAN", "CYCLIST", "ANIMAL")}

ALIASES = {"OBSTACLE": "BLOCKED", "VEHICLE": "BLOCKED", "YES": "BLOCKED", "NO": "NONE"}   # v4 把 BLOCKED 拆成"前方有车"   # simlingo 的 taskA prompt 用 OBSTACLE


_SPEED = {}


def ego_speed(run, tok, f):
    """vehicle_states.npy 是每帧（10 Hz）自车位置，车速 = 相邻两帧的位移 × 10。"""
    if (run, tok) not in _SPEED:
        import numpy as np
        p = np.load(f"{RUNS}/{run}/scenarios/{tok}/vehicle_states.npy")[:, :2]
        v = np.r_[0.0, np.linalg.norm(np.diff(p, axis=0), axis=1) * 10]
        _SPEED[(run, tok)] = v
    v = _SPEED[(run, tok)]
    return float(v[min(max(f, 1), len(v) - 1)])


def parse(text):
    hits = [(m.start(), ALIASES.get(lab, lab)) for lab in LABELS + tuple(ALIASES)
            for m in [re.search(rf"\b{lab}\b", text.upper())] if m]
    if not hits:
        return "UNPARSED"
    # 先描述再给 token 的 prompt：取最后一个 <TOKEN>（描述里可能顺口提到别的词）
    br = [(m.start(), ALIASES.get(m.group(1), m.group(1))) for m in re.finditer(r"<([A-Z]+)>", text.upper())
          if m.group(1) in LABELS + tuple(ALIASES)]
    return max(br)[1] if br else min(hits)[1]


class VLM:
    def __init__(self, repo, device):
        import torch  # noqa: F401
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(repo)
        self.model = AutoModelForImageTextToText.from_pretrained(
            repo, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
        self.device = device

    def ask(self, images, prompt, max_new_tokens=8):
        msgs = [{"role": "user", "content": [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": prompt}]}]
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = self.proc(text=[text], images=images, return_tensors="pt").to(self.device)
        import torch
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        return self.proc.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def report(name, rows):
    conf = collections.Counter((r["gt"], r["pred"]) for r in rows)
    ok = sum(r["ok"] for r in rows)
    print(f"\n=== {name}: {ok}/{len(rows)} 对（{100 * ok / len(rows):.1f}%）")
    # 混淆矩阵：行 = 标注（多个可接受时写成 A|B），列 = 回答
    gts = sorted({r["gt"] for r in rows}, key=lambda g: (len(g.split("|")), g))
    cols = list(LABELS) + ["UNPARSED"]
    print(f"{'标注 / 回答':22s}" + "".join(f"{c[:5]:>7s}" for c in cols))
    for g in gts:
        n = sum(v for (gg, _), v in conf.items() if gg == g)
        print(f"{g:22s}" + "".join(f"{conf[(g, c)] or '':>7}" for c in cols) + f"   ({n})")
    bad = [r for r in rows if not r["ok"]]
    by = collections.defaultdict(list)
    for r in bad:
        by[(r["run"][:2], r["token"], r["gt"], r["pred"])].append(r["frame"])
    print("错的帧：")
    for (run, tok, gt, pred), fs in sorted(by.items()):
        print(f"  {run} {tok}  标 {gt:18s} 答 {pred:10s}  f{fs}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--model", default="qwen3vl-8b", choices=list(MODELS))
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--out", default="/scratch/mh2803/vla/navsafe/vlm_probe/results")
    ap.add_argument("--only", default="", help="只跑这些 token（逗号分隔）")
    ap.add_argument("--prepare", action="store_true", help="只建图片缓存（要 cv2）")
    ap.add_argument("--crop", default="", help="只给前方路面：x0,y0,x1,y1（相对宽高的比例），例如 0.25,0.3,0.75,0.8")
    ap.add_argument("--views", default="f0", help="给哪几路相机、按什么顺序，例如 l0,f0,r0（左前、前、右前）")
    ap.add_argument("--prev", type=int, default=0, help=">0 时多给一张 prev 帧之前的图（10 Hz），放在前面，用来判断动没动")
    ap.add_argument("--labels", default="labels_v1", help="标注模块：labels_v1（旧的单前视图 run）/ labels_v2（三路相机 run）")
    ap.add_argument("--task", default="all", choices=list(TASKS),
                    help="all：五类一起判；vru：只判人 / 自行车 / 动物（BLOCKED 帧算 NONE）；blocked：只判 BLOCKED（人车动物帧算 NONE）")
    a = ap.parse_args()
    import importlib
    items = [(r, t, f, frozenset("NONE" if x in TASKS[a.task] else x for x in ok))
             for r, t, f, ok in importlib.import_module(a.labels).frames() if not a.only or t in a.only.split(",")]
    views = [f"cam_{v}" for v in a.views.split(",")]
    side = {v: {(r, t, f): image(r, t, f, a.width, v) for r, t, f, _ in items} for v in views if v != "cam_f0"}
    ims = {(r, t, f): image(r, t, f, a.width) for r, t, f, _ in items}
    if a.crop:
        x0, y0, x1, y1 = map(float, a.crop.split(","))
        ims = {k: im.crop((int(x0 * im.width), int(y0 * im.height), int(x1 * im.width), int(y1 * im.height)))
               for k, im in ims.items()}
    if a.prev:
        ims.update({(r, t, max(f - a.prev, 0)): image(r, t, max(f - a.prev, 0), a.width) for r, t, f, _ in items})
    print(f"{len(items)} 帧，图宽 {a.width}")
    if a.prepare:
        return
    vlm = VLM(MODELS[a.model], a.device)
    os.makedirs(f"{a.out}/{a.model}", exist_ok=True)
    for pf in a.prompts:
        prompt = open(pf).read().strip()
        name = os.path.splitext(os.path.basename(pf))[0]
        rows, t0 = [], time.time()
        for run, tok, f, okset in items:
            seq = ([ims[(run, tok, max(f - a.prev, 0))]] if a.prev else []) + \
                [ims[(run, tok, f)] if v == "cam_f0" else side[v][(run, tok, f)] for v in views]
            text = vlm.ask(seq, prompt.replace("{speed}", f"{ego_speed(run, tok, f):.1f}"),
                           max_new_tokens=60 if "First write" in prompt else 8)
            pred = parse(text)
            rows.append({"run": run, "token": tok, "frame": f, "gt": "|".join(sorted(okset)), "pred": pred,
                         "ok": pred in okset, "text": text})
        with open(f"{a.out}/{a.model}/{name}_{a.labels}{'' if a.task == 'all' else '_' + a.task}_w{a.width}_prev{a.prev}{'_crop' if a.crop else ''}_{a.views.replace(',', '')}.jsonl", "w") as fo:
            for r in rows:
                fo.write(json.dumps(r) + "\n")
        report(f"{name}  ({time.time() - t0:.0f}s, {(time.time() - t0) / len(rows):.2f}s/帧)", rows)


if __name__ == "__main__":
    main()
