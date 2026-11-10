"""CoLT-Drive 上测"VLM 两步分类 + 固定映射"的分数。

第 1 步（人 / 骑车的人 / 动物）和第 2 步（绕左 / 绕右 / 堵死）都对所有样本跑一遍。
最终标签：第 1 步答了 PEDESTRIAN / CYCLIST / ANIMAL 就用第 1 步，否则用第 2 步。
这样一次就能看两步各自的表现和合起来的分数。
标签按 docs/1102/experiment_design.md 4.1 节映射成（纵向, 横向），直接查 gt.json 的 acceptable_pairs，
不需要 LLM judge。

    CUDA_VISIBLE_DEVICES=0 /scratch/mh2803/vla/simlingo/conda_envs/qwen3vl/bin/python tools/navsafe_vlm/colt_probe.py \
        --step1 tools/navsafe_vlm/prompts/v18_vru_f0.txt \
        --step2 tools/navsafe_vlm/prompts/v19_obst_f0.txt --device cuda:0

每一步的回答按（split、模型、prompt 名 + 内容哈希、图宽）逐条追加到 jsonl：
只换一步的 prompt 时，另一步直接读缓存；中途断了，重跑会接着做。

--single：七个标签合在一个 prompt 里只调一次（和两步对比用），这时不用 --step1 / --step2。
--model2：第 2 步换模型（例如第 1 步 8B、第 2 步 32B）。
--cache-only：只把给了的 prompt 跑完写进缓存、不出报告；两张卡分开跑时用，配合 --reverse 从后往前跑。
"""
import argparse
import collections
import functools
import hashlib
import json
import os
import re
import sys
import time

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe import MODELS, VLM  # noqa: E402

DATA = "/scratch/mh2803/vla/CoLT-Drive/data"
OUT = "/scratch/mh2803/vla/colt/vlm_probe/results"
VRU = ("PEDESTRIAN", "CYCLIST", "ANIMAL")
STEP_LABELS = {1: VRU + ("NONE",), 2: ("AROUND_LEFT", "AROUND_RIGHT", "BLOCKED", "NONE")}
STEP_LABELS[0] = VRU + STEP_LABELS[2]   # --single：一次判七个标签
# 4.1 节：标签 -> CoLT 的（纵向, 横向）
ACTION = {
    "PEDESTRIAN": ("stop", "keep_lane"),
    "CYCLIST": ("slow_down", "keep_lane"),
    "ANIMAL": ("stop", "keep_lane"),
    "AROUND_LEFT": ("slow_down", "nudge_left"),
    "AROUND_RIGHT": ("slow_down", "nudge_right"),
    "BLOCKED": ("stop", "keep_lane"),
    "NONE": ("keep_speed", "keep_lane"),
}
CAT = {"entity_living": "living", "entity_nonliving": "object", "road_hazard": "hazard",
       "full_block": "full_block", "false_positive": "false_pos"}
POS = {"center": "center", "left_biased": "left", "right_biased": "right", "airborne": "airborne"}
COLS = ("PEDESTRIAN", "CYCLIST", "ANIMAL", "AROUND_LEFT", "AROUND_RIGHT", "BLOCKED", "NONE")


def parse(text, labels):
    up = text.upper()
    br = [(m.start(), m.group(1)) for m in re.finditer(r"<([A-Z_]+)>", up) if m.group(1) in labels]
    if br:
        return max(br)[1]
    hits = [(m.start(), lab) for lab in labels for m in [re.search(rf"\b{lab}\b", up)] if m]
    return min(hits)[1] if hits else "UNPARSED"


def group(m):
    c = CAT[m["obstacle_category"]]
    return c if c in ("living", "full_block") else f"{c}/{POS[m['position']]}"


def load_split(split):
    root = f"{DATA}/{split}"
    ids = [str(x["sample_id"] if isinstance(x, dict) else x) for x in json.load(open(f"{root}/manifest.json"))]
    meta = {s: json.load(open(f"{root}/{s}/meta.json")) for s in ids}
    gt = {s: json.load(open(f"{root}/{s}/gt.json")) for s in ids}
    return root, ids, meta, gt


def run_step(step, pf, ids, root, meta, a, get_vlm, model):
    prompt = open(pf).read().strip()
    name = os.path.splitext(os.path.basename(pf))[0]
    h = hashlib.sha1(prompt.encode()).hexdigest()[:6]
    path = f"{a.out}/{a.split}/{model}/s{step}_{name}_{h}_w{a.width}.jsonl"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    done = {}
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:   # 进程被停时最后一行可能没写完
                continue
            done[r["sample_id"]] = r
    todo = [s for s in ids if s not in done]
    if a.reverse:
        todo = todo[::-1]
    print(f"第 {step} 步 {name}（{h}）：缓存 {len(ids) - len(todo)}，要跑 {len(todo)}  -> {path}", flush=True)
    if todo:
        vlm, t0 = get_vlm(model), time.time()
        with open(path, "a") as fo:
            for i, s in enumerate(todo):
                im = Image.open(f"{root}/{s}/front_camera.jpg").convert("RGB")
                if a.width:
                    im = im.resize((a.width, round(im.height * a.width / im.width)), Image.BICUBIC)
                speed = meta[s]["history"][-1]["ego_speed"]
                text = vlm.ask([im], prompt.replace("{speed}", f"{speed:.1f}"), max_new_tokens=a.max_new_tokens)
                r = {"sample_id": s, "pred": parse(text, STEP_LABELS[step]), "text": text}
                fo.write(json.dumps(r) + "\n")
                fo.flush()
                done[s] = r
                if (i + 1) % 200 == 0:
                    el = time.time() - t0
                    print(f"  {i + 1}/{len(todo)}  {el / (i + 1):.2f}s/张，还剩约 {el / (i + 1) * (len(todo) - i - 1) / 60:.0f} 分钟", flush=True)
    return name, h, done


def pct(k, n):
    return f"{100 * k / n:5.1f}" if n else "    -"


def report(rows, has_s2):
    n = len(rows)
    ok = sum(r["ok"] for r in rows)
    un = collections.Counter(k for r in rows for k in ("s1", "s2") if r[k] == "UNPARSED")
    print(f"\n=== VLM + 映射：{ok}/{n} = {100 * ok / n:.1f}%"
          f"  解析失败：第 1 步 {un['s1']}，第 2 步 {un['s2']}")

    print("\n按组：最终标签分布")
    print(f"{'组':20s}{'样本':>5s}{'对%':>7s}" + "".join(f"{c[:7]:>9s}" for c in COLS))
    for g in sorted({r["group"] for r in rows}):
        rs = [r for r in rows if r["group"] == g]
        cnt = collections.Counter(r["final"] for r in rs)
        print(f"{g:20s}{len(rs):5d}{pct(sum(r['ok'] for r in rs), len(rs)):>7s}"
              + "".join(f"{cnt[c] or '':>9}" for c in COLS))

    print("\n第 1 步：生物召回（答了人 / 骑车 / 动物任一个都算）")
    liv = [r for r in rows if r["group"] == "living"]
    for t in sorted({r["type"] for r in liv}):
        rs = [r for r in liv if r["type"] == t]
        cnt = collections.Counter(r["s1"] for r in rs)
        print(f"  {t:28s}{pct(sum(r['s1'] in VRU for r in rs), len(rs))}%   " + "  ".join(f"{k}:{v}" for k, v in cnt.most_common()))
    print(f"  合计 {pct(sum(r['s1'] in VRU for r in liv), len(liv))}%")
    print("第 1 步：非生物样本上的误报（答了人 / 骑车 / 动物）")
    non = [r for r in rows if r["group"] != "living"]
    fa = collections.Counter(r["type"] for r in non if r["s1"] in VRU)
    print(f"  合计 {sum(fa.values())}/{len(non)}  " + "  ".join(f"{t}:{v}" for t, v in fa.most_common()))

    if has_s2:
        print("\n第 2 步单独看（非生物样本，不管第 1 步答了什么）")
        print(f"{'组':20s}{'样本':>5s}{'对%':>7s}" + "".join(f"{c[:7]:>9s}" for c in COLS[3:]))
        for g in sorted({r["group"] for r in non}):
            rs = [r for r in non if r["group"] == g]
            cnt = collections.Counter(r["s2"] for r in rs)
            print(f"{g:20s}{len(rs):5d}{pct(sum(r['s2_ok'] for r in rs), len(rs)):>7s}"
                  + "".join(f"{cnt[c] or '':>9}" for c in COLS[3:]))
        print(f"  合计 {pct(sum(r['s2_ok'] for r in non), len(non))}%")

    print("\n最差的 15 种障碍物（最终结果）")
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["group"].split("/")[0], r["type"])].append(r)
    worst = sorted(by.items(), key=lambda kv: sum(r["ok"] for r in kv[1]) / len(kv[1]))[:15]
    for (c, t), rs in worst:
        cnt = collections.Counter(r["final"] for r in rs)
        print(f"  {c:11s}{t:28s}{pct(sum(r['ok'] for r in rs), len(rs))}%   " + "  ".join(f"{k}:{v}" for k, v in cnt.most_common()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step1", default="")
    ap.add_argument("--step2", default="", help="不给就只看第 1 步，第 1 步答 NONE 的最终都算 NONE")
    ap.add_argument("--single", default="", help="七个标签一次判完的 prompt；给了就不用 --step1 / --step2")
    ap.add_argument("--cache-only", action="store_true", help="只把给了的 prompt 跑完写进缓存、不出报告")
    ap.add_argument("--model", default="qwen3vl-8b", choices=list(MODELS), help="第 1 步（或 --single）的模型")
    ap.add_argument("--model2", default="", choices=[""] + list(MODELS), help="第 2 步的模型，不给就和 --model 一样")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--split", default="vfull", choices=["vfull", "vclean"])
    ap.add_argument("--width", type=int, default=1280, help="缩到这个宽度（原图 1376×768）；0 = 不缩")
    ap.add_argument("--every", type=int, default=1, help="每隔几个样本取一个，快速试 prompt 用")
    ap.add_argument("--max-new-tokens", type=int, default=12, help="先描述再答的 prompt 要调大，例如 80")
    ap.add_argument("--reverse", action="store_true",
                    help="从后往前跑：另一张卡上已经有一个进程从前往后跑同一个缓存时用，两边在中间汇合")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    root, ids, meta, gt = load_split(a.split)
    ids = ids[::a.every]
    m2 = a.model2 or a.model
    get_vlm = functools.lru_cache(None)(lambda m: VLM(MODELS[m], a.device))
    if a.single:
        n1, h1, s1 = run_step(0, a.single, ids, root, meta, a, get_vlm, a.model)
        n2, h2, s2 = "", "", {}
    else:
        n1, h1, s1 = run_step(1, a.step1, ids, root, meta, a, get_vlm, a.model) if a.step1 else ("", "", {})
        n2, h2, s2 = run_step(2, a.step2, ids, root, meta, a, get_vlm, m2) if a.step2 else ("", "", {})
    if a.cache_only:
        return
    assert s1, "要出报告必须给 --step1 或 --single"

    rows = []
    for s in ids:
        pairs = {(p["lon_action"], p["lat_action"]) for p in gt[s]["acceptable_pairs"]}
        ok_labels = {lab for lab, act in ACTION.items() if act in pairs}
        p1 = s1[s]["pred"]
        p2 = s2[s]["pred"] if s2 else "NONE"
        if a.single:
            final = p1 if p1 in ACTION else "NONE"
        else:
            final = p1 if p1 in VRU else (p2 if p2 in STEP_LABELS[2] else "NONE")
        rows.append({"sample_id": s, "group": group(meta[s]), "type": meta[s]["obstacle_type"],
                     "s1": p1, "s2": p2, "final": final, "pair": ACTION[final],
                     "ok": final in ok_labels, "s2_ok": p2 in ok_labels, "ok_labels": sorted(ok_labels)})
    tag = ("single_" if a.single else "") + f"{n1}_{h1}" \
        + (f"__{m2 if m2 != a.model else ''}{n2}_{h2}" if n2 else "") \
        + f"_w{a.width}" + (f"_every{a.every}" if a.every > 1 else "")
    with open(f"{a.out}/{a.split}/{a.model}/final_{tag}.jsonl", "w") as fo:
        for r in rows:
            fo.write(json.dumps(r) + "\n")
    if a.single:
        print(f"\n{a.split}  一次判完：{a.model} {n1}  {len(rows)} 个样本")
    else:
        print(f"\n{a.split}  第 1 步 {a.model} {n1}  第 2 步 {m2 + ' ' + n2 if n2 else '（无）'}  {len(rows)} 个样本")
    report(rows, bool(s2))


if __name__ == "__main__":
    main()
