"""CoLT-Drive 上测输出 <SPEED PATH PLAN> 的模型（Impromptu VLA、Senna）：公共部分 —— docs/1104/colt_eval.md §8.1。

两家的规划问答是同一套词表（Impromptu 照搬了 Senna 的格式）：
  SPEED  KEEP / ACCELERATE / DECELERATE / STOP
  PATH   STRAIGHT / RIGHT_CHANGE / LEFT_CHANGE / RIGHT_TURN / LEFT_TURN
读成 CoLT 的（纵向, 横向），直接查 gt.json 的 acceptable_pairs，口径和 eval_autovla.py 的 PLAN 读数一致：
  KEEP / ACCELERATE / DECELERATE / STOP → keep_speed / speed_up / slow_down / stop
  STRAIGHT / LEFT_TURN / RIGHT_TURN（顺着路走）→ keep_lane；LEFT_CHANGE / RIGHT_CHANGE → nudge_left / nudge_right
  --lane-readout avoid（默认）：输出换道时不管速度词，整体读成 slow_down + nudge（和 AutoVLA 同一规则）
  解析不出算错。
输入：车速 = history[-1].ego_speed 取整；导航 GO STRAIGHT / SLIGHT LEFT → go straight，TURN LEFT → turn left。
不接 VLM，和表里 AutoVLA 那行一样是零样本。

只用标准库：Senna 的老环境（torch 2.0 / transformers 4.31）里也能 import。

    python tools/colt/speed_path_plan.py agg --run-dir /scratch/mh2803/vla/colt/impromptu/vfull/3B_AD
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

DATA = "/scratch/mh2803/vla/CoLT-Drive/data"

SPEED2LON = {"KEEP": "keep_speed", "ACCELERATE": "speed_up", "DECELERATE": "slow_down", "STOP": "stop"}
PATH2LAT = {"STRAIGHT": "keep_lane", "LEFT_TURN": "keep_lane", "RIGHT_TURN": "keep_lane",
            "LEFT_CHANGE": "nudge_left", "RIGHT_CHANGE": "nudge_right"}
NAV = {"GO STRAIGHT": "go straight", "SLIGHT LEFT": "go straight", "SLIGHT RIGHT": "go straight",
       "TURN LEFT": "turn left", "TURN RIGHT": "turn right"}

CAT = {"entity_living": "living", "entity_nonliving": "object", "road_hazard": "hazard",
       "full_block": "full_block", "false_positive": "false_pos"}
POS = {"center": "center", "left_biased": "left", "right_biased": "right", "airborne": "airborne"}
GROUP_ORDER = ("living", "full_block", "object", "hazard", "false_pos")

_SP = "|".join(SPEED2LON)
_PA = "|".join(PATH2LAT)
_TAG_RE = re.compile(rf"<SPEED PATH PLAN>\s*'?\s*({_SP})\s*,\s*({_PA})")
_BARE_RE = re.compile(rf"\b({_SP})\s*,\s*({_PA})\b")
# Impromptu 7B_AD 用单图 prompt 时会写成 "SPEED:KEEP,PATH:STRAIGHT" 或 "SPEED: KEEP\nPATH: STRAIGHT"
_KV_RE = re.compile(rf"SPEED\s*:\s*({_SP})\b[\s,]*PATH\s*:\s*({_PA})\b")


# ── prompt ───────────────────────────────────────────────────────────────────
def speed_of(m):
    return int(m["history"][-1]["ego_speed"])


def impromptu_prompt(m):
    """Impromptu q6，单张前视图版（训练数据里 Mapillary 的格式：data_qa_generate/map_qa.py q6，原文照抄）。"""
    return ("<FRONT VIEW>:\n<image>\n"
            f"Your current speed is {speed_of(m)} m/s, the navigation command is {NAV[m['navigation']]},"
            " based on the understanding of the driving scene and the navigation information,"
            "what is your plan for the next three seconds?"
            "Please answer your SPEED plan and your PATH plan. SPEED includes KEEP, ACCELERATE and DECELERATE, and STOP, "
            "PATH includes STRAIGHT, RIGHT_CHANGE, LEFT_CHANGE, RIGHT_TURN, LEFT_TURN. "
            "Based on the provided image of the driving environment, "
            "For example, a correct answer format is like 'KEEP, LEFT_CHANGE'.")


SENNA_VIEWS = ["<FRONT VIEW>", "<FRONT LEFT VIEW>", "<FRONT RIGHT VIEW>",
               "<BACK LEFT VIEW>", "<BACK RIGHT VIEW>", "<BACK VIEW>"]


def senna_prompt(m, views=6):
    """Senna 规划问答（data_tools/senna_nusc_data_converter.py get_plan_qa + format_qa，原文照抄）。
    views=6：训练格式，6 个 <image>；views=1：只留前视图那一个。"""
    image_prompt = "".join(f"{v}:\n<image>\n" for v in SENNA_VIEWS[:views])
    return image_prompt + (
        f"Your current speed is {speed_of(m)} m/s, "
        f"the navigation command is '{NAV[m['navigation']]}', "
        "based on the understanding of the driving scene and the navigation information, "
        "what is your plan for the next three seconds? "
        "Please answer your SPEED plan and your PATH plan. "
        "SPEED includes KEEP, ACCELERATE and DECELERATE, and STOP, "
        "PATH includes STRAIGHT, RIGHT_CHANGE, LEFT_CHANGE, RIGHT_TURN, LEFT_TURN. "
        "For example, a correct answer format is like 'KEEP, LEFT_CHANGE'.")


# ── 数据 / 读数 ───────────────────────────────────────────────────────────────
def load_split(split, data=DATA):
    root = f"{data}/{split}"
    ids = [str(x["sample_id"] if isinstance(x, dict) else x) for x in json.load(open(f"{root}/manifest.json"))]
    meta = {s: json.load(open(f"{root}/{s}/meta.json")) for s in ids}
    acc = {s: {(p["lon_action"], p["lat_action"]) for p in json.load(open(f"{root}/{s}/gt.json"))["acceptable_pairs"]}
           for s in ids}
    return root, ids, meta, acc


def select(ids, every=1, shard=0, nshard=1):
    return sorted(ids)[::every][shard::nshard]


def group(m):
    c = CAT[m["obstacle_category"]]
    return c if c in ("living", "full_block") else f"{c}/{POS[m['position']]}"


def parse(text):
    """→ [SPEED, PATH]；依次找 <SPEED PATH PLAN> 标签、'SPEED: X, PATH: Y'、裸的 'KEEP, STRAIGHT'。"""
    m = _TAG_RE.search(text or "") or _KV_RE.search(text or "") or _BARE_RE.search(text or "")
    return [m.group(1), m.group(2)] if m else None


def to_pair(plan, mode="avoid"):
    if not plan:
        return None
    lon, lat = SPEED2LON[plan[0]], PATH2LAT[plan[1]]
    if mode == "avoid" and lat != "keep_lane":
        lon = "slow_down"
    return lon, lat


def read_rows(run_dir):
    rows = {}
    for fn in sorted(glob.glob(os.path.join(run_dir, "gen_s*.jsonl"))):
        for line in open(fn):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # 被打断时最后一行可能不完整
                continue
            rows[r["sample_id"]] = r
    return rows


def open_run(run_dir, shard):
    os.makedirs(run_dir, exist_ok=True)
    done = set(read_rows(run_dir))
    return done, open(os.path.join(run_dir, f"gen_s{shard}.jsonl"), "a")


# ── 汇总 ─────────────────────────────────────────────────────────────────────
def pct(xs):
    return 100.0 * sum(xs) / len(xs) if xs else float("nan")


def agg(args):
    root, ids, meta, acc = load_split(args.split, args.data)
    ids = select(ids, args.every)
    rows = read_rows(args.run_dir)
    if not rows:
        sys.exit(f"没有结果：{args.run_dir}/gen_s*.jsonl")
    stats = collections.defaultdict(lambda: collections.defaultdict(list))
    dist = collections.Counter()
    missing = 0
    for s in ids:
        r = rows.get(s)
        if r is None:
            missing += 1
            continue
        plan = parse(r.get("text"))  # 从原文重新解析：改了解析器不用重跑模型
        pp = to_pair(plan, args.lane_readout)
        for g in (group(meta[s]), "总计"):
            stats[g]["ok"].append(pp in acc[s])
            stats[g]["parsed"].append(pp is not None)
        dist[",".join(plan) if plan else "无法解析"] += 1
    n = len(stats["总计"]["ok"])
    print(f"\n===== {args.run_dir}：{n} 个样本" + (f"（还缺 {missing} 个没跑）" if missing else "")
          + f"，换道读数：{args.lane_readout}")
    print(f"{'组':<20}{'n':>5}{'pair acc':>10}{'可解析':>8}")
    summary = {}
    gord = lambda g: (GROUP_ORDER.index(g.split("/")[0]), g)  # noqa: E731
    for g in sorted((g for g in stats if g != "总计"), key=gord) + ["总计"]:
        st = stats[g]
        summary[g] = {"n": len(st["ok"]), "acc": pct(st["ok"]), "parsed": pct(st["parsed"])}
        print(f"{g:<20}{summary[g]['n']:>5}{summary[g]['acc']:>10.1f}{summary[g]['parsed']:>8.1f}")
    # 论文表的五列：生物 / 非生物 / 路面危险 / 堵死 / 误报
    cats = collections.defaultdict(list)
    for s in ids:
        if s in rows:
            cats[CAT[meta[s]["obstacle_category"]]].append(to_pair(parse(rows[s].get("text")), args.lane_readout) in acc[s])
    paper = ("living", "object", "hazard", "full_block", "false_pos")  # tables/colt.tex 的列顺序
    summary["_paper"] = {c: pct(cats[c]) for c in paper}
    print("论文表（Living / Nonliv. / Hazard / Block / FP / Overall）："
          + " & ".join(f"{summary['_paper'][c]:.1f}" for c in paper) + f" & {summary['总计']['acc']:.1f}")
    print("输出分布：" + "，".join(f"{k} {v}" for k, v in dist.most_common(8)))
    json.dump(summary, open(os.path.join(args.run_dir, f"summary_{args.lane_readout}.json"), "w"),
              indent=1, ensure_ascii=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["agg"])
    ap.add_argument("--run-dir", required=True, help="eval_impromptu.py / eval_senna.py 的输出目录（gen_s*.jsonl）")
    ap.add_argument("--split", default="vfull", choices=["vfull", "vclean"])
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--every", type=int, default=1, help="和 run 时一样（冒烟用 50）")
    ap.add_argument("--lane-readout", default="avoid", choices=["avoid", "plan"])
    agg(ap.parse_args())
