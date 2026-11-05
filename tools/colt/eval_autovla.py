"""AutoVLA（cf SFT）在 CoLT-Drive 上输出 meta-action 并打分 —— docs/1102/experiment_design.md §5.1。

每个样本给 AutoVLA 一种 prompt，读它输出的 <PLAN>SPEED,PATH</PLAN> 作为 meta-action，映射成 CoLT 的
（纵向, 横向），直接查 gt.json 的 acceptable_pairs（和 tools/navsafe_vlm/colt_probe.py 一样，不用 LLM judge）。

指令来源（--sources，可多选）：
  none    不接 VLM：普通 reason_then_act prompt，即 AutoVLA 原本的行为
  vlm     接 VLM：v28 的标签（colt_probe.py 的缓存，--vlm-labels）按 §4.1 变成 Scene report + Suggested action；
          标签为 NONE 时不给指令，结果和 none 相同
  oracle  GT 指令：按类别、位置和 gt.json 选一个映射后可接受的标签，再同样变成指令（"底座 + GT 指令"）
同一个（样本, 标签）只跑一次，三种来源共用缓存。

输入：CoLT 只有一张前视图，没有历史帧和左右相机。这里用 §3 的备选做法，属于分布外输入：
  前视      = front_camera.jpg 复制 4 帧
  左前/右前 = 同一张前视图（--sides front，默认）或全黑图（--sides black），各 4 帧
  车速      = history[-1].ego_speed
  加速度    = |history[-1].ego_accel|（训练 JSON 里是 [ax, ay] 的模长，非负）
  导航      = GO STRAIGHT / SLIGHT LEFT → keep forward，TURN LEFT → turn left

读数：
  PLAN（主）：速度 STOP / DECELERATE / KEEP / ACCELERATE → stop / slow_down / keep_speed / speed_up；
              方向 LANE_LEFT / LANE_RIGHT → nudge_left / nudge_right，STRAIGHT / LEFT / RIGHT（顺着路走）→ keep_lane。
              解析不出 <PLAN> 算错。
              --lane-readout avoid（默认）：AutoVLA 输出 LANE_LEFT / LANE_RIGHT 时，不管速度词，整体读成
              slow_down + nudge_left / nudge_right，和 VLM 映射表里 AROUND 的动作同一口径（2026-11-05 定）；
              plan：换道时也照 <PLAN> 的速度词读（cf 数据里换道样本约 30% 配 KEEP，§4.4，会被判错）。
              注意按的是 AutoVLA 自己的 PLAN，不是 VLM 的指令：VLA 不换道或换反方向时照样按它的输出算。
  轨迹（辅）：车实际怎么开，不套 --lane-readout。速度用 plan_from_track；横向看 3 s 处的 y（向左为正），≥ 1 m 算 nudge。
              转弯样本（vfull 中 183 个）的横向读数不准。

用法（在仓库根目录；两张卡见 scripts/1104/run_colt_autovla.sh）：
  PY=/home/mh2803/miniconda3/envs/autovla_codeclean/bin/python
  $PY tools/colt/eval_autovla.py run --dry-run                    # 不加载权重、不占 GPU：打印第一个样本各指令的 prompt 和视觉输入尺寸
  CUDA_VISIBLE_DEVICES=0 $PY tools/colt/eval_autovla.py run --every 50          # 冒烟：每 50 个取 1 个
  CUDA_VISIBLE_DEVICES=0 $PY tools/colt/eval_autovla.py run --shard 0 --nshard 2
  $PY tools/colt/eval_autovla.py agg                              # 合并所有分片出表（跑到一半也能看）

run 每个进程跑一片，逐条追加到 <out>/gen_s{shard}.jsonl，中断后重跑会接着做。
agg 出表并写 <out>/summary.json。run 和 agg 要用同样的 --split / --sides / --tag。

换机器时要改的路径（都有参数）：--data（CoLT 数据）、--ckpt、--base（Qwen2.5-VL-3B 底座）、
--vlm-labels（v28 标签，只有 vlm 来源要）、--out。要在仓库根目录下跑（codebook_cache/agent_vocab.pkl 是相对路径）。
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from dataset_utils import prompt_spec  # noqa: E402
from models.utils.rl_rewards import trajectory_plan  # noqa: E402

DATA = "/scratch/mh2803/vla/CoLT-Drive/data"
OUT = "/scratch/mh2803/vla/colt/autovla"
CKPT = "/data/mh2803/navsafe/navsafe_data/model_zoo/autovla_cf018/epoch=4-loss=0.6051.ckpt"
BASE = "/data/mh2803/navsafe/navsafe_data/model_zoo/_base/qwen2.5-vl-3b-instruct"
CONFIG = os.path.join(REPO, "config/eval/1001/autovla-navtest-eval-cot-tigris.yaml")
VLM_LABELS = ("/scratch/mh2803/vla/colt/vlm_probe/results/{split}/qwen3vl-32b/"
              "final_single_v28_single_f0_01aabc_w1280.jsonl")

# §4.1：VLM 标签 → (scene report 的键, suggestion 的类别)。NONE 不给指令。
INSTR = {
    "PEDESTRIAN": ("pedestrian", "STOP"),
    "CYCLIST": ("cyclist", "DECELERATE"),
    "ANIMAL": ("animal", "STOP"),
    "AROUND_LEFT": ("obstacle", "LANE_LEFT"),
    "AROUND_RIGHT": ("obstacle", "LANE_RIGHT"),
    "BLOCKED": ("obstacle", "STOP"),
}
# §4.1：标签 → CoLT 的（纵向, 横向）。"VLM + 映射"的分数，和 colt_probe.py 的 ACTION 相同
ACTION = {
    "PEDESTRIAN": ("stop", "keep_lane"),
    "CYCLIST": ("slow_down", "keep_lane"),
    "ANIMAL": ("stop", "keep_lane"),
    "AROUND_LEFT": ("slow_down", "nudge_left"),
    "AROUND_RIGHT": ("slow_down", "nudge_right"),
    "BLOCKED": ("stop", "keep_lane"),
    "NONE": ("keep_speed", "keep_lane"),
}
SPEED2LON = {"STOP": "stop", "DECELERATE": "slow_down", "KEEP": "keep_speed", "ACCELERATE": "speed_up"}
PATH2LAT = {"STRAIGHT": "keep_lane", "LEFT": "keep_lane", "RIGHT": "keep_lane",
            "LANE_LEFT": "nudge_left", "LANE_RIGHT": "nudge_right"}
NAV = {"GO STRAIGHT": "keep forward", "SLIGHT LEFT": "keep forward", "TURN LEFT": "turn left",
       "SLIGHT RIGHT": "keep forward", "TURN RIGHT": "turn right"}
LAT_TH = 1.0  # m，轨迹读数里 3 s 处横向位移超过它算 nudge

CAT = {"entity_living": "living", "entity_nonliving": "object", "road_hazard": "hazard",
       "full_block": "full_block", "false_positive": "false_pos"}
POS = {"center": "center", "left_biased": "left", "right_biased": "right", "airborne": "airborne"}
GROUP_ORDER = ("living", "full_block", "object", "hazard", "false_pos")
# oracle 用：生物按类型给标签（只用于 GT 指令，不进 VLM prompt）；其余生物算 PEDESTRIAN
LIVING = {"Dog": "ANIMAL", "Deer": "ANIMAL", "Flock of ducks/geese": "ANIMAL",
          "Cyclist": "CYCLIST", "E-scooter rider": "CYCLIST"}
_PLAN_RE = re.compile(r"<PLAN>\s*([A-Z_]+)\s*,\s*([A-Z_]+)\s*</PLAN>")


# ── 数据 ─────────────────────────────────────────────────────────────────────
def group(m):
    c = CAT[m["obstacle_category"]]
    return c if c in ("living", "full_block") else f"{c}/{POS[m['position']]}"


def load_split(split, data=DATA):
    root = f"{data}/{split}"
    ids = [str(x["sample_id"] if isinstance(x, dict) else x) for x in json.load(open(f"{root}/manifest.json"))]
    meta = {s: json.load(open(f"{root}/{s}/meta.json")) for s in ids}
    acc = {s: {(p["lon_action"], p["lat_action"]) for p in json.load(open(f"{root}/{s}/gt.json"))["acceptable_pairs"]}
           for s in ids}
    return root, ids, meta, acc


def vlm_labels(split, path):
    path = path.format(split=split)
    if not os.path.exists(path):
        sys.exit(f"没有 VLM 标签文件 {path}：先用 colt_probe.py --single v28 在 {split} 上跑一遍，或去掉 --sources 里的 vlm")
    return {r["sample_id"]: r["final"] for r in map(json.loads, open(path))}


def oracle_label(m, acc):
    """GT 指令：生物按类型；其余按位置排优先级，取第一个映射后可接受的标签。"""
    if m["obstacle_category"] == "entity_living":
        return LIVING.get(m["obstacle_type"], "PEDESTRIAN")
    c, p = CAT[m["obstacle_category"]], m["position"]
    if c == "false_pos":
        order = ("NONE", "AROUND_LEFT", "AROUND_RIGHT", "BLOCKED")
    elif c == "full_block":
        order = ("BLOCKED", "AROUND_LEFT", "AROUND_RIGHT", "NONE")
    elif p == "left_biased":
        order = ("AROUND_RIGHT", "AROUND_LEFT", "BLOCKED", "NONE")
    else:
        order = ("AROUND_LEFT", "AROUND_RIGHT", "BLOCKED", "NONE")
    return next((lab for lab in order if ACTION[lab] in acc), order[0])


def labels_for(sources, ids, meta, acc, args):
    out = {}
    for src in sources:
        if src == "none":
            out[src] = {s: "NONE" for s in ids}
        elif src == "vlm":
            lab = vlm_labels(args.split, args.vlm_labels)
            missing = [s for s in ids if s not in lab]
            if missing:
                sys.exit(f"VLM 标签缺 {len(missing)} 个样本，例如 {missing[:3]}")
            out[src] = {s: lab[s] for s in ids}
        elif src == "oracle":
            out[src] = {s: oracle_label(meta[s], acc[s]) for s in ids}
    return out


def key_of(label):
    return "none" if label == "NONE" else label


def prompt_kwargs(key):
    if key == "none":
        return {}  # get_prompt 默认 reason_then_act
    rep, cls = INSTR[key]
    return {"cot_mode": "counterfactual", "report": prompt_spec.CF_REPORTS[rep],
            "suggestion": prompt_spec.CF_SUGGESTIONS[cls]}


def side_image(out_dir, front, sides):
    if sides == "front":
        return front
    from PIL import Image
    w, h = Image.open(front).size
    p = os.path.join(out_dir, f"black_{w}x{h}.jpg")
    if not os.path.exists(p):
        Image.new("RGB", (w, h)).save(p, quality=95)
    return p


def features(root, sid, m, out_dir, sides):
    front = f"{root}/{sid}/front_camera.jpg"
    side = side_image(out_dir, front, sides)
    h = m["history"][-1]
    return {
        "images": {"front_camera": [front] * 4, "front_left_camera": [side] * 4, "front_right_camera": [side] * 4},
        "vehicle_velocity": float(h["ego_speed"]),
        "vehicle_acceleration": abs(float(h["ego_accel"])),
        "driving_command": NAV[m["navigation"]],
        "sensor_data_path": None,  # 路径已是绝对路径
    }


# ── 读数 ─────────────────────────────────────────────────────────────────────
def parse_plan(text):
    m = _PLAN_RE.search(text or "")
    if not m or m.group(1) not in SPEED2LON or m.group(2) not in PATH2LAT:
        return None
    return [m.group(1), m.group(2)]


def lane_readout(lon, lat, mode):
    """--lane-readout：avoid = 横向是 nudge（AutoVLA 换道）时纵向一律读成 slow_down；plan = 照原样。"""
    if mode == "avoid" and lat != "keep_lane":
        lon = "slow_down"
    return lon, lat


def plan_pair(plan, mode="plan"):
    return lane_readout(SPEED2LON[plan[0]], PATH2LAT[plan[1]], mode) if plan else None


def kin_pair(traj, mode="plan"):
    if not traj:
        return None
    sp = trajectory_plan(traj)
    if sp is None:
        return None
    y3 = float(traj[5][1])
    lat = "nudge_left" if y3 >= LAT_TH else "nudge_right" if y3 <= -LAT_TH else "keep_lane"
    return lane_readout(SPEED2LON[sp[0]], lat, mode)


def follows(label, plan):
    """照做：速度类 suggestion 看速度维，换道类看方向维。"""
    if not plan:
        return False
    cls = INSTR[label][1]
    return plan[1] == cls if cls.startswith("LANE_") else plan[0] == cls


# ── 模型 ─────────────────────────────────────────────────────────────────────
def load_cfg(args):
    import yaml
    cfg = yaml.safe_load(open(args.config))
    cfg["model"]["pretrained_model_path"] = args.base
    assert cfg["model"]["use_cot"], "cf 模型是带 CoT 训的，use_cot 必须为 true"
    return cfg


def load_model(args, cfg):
    import torch
    from models.autovla import AutoVLA
    model = AutoVLA(cfg, device="cuda:0")
    sd = torch.load(args.ckpt, map_location="cpu", mmap=True, weights_only=False)["state_dict"]
    res = model.load_state_dict({k.replace("autovla.", ""): v for k, v in sd.items()}, strict=False)
    print(f"[colt-vla] 载入 {args.ckpt}：missing {len(res.missing_keys)}，unexpected {len(res.unexpected_keys)}", flush=True)
    for name, ks in (("missing", res.missing_keys), ("unexpected", res.unexpected_keys)):
        if ks:
            print(f"  {name} 例：{ks[:5]}", flush=True)
    del sd
    model.eval()
    return model


def dry_run(args, cfg, todo, root, meta, out_dir):
    """不加载权重：用 AutoVLA.get_prompt 原样拼第一个样本各指令的输入，打印 prompt 文本和视觉输入尺寸。"""
    import torch
    from transformers import AutoProcessor
    from models.autovla import AutoVLA
    m = AutoVLA.__new__(AutoVLA)
    torch.nn.Module.__init__(m)
    m.video_conf, m.use_cot = cfg["model"]["video"], cfg["model"]["use_cot"]
    m.processor = AutoProcessor.from_pretrained(args.base)
    sid = todo[0][0]
    feats = features(root, sid, meta[sid], out_dir, args.sides)
    print(f"样本 {sid}（{meta[sid]['obstacle_type']}，{group(meta[sid])}，导航 {meta[sid]['navigation']}）")
    print(f"图像：{json.dumps(feats['images'], ensure_ascii=False)[:300]} …")
    for key in sorted({k for s, k in todo if s == sid} | {"none", "AROUND_LEFT"}):
        inputs = m.get_prompt(feats, **prompt_kwargs(key))
        text = m.processor.tokenizer.decode(inputs["input_ids"][0])
        text = re.sub(r"(<\|video_pad\|>)+", lambda x: f"<|video_pad|>×{x.group(0).count('<|video_pad|>')}", text)
        print(f"\n===== key={key}  input_ids {tuple(inputs['input_ids'].shape)}  "
              f"video_grid_thw {inputs['video_grid_thw'].tolist()}\n{text}")


def run(args):
    root, ids, meta, acc = load_split(args.split, args.data)
    ids = sorted(ids)[::args.every][args.shard::args.nshard]
    labs = labels_for(args.sources, ids, meta, acc, args)
    out_dir = out_path(args)
    os.makedirs(out_dir, exist_ok=True)
    done = {(r["sample_id"], r["key"]) for r in read_rows(out_dir)}
    todo = sorted({(s, key_of(labs[src][s])) for src in args.sources for s in ids} - done)
    print(f"[colt-vla] {args.split} 片 {args.shard}/{args.nshard}：{len(ids)} 个样本，待跑 {len(todo)} 条"
          f"（已有 {len(done)} 条）→ {out_dir}", flush=True)
    cfg = load_cfg(args)
    if args.dry_run:
        return dry_run(args, cfg, todo or [(ids[0], "none")], root, meta, out_dir)
    if not todo:
        return

    import torch
    torch.manual_seed(0)
    model = load_model(args, cfg)
    fn = os.path.join(out_dir, f"gen_s{args.shard}.jsonl")
    with open(fn, "a") as f, torch.no_grad():
        for n, (sid, key) in enumerate(todo):
            feats = features(root, sid, meta[sid], out_dir, args.sides)
            rec = {"sample_id": sid, "key": key}
            try:
                traj, text = model.predict(feats, **prompt_kwargs(key))
                traj = np.asarray(traj.float().cpu() if hasattr(traj, "cpu") else traj, dtype=np.float64)[:, :3]
                rec.update(text=text, plan=parse_plan(text), traj=traj.round(3).tolist())
            except Exception as e:  # action token 不足 10 个等，记下来算错
                rec.update(text=locals().get("text"), plan=None, traj=None, error=repr(e))
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 20 == 0:
                print(f"[colt-vla] {n + 1}/{len(todo)} {sid} {key} → {rec['plan']}", flush=True)
    print(f"[colt-vla] 片 {args.shard} 完成", flush=True)


# ── 汇总 ─────────────────────────────────────────────────────────────────────
def out_path(args):
    return os.path.join(args.out, args.split, f"{args.tag}_sides-{args.sides}")


def read_rows(out_dir):
    rows = []
    for fn in sorted(glob.glob(os.path.join(out_dir, "gen_s*.jsonl"))):
        for line in open(fn):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:  # 被打断时最后一行可能不完整
                pass
    return rows


def pct(xs):
    return 100.0 * sum(xs) / len(xs) if xs else float("nan")


def agg(args):
    root, ids, meta, acc = load_split(args.split, args.data)
    ids = sorted(ids)[::args.every]
    labs = labels_for(args.sources, ids, meta, acc, args)
    out_dir = out_path(args)
    gen = {(r["sample_id"], r["key"]): r for r in read_rows(out_dir)}
    if not gen:
        sys.exit(f"没有结果：{out_dir}/gen_s*.jsonl")
    gord = lambda g: (GROUP_ORDER.index(g.split("/")[0]), g)  # noqa: E731
    summary = {}
    for src in args.sources:
        stats = collections.defaultdict(lambda: collections.defaultdict(list))
        dist = collections.defaultdict(collections.Counter)
        missing = 0
        for s in ids:
            lab = labs[src][s]
            r = gen.get((s, key_of(lab)))
            if r is None:
                missing += 1
                continue
            pp, kp = plan_pair(r["plan"], args.lane_readout), kin_pair(r["traj"])  # 轨迹读数不套换道规则：看车实际怎么开
            for g in (group(meta[s]), "总计"):
                st = stats[g]
                st["map"].append(ACTION[lab] in acc[s])
                st["plan"].append(pp in acc[s])
                st["kin"].append(kp in acc[s])
                st["parsed"].append(pp is not None)
                if lab != "NONE":
                    st["follow"].append(follows(lab, r["plan"]))
            dist[lab]["+".join(pp) if pp else "无PLAN"] += 1
        n = len(stats["总计"]["map"])
        print(f"\n===== 指令来源 {src}：{n} 个样本" + (f"（还缺 {missing} 个没跑）" if missing else "")
              + f"，换道读数：{args.lane_readout}")
        print(f"{'组':<20}{'n':>5}{'VLM+映射':>10}{'VLA PLAN':>10}{'VLA 轨迹':>10}{'PLAN 可解析':>12}{'照做率':>8}")
        summary[src] = {}
        for g in sorted((g for g in stats if g != "总计"), key=gord) + ["总计"]:
            st = stats[g]
            row = {k: pct(st[k]) for k in ("map", "plan", "kin", "parsed", "follow")}
            row["n"] = len(st["map"])
            summary[src][g] = row
            fol = f"{row['follow']:>8.1f}" if st["follow"] else f"{'—':>8}"
            print(f"{g:<20}{row['n']:>5}{row['map']:>10.1f}{row['plan']:>10.1f}{row['kin']:>10.1f}"
                  f"{row['parsed']:>12.1f}{fol}")
        print("标签 → VLA 的 PLAN（纵向+横向）：")
        for lab in ACTION:
            if dist[lab]:
                tot = sum(dist[lab].values())
                top = "，".join(f"{k} {100 * v / tot:.0f}%" for k, v in dist[lab].most_common(4))
                print(f"  {lab:<13}n={tot:<5}{top}")
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1, ensure_ascii=False)
    print(f"\n逐条：{out_dir}/gen_s*.jsonl   汇总：{out_dir}/summary.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["run", "agg"])
    ap.add_argument("--split", default="vfull", choices=["vfull", "vclean"])
    ap.add_argument("--sources", nargs="+", default=["none", "vlm", "oracle"], choices=["none", "vlm", "oracle"])
    ap.add_argument("--sides", default="front", choices=["front", "black"], help="左前 / 右前两路喂什么")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--tag", default=None, help="结果子目录名，默认取 ckpt 所在目录名")
    ap.add_argument("--base", default=BASE, help="Qwen2.5-VL-3B 底座（processor 和模型结构）")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--vlm-labels", default=VLM_LABELS)
    ap.add_argument("--data", default=DATA, help="CoLT-Drive 的 data 目录（下面是 vfull / vclean）")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--every", type=int, default=1, help="每 k 个样本取 1 个（冒烟用）")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--lane-readout", default="avoid", choices=["avoid", "plan"],
                    help="AutoVLA 换道时：avoid = 读成 slow_down + nudge（默认），plan = 照 <PLAN> 的速度词")
    a = ap.parse_args()
    a.tag = a.tag or os.path.basename(os.path.dirname(os.path.abspath(a.ckpt)))
    run(a) if a.mode == "run" else agg(a)
