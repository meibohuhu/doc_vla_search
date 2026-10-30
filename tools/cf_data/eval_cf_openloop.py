"""cf（反事实指令）SFT 的开环评测 —— docs/1026/cf_sft_autovla_plan.md §5.3。

在 cf val 索引（cf_navtrain_v1_val.npz）上，对每一条非 KEEP 的 cf 候选（suggestion = 它自己的类别）：
  follow   : prompt 带 Scene report + Suggested action（和训练里"照做"样本同格式），greedy 解码
  default  : 同一帧、普通 reason_then_act prompt（不给 suggestion）—— 模型本来会怎么开，作对照
  report   : （--report_check）固定 suggestion，7 句 report 轮流换，看 <PLAN> / 轨迹变不变
  shortcut : （--shortcut_check）prefill 一个和 suggestion 不同的 <PLAN>，看轨迹跟 <PLAN> 还是跟 suggestion

两步：
  run  —— 每个进程跑一片（--shard i --nshard n），逐条写 <out>/rows_shard{i}.jsonl（可断点续跑）
  agg  —— 合并所有分片，按类别出表，写 <out>/summary.json

判定（§2「只看 <PLAN> 里对应的那一维」）：
  速度类 suggestion（STOP / DECELERATE / ACCELERATE）看速度维；换道类（LANE_LEFT / LANE_RIGHT）看方向维。
  运动学：速度类用 rl_rewards.plan_from_track(解码轨迹) 的 SPEED；
          换道类看 2.5 s / 3.0 s 时的横向位移是否 ≥ 0.6 × 车道宽（3.7 m → 2.22 m），且方向对。
"""
import argparse
import glob
import json
import os
import random
import re
import sys
from collections import defaultdict

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "navsim"))

from dataset_utils import prompt_spec  # noqa: E402
from models.utils.rl_rewards import plan_from_track  # noqa: E402

LANE_W, LANE_FRAC = 3.7, 0.6
_PLAN_RE = re.compile(r"<PLAN>\s*([A-Z_]+)\s*,\s*([A-Z_]+)\s*</PLAN>")
SPEED_CLS = ("STOP", "DECELERATE", "ACCELERATE")
LANE_CLS = ("LANE_LEFT", "LANE_RIGHT")


def parse_plan(text):
    """<PLAN>SPEED,PATH</PLAN>，PATH 含 LANE_LEFT / LANE_RIGHT（rl_rewards.parse_plan 的词表没有换道）。"""
    m = _PLAN_RE.search(text or "")
    if not m:
        return None
    sp, pa = m.group(1), m.group(2)
    return [sp, pa] if sp in prompt_spec.PLAN_SPEED_WORDS and pa in prompt_spec.PLAN_PATH_WORDS else None


def kin_speed(traj):
    """解码轨迹 (10,≥2) → plan_from_track 的 SPEED；不可评返回 None。"""
    xy = [(0.0, 0.0)] + [(float(p[0]), float(p[1])) for p in traj]
    r = plan_from_track(xy)
    return r[1] if r else None


def lateral(traj):
    """2.5 s / 3.0 s（第 5、6 个点，0.5 s 间隔）横向位移，ego 系 y 向左为正。"""
    return float(traj[4][1]), float(traj[5][1])


def lane_ok(traj, cls):
    sign = 1.0 if cls == "LANE_LEFT" else -1.0
    return max(sign * y for y in lateral(traj)) >= LANE_FRAC * LANE_W


def follows(cls, plan, traj):
    """(plan 照做, 运动学照做)。plan 解析失败算没照做。"""
    if cls in SPEED_CLS:
        p_ok = bool(plan) and plan[0] == cls
        k_ok = kin_speed(traj) == cls
    else:
        p_ok = bool(plan) and plan[1] == cls
        k_ok = lane_ok(traj, cls)
    return p_ok, k_ok


# ─────────────────────────────────────────────────────────────────────────────
def run(args):
    import torch
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.agents.autovla_agent import AutoVLAAgent

    z = np.load(args.index, allow_pickle=True)
    rows = [i for i in range(len(z["token"])) if str(z["cls"][i]) != "KEEP"]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.limit:
        rows = rows[:args.limit]
    rows = sorted(rows[args.shard::args.nshard])

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"rows_shard{args.shard}.jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["row"] for l in open(out_path)}
    todo = [i for i in rows if i not in done]
    print(f"[cf-eval] shard {args.shard}/{args.nshard}: {len(rows)} 条，已完成 {len(done)}，待跑 {len(todo)}", flush=True)
    if not todo:
        return

    agent = AutoVLAAgent(
        trajectory_sampling=TrajectorySampling(time_horizon=5.0, interval_length=0.5),
        checkpoint_path=args.ckpt, sensor_data_path=args.sensor_data_path,
        config_path=args.eval_config, lora_conf={"use_lora": False})
    agent.initialize()
    agent.autovla.eval()
    builders = agent.get_feature_builders()

    default_cache = {}  # 同一帧的 default 只跑一次
    with open(out_path, "a") as fout, torch.no_grad():
        for n, i in enumerate(todo):
            token, cls = str(z["token"][i]), str(z["cls"][i])
            scene = json.load(open(os.path.join(args.json_dir, f"{token}.json")))
            feats = {}
            for b in builders:
                feats.update(b.compute_features(scene))
            feats["sensor_data_path"] = args.sensor_data_path

            def gen(**kw):
                traj, text = agent.autovla.predict(feats, **kw)
                traj = np.asarray(traj.cpu() if hasattr(traj, "cpu") else traj, dtype=np.float64)[:, :3]
                return traj.round(3).tolist(), text, parse_plan(text)

            sug = prompt_spec.CF_SUGGESTIONS[cls]
            rep_key = rng.choice(prompt_spec.CF_ACTION_REPORTS[cls])
            rec = {"row": i, "token": token, "cls": cls, "v0": float(z["v0"][i]),
                   "gt_plan": [str(z["plan_speed"][i]), str(z["plan_path"][i])],
                   "gt_traj": np.asarray(z["traj"][i], dtype=np.float64).round(3).tolist(),
                   "expert_traj": scene.get("gt_trajectory"), "report": rep_key, "suggestion": sug}

            t, txt, pl = gen(cot_mode="counterfactual", report=prompt_spec.CF_REPORTS[rep_key], suggestion=sug)
            rec["follow"] = {"traj": t, "text": txt, "plan": pl}

            if token not in default_cache:
                t, txt, pl = gen()  # 原行为：reason_then_act，无 suggestion
                default_cache[token] = {"traj": t, "text": txt, "plan": pl}
            rec["default"] = default_cache[token]

            if args.report_check:
                rec["report_check"] = {}
                for rk, rtext in prompt_spec.CF_REPORTS.items():
                    t, txt, pl = gen(cot_mode="counterfactual", report=rtext, suggestion=sug)
                    rec["report_check"][rk] = {"traj": t, "plan": pl}

            if args.shortcut_check:
                # 和 suggestion 冲突的 <PLAN>：速度类换成另一个速度，换道类换成 STRAIGHT
                if cls in SPEED_CLS:
                    other = rng.choice([s for s in ("STOP", "DECELERATE", "KEEP", "ACCELERATE") if s != cls])
                    forced = [other, str(z["plan_path"][i]) if str(z["plan_path"][i]) not in LANE_CLS else "STRAIGHT"]
                else:
                    forced = [str(z["plan_speed"][i]), "STRAIGHT"]
                pre = f"<PLAN>{forced[0]},{forced[1]}</PLAN>\n"
                t, txt, pl = gen(cot_mode="counterfactual", report=prompt_spec.CF_REPORTS[rep_key],
                                 suggestion=sug, prefill=pre)
                rec["shortcut"] = {"forced_plan": forced, "traj": t, "text": txt}

            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            if n % 50 == 0:
                print(f"[cf-eval] shard {args.shard}: {n + 1}/{len(todo)}  {cls}  follow_plan={rec['follow']['plan']}"
                      f"  default_plan={rec['default']['plan']}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
def agg(args):
    recs = []
    for f in sorted(glob.glob(os.path.join(args.out, "rows_shard*.jsonl"))):
        recs += [json.loads(l) for l in open(f)]
    if not recs:
        sys.exit(f"没有结果: {args.out}/rows_shard*.jsonl")
    by = defaultdict(list)
    for r in recs:
        by[r["cls"]].append(r)

    def ade(a, b):
        a, b = np.asarray(a)[:, :2], np.asarray(b)[:, :2]
        return float(np.linalg.norm(a - b, axis=1).mean())

    summary = {}
    head = (f"{'类别':<11}{'n':>6}{'PLAN可解析':>10}{'PLAN照做':>10}{'运动学照做':>11}"
            f"{'默认就这样(PLAN)':>17}{'默认就这样(运动学)':>18}{'因建议改变':>11}{'ADE→cf':>9}{'ADE→专家':>10}")
    print(head)
    for cls in SPEED_CLS + LANE_CLS:
        rs = by.get(cls, [])
        if not rs:
            continue
        f_parse = np.mean([r["follow"]["plan"] is not None for r in rs])
        fp, fk, dp, dk, chg = [], [], [], [], []
        for r in rs:
            a, b = follows(cls, r["follow"]["plan"], r["follow"]["traj"])
            c, d = follows(cls, r["default"]["plan"], r["default"]["traj"])
            fp.append(a); fk.append(b); dp.append(c); dk.append(d); chg.append(b and not d)
        ade_cf = np.mean([ade(r["follow"]["traj"], r["gt_traj"]) for r in rs])
        ade_ex = np.mean([ade(r["follow"]["traj"], r["expert_traj"]) for r in rs if r.get("expert_traj")])
        s = {"n": len(rs), "plan_parse": f_parse, "plan_follow": np.mean(fp), "kin_follow": np.mean(fk),
             "default_plan": np.mean(dp), "default_kin": np.mean(dk), "changed_by_suggestion": np.mean(chg),
             "ade_to_cf": ade_cf, "ade_to_expert": ade_ex}
        summary[cls] = {k: float(v) for k, v in s.items()}
        print(f"{cls:<11}{len(rs):>6}{f_parse:>10.1%}{np.mean(fp):>10.1%}{np.mean(fk):>11.1%}"
              f"{np.mean(dp):>17.1%}{np.mean(dk):>18.1%}{np.mean(chg):>11.1%}{ade_cf:>9.2f}{ade_ex:>10.2f}")

    # report 检查：固定 suggestion，换 report 后 <PLAN> 和 follow 时是否一致
    rc = [r for r in recs if "report_check" in r]
    if rc:
        same = [np.mean([v["plan"] == r["follow"]["plan"] for v in r["report_check"].values()]) for r in rc]
        print(f"\nreport 检查 (n={len(rc)}): 换 7 句 report 后 <PLAN> 不变的比例 = {np.mean(same):.1%}"
              f"（≈100% 说明 report 基本被忽略，见 plan §6）")
        summary["report_check_plan_unchanged"] = float(np.mean(same))

    # shortcut 检查：prefill 冲突 <PLAN> 后，轨迹跟 <PLAN> 还是跟 suggestion
    sc = [r for r in recs if "shortcut" in r and r["cls"] in SPEED_CLS]
    if sc:
        to_plan = np.mean([kin_speed(r["shortcut"]["traj"]) == r["shortcut"]["forced_plan"][0] for r in sc])
        to_sug = np.mean([kin_speed(r["shortcut"]["traj"]) == r["cls"] for r in sc])
        print(f"shortcut 检查 (速度类 n={len(sc)}): 轨迹跟 prefill 的 <PLAN> {to_plan:.1%}，跟 suggestion {to_sug:.1%}")
        summary["shortcut_traj_follows_plan"] = float(to_plan)
        summary["shortcut_traj_follows_suggestion"] = float(to_sug)

    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=2)
    print(f"\n逐条结果: {args.out}/rows_shard*.jsonl   汇总: {args.out}/summary.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "agg"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt")
    ap.add_argument("--eval_config", default=os.path.join(REPO, "config/eval/1001/autovla-navtest-eval-cot-tigris.yaml"))
    ap.add_argument("--index", default="/shared/rc/llm-gen-agent/mhu/auto_vla/nuplan/cf_navtrain_v1/cf_navtrain_v1_val.npz")
    ap.add_argument("--json_dir", default="/shared/rc/llm-gen-agent/mhu/auto_vla/nuplan/navtrain_cot_val")
    ap.add_argument("--sensor_data_path", default="/shared/rc/llm-gen-agent/mhu/auto_vla/nuplan/sensor_blobs")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="先随机抽这么多条（seed 固定），0 = 全部 3,385 条")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report_check", action="store_true")
    ap.add_argument("--shortcut_check", action="store_true")
    args = ap.parse_args()
    run(args) if args.mode == "run" else agg(args)
