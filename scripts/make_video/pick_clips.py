#!/usr/bin/env python
"""在 navtest 上找 Ours 明显好于 Baseline 的【连续帧片段】,用作对比视频素材。

PDMS 是乘法分: score = NC × DAC × (5·EP + 5·TTC + 2·C)/12
(乘法项只有 no_at_fault_collisions 和 drivable_area_compliance,见 MultiMetricIndex;
 driving_direction 的权重是 0,根本不进分数,所以不能拿它当"硬违规"。)

DAC 一归零整帧就是 0 分,progress 这种加权项根本轮不到说话。所以要凸显
【速度 reasoning】,必须按"到底是哪一项拖垮了 baseline"来分类:

  too_slow   乘法项两边都满分,只是 baseline 的 ego_progress 明显低  -> 该走不走
  too_fast   baseline 撞车 / TTC 归零,但没出界,ours 干净            -> 该停不停
  geom       baseline DAC 归零 -> 转弯几何错,不是速度问题,排除

两种模式:
  默认      在连续段里滑窗,找"命中密度高"的子窗口 —— 分差最大,但往往只有几帧。
  --whole   不切窗口,一整段连续 token 就是一个候选 —— 视频要的是能连着播的一整段,
            代价是平均分差被两边都做对的帧摊薄(那些帧正是视频需要的铺垫)。

用法:
    python scripts/make_video/pick_clips.py --top 8
    python scripts/make_video/pick_clips.py --win 4 --gap 0.15 --dprog 0.40 --frac 0.75
    python scripts/make_video/pick_clips.py --whole --min-len 16
"""
import os, glob, json, pickle, argparse
import numpy as np
import pandas as pd

LOGS = "/data/autovla_data/nuplan/navsim_logs/test"
JSON_DIR = "/data/autovla_data/nuplan/navtest_nocot"
INDEX = "/data/autovla_data/nuplan/navtest_token_index.json"
OURS_CSV = "/data/autovla_data/nuplan/sft_eval_0913_ep4_cot_full/epoch_4-loss_0_4064_merged.csv"
BASE_CSV = "/data/autovla_data/nuplan/sft_eval_0903_ep3_cot_full/epoch_3-loss_0_3547_merged.csv"

SUB = ["no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
       "time_to_collision_within_bound", "comfort", "driving_direction_compliance"]
KIND_DESC = {"too_slow": "该走不走 — baseline 乘法项全满分,纯粹是 progress 低",
             "too_fast": "该停不停 — baseline 撞车/TTC 归零,没出界"}


def token_index(path):
    """token -> (log, 在 log pkl 里的列表位置)。

    不能用 frame['frame_idx']:那是【场景内】编号(每 ~20 帧归零),同一个 log 里会重复,
    拿它排序会把不相邻的帧当成连续帧。viz_case_study 里的 129/130 也是列表位置。
    """
    if os.path.exists(path):
        return json.load(open(path))
    idx = {}
    for f in sorted(glob.glob(f"{LOGS}/*.pkl")):
        log = os.path.basename(f)[:-4]
        for i, fr in enumerate(pickle.load(open(f, "rb"))):
            idx[fr["token"]] = [log, i]
    json.dump(idx, open(path, "w"))
    print(f"index: {len(idx)} 帧 -> {path}")
    return idx


def load(csv):
    df = pd.read_csv(csv)
    df = df[df.token.notna()].drop_duplicates("token").set_index("token")
    return df[["score"] + SUB]


def classify(df, a):
    o = lambda c: df[c + "_o"]
    b = lambda c: df[c + "_b"]
    hard_o = o("no_at_fault_collisions") * o("drivable_area_compliance")
    hard_b = b("no_at_fault_collisions") * b("drivable_area_compliance")
    df["hard_o"], df["hard_b"] = hard_o, hard_b
    df["too_slow"] = ((hard_b == 1) & (hard_o == 1)
                      & (o("ego_progress") - b("ego_progress") >= a.dprog))
    df["too_fast"] = ((b("drivable_area_compliance") == 1)
                      & ((b("no_at_fault_collisions") == 0)
                         | (b("time_to_collision_within_bound") == 0))
                      & (hard_o == 1) & (o("time_to_collision_within_bound") == 1))
    df["geom"] = b("drivable_area_compliance") == 0
    return df


def main(a):
    idx = token_index(a.index)
    df = load(a.ours).join(load(a.base), lsuffix="_o", rsuffix="_b", how="inner")
    df["log"] = [idx.get(t, ["?", -1])[0] for t in df.index]
    df["fi"] = [idx.get(t, ["?", -1])[1] for t in df.index]
    df = df[df.fi >= 0]
    df["gap"] = df.score_o - df.score_b
    df = classify(df, a)
    print(f"token {len(df)}   PDMS: Ours {df.score_o.mean()*100:.2f}  "
          f"Baseline {df.score_b.mean()*100:.2f}")
    print(f"逐帧: too_slow {int(df.too_slow.sum())}  too_fast {int(df.too_fast.sum())}  "
          f"(几何错 {int(df.geom.sum())})")

    if a.whole:
        return whole_runs(df, a)

    for kind in ("too_slow", "too_fast"):
        df["hit"] = df[kind] & (df.gap > a.gap)
        wins = []
        for log, g in df.groupby("log"):
            g = g.sort_values("fi")
            fis = g.fi.tolist()
            run = [0]
            for i in range(1, len(fis)):
                if fis[i] == fis[i - 1] + 1:
                    run.append(i)
                else:
                    wins += scan(g, run, a, log, kind)
                    run = [i]
            wins += scan(g, run, a, log, kind)
        wins.sort(key=lambda w: -w["rank"])
        print(f"\n{'='*78}\n【{kind}】{KIND_DESC[kind]}  候选 {len(wins)} 段\n{'='*78}")
        show(wins[:a.top])
        out = os.path.join(a.out_dir, f"clips_{kind}.json")
        json.dump(wins[:40], open(out, "w"), indent=1)
        print(f"\n-> {out}")


def whole_runs(df, a):
    """把每一段极大连续 token 当成一个候选,按 分差 × √长度 排序。"""
    runs = []
    for log, g in df.groupby("log"):
        g = g.sort_values("fi")
        fis = g.fi.tolist()
        start = 0
        for i in range(1, len(fis) + 1):
            if i == len(fis) or fis[i] != fis[i - 1] + 1:
                s = g.iloc[start:i]
                if len(s) >= a.min_len:
                    runs.append(dict(
                        log=log, i0=int(s.fi.iloc[0]), i1=int(s.fi.iloc[-1]), n=len(s),
                        tokens=s.index.tolist(),
                        pdms_o=round(float(s.score_o.mean()) * 100, 1),
                        pdms_b=round(float(s.score_b.mean()) * 100, 1),
                        gap=round(float(s.gap.mean()) * 100, 1),
                        n_slow=int(s.too_slow.sum()), n_fast=int(s.too_fast.sum()),
                        n_story=int((s.too_slow | s.too_fast).sum()),
                        coll_b=int((s.no_at_fault_collisions_b == 0).sum()),
                        ttc_b=int((s.time_to_collision_within_bound_b == 0).sum()),
                        dac_b=int(s.geom.sum()),
                        coll_o=int((s.no_at_fault_collisions_o == 0).sum()),
                        dac_o=int((s.drivable_area_compliance_o == 0).sum())))
                start = i
    runs = [r for r in runs if r["gap"] >= a.min_gap and r["n_story"] >= a.min_story]
    runs.sort(key=lambda r: -(r["gap"] * r["n"] ** 0.5))
    print(f"\n{'='*78}\n整段连续 >= {a.min_len} 帧、平均 gap >= {a.min_gap}、"
          f"速度故事 >= {a.min_story} 帧: {len(runs)} 段\n{'='*78}")
    for k, r in enumerate(runs[:a.top], 1):
        c0, c1 = context(r["tokens"][0]), context(r["tokens"][-1])
        print(f"\n[{k}] {r['log']}  f{r['i0']}-{r['i1']}   {r['n']} 帧 / {r['n']*0.5:.1f}s")
        print(f"     PDMS  Ours {r['pdms_o']:5.1f}  Baseline {r['pdms_b']:5.1f}  "
              f"gap {r['gap']:+5.1f}     速度故事 {r['n_story']} 帧 "
              f"(慢 {r['n_slow']} / 快 {r['n_fast']})")
        print(f"     base 撞车 {r['coll_b']} · TTC0 {r['ttc_b']} · 出界 {r['dac_b']}"
              f"     ours 撞车 {r['coll_o']} · 出界 {r['dac_o']}")
        print(f"     自车 {c0['v']:.1f}→{c1['v']:.1f} m/s   "
              f"GT 5s 位移 {c0['gt']:.1f}→{c1['gt']:.1f} m   指令 {c0['instr']}")
    out = os.path.join(a.out_dir, "clips_whole.json")
    json.dump(runs[:40], open(out, "w"), indent=1)
    print(f"\n-> {out}")


def scan(g, run, a, log, kind):
    """在一段连续帧里滑窗,找命中率达标的窗口(不重叠,尽量往后延长)。"""
    if len(run) < a.win:
        return []
    sub = g.iloc[run]
    hit = sub.hit.values
    out, i = [], 0
    while i + a.win <= len(sub):
        if hit[i:i + a.win].mean() >= a.frac:
            j = i + a.win
            while j < len(sub) and hit[i:j + 1].mean() >= a.frac:
                j += 1
            s = sub.iloc[i:j]
            out.append(dict(
                kind=kind, log=log, i0=int(s.fi.iloc[0]), i1=int(s.fi.iloc[-1]), n=len(s),
                tokens=s.index.tolist(),
                pdms_o=round(float(s.score_o.mean()) * 100, 1),
                pdms_b=round(float(s.score_b.mean()) * 100, 1),
                gap=round(float(s.gap.mean()) * 100, 1),
                prog_o=round(float(s.ego_progress_o.mean()), 2),
                prog_b=round(float(s.ego_progress_b.mean()), 2),
                coll_b=int((s.no_at_fault_collisions_b == 0).sum()),
                ttc_b=int((s.time_to_collision_within_bound_b == 0).sum()),
                dac_b=int((s.drivable_area_compliance_b == 0).sum()),
                hard0_o=int((s.hard_o == 0).sum()),
                rank=round(float(s.gap.mean()) * 100 * min(len(s), 16) ** 0.5, 1)))
            i = j
        else:
            i += 1
    return out


def show(wins):
    for k, w in enumerate(wins, 1):
        c = [context(t) for t in w["tokens"]]
        v = [x["v"] for x in c]
        gt = [x["gt"] for x in c]
        print(f"\n[{k}] {w['log']}  frames {w['i0']}-{w['i1']}  "
              f"({w['n']} 帧 / {w['n']*0.5:.1f}s)")
        print(f"     PDMS  Ours {w['pdms_o']:5.1f}   Baseline {w['pdms_b']:5.1f}   "
              f"gap {w['gap']:+5.1f}")
        print(f"     progress  Ours {w['prog_o']:.2f} / Base {w['prog_b']:.2f}"
              f"     base 撞车 {w['coll_b']} 帧 · TTC0 {w['ttc_b']} 帧 · 出界 {w['dac_b']} 帧"
              f"     ours 硬违规 {w['hard0_o']} 帧")
        print(f"     自车速度 {min(v):.1f}→{max(v):.1f} m/s   "
              f"GT 5s 位移 {min(gt):.1f}~{max(gt):.1f} m   指令 {c[0]['instr']}")


_ctx = {}


def context(tok):
    if tok not in _ctx:
        j = json.load(open(f"{JSON_DIR}/{tok}.json"))
        gt = np.array(j["gt_trajectory"])[:, :2]
        _ctx[tok] = dict(v=float(np.hypot(*j["velocity"][:2])), instr=j["instruction"],
                         gt=float(np.hypot(*gt[-1])))
    return _ctx[tok]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ours", default=OURS_CSV, help="Ours 的 merged CSV")
    p.add_argument("--base", default=BASE_CSV, help="Baseline 的 merged CSV")
    p.add_argument("--index", default=INDEX, help="token->(log,帧位置) 索引,不存在就现建")
    p.add_argument("--out-dir", default="logs/0920_nvidia")
    p.add_argument("--win", type=int, default=6, help="最短窗口(帧), 6 帧 = 3s")
    p.add_argument("--frac", type=float, default=0.8, help="窗口内命中帧占比下限")
    p.add_argument("--gap", type=float, default=0.20, help="单帧 PDMS 差下限")
    p.add_argument("--dprog", type=float, default=0.45, help="too_slow: progress 差下限")
    p.add_argument("--whole", action="store_true", help="整段模式:不切窗口,整段连续帧就是候选")
    p.add_argument("--min-len", type=int, default=16, help="整段模式:最少帧数")
    p.add_argument("--min-gap", type=float, default=10.0, help="整段模式:整段平均 PDMS 差下限")
    p.add_argument("--min-story", type=int, default=4, help="整段模式:速度故事帧数下限")
    p.add_argument("--top", type=int, default=8)
    main(p.parse_args())
