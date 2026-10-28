"""D6：把速度反事实（speed.py）和合成换道（lane_change.py）的 ok.jsonl 合成训练用的小索引。

每条一个 cf 样本：token、类别、suggestion、答案决策句、<PLAN>、10×3 轨迹（和 JSON 的 gt_trajectory 同格式）。
训练时按 token 读 navtrain_cot 的 JSON，把 gt_trajectory 换成这里的 traj（compute_targets 会现场编码成 action token）。

    python tools/cf_data/export_index.py            # 写 cf_navtrain_v1_{train,val}.npz 和 stats.json

字段（全是定长 dtype，np.load 不需要 allow_pickle）：
    token, cls（STOP/DECELERATE/KEEP/ACCELERATE/LANE_LEFT/LANE_RIGHT）, suggestion（KEEP 为空）, decision,
    plan_speed, plan_path, v0, traj (N,10,3), pdm (N,6: score nc dac ep ttc ddc), token_err,
    start_from_stop, differs_from_expert, expert_speed
"""
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.utils.rl_rewards import plan_from_track
from models.utils.teacher import speed_verb

NUPLAN = "/scratch/mh2803/vla/nuplan"
CF = f"{NUPLAN}/cf_navtrain_v1"
SPLITS = {"train": ("navtrain", "navtrain_cot"), "val": ("navtrain_val", "navtrain_cot_val")}
TOKEN_ERR_MAX = 1.0                   # 编码再解码后偏差 > 1 m 的不要（训练学的是解码后的轨迹）
LC_MIN_SEP = 0.5                      # 换道：5 s 末端和专家的横向差 < 半个车道宽 → 专家自己也换进了目标车道，不算反事实
# 换道：5 s 内离最近他车 < 2.5 m，而且比专家在同一帧的最小车距还近 → 硬挤进去，不要。
# 只看绝对值不行：相邻车道并排开时车框本来就只隔 1 m 左右，专家自己的 p5 是 0.82 m，< 2.5 m 会去掉 41%。
# 10-27 先用 2.0 / 近 1.0 m，用户看了 c18eb96197aa5b1c（cf 2.2 m、专家 2.7 m）要求也丢，收到 2.5 / 0。
LC_GAP_ABS, LC_GAP_REL = 2.5, 0.0
SUGGESTION = {"STOP": "come to a stop", "DECELERATE": "slow down", "ACCELERATE": "speed up", "KEEP": "",
              "LANE_LEFT": "change lanes to the left", "LANE_RIGHT": "change lanes to the right"}
PDM_KEYS = ("score", "nc", "dac", "ep", "ttc", "ddc")


def decision(speed: str, path: str, v0: float) -> str:
    verb = speed_verb(speed, v0)
    if path in ("LEFT", "RIGHT"):
        verb += f" and bear {path.lower()}"
    elif path in ("LANE_LEFT", "LANE_RIGHT"):
        verb += f" and change lanes to the {path[5:].lower()}"
    return f"I should {verb}: <PLAN>{speed},{path}</PLAN>"


def label(traj):
    """按训练时真正用的 float32 轨迹算标签：几乎不动的 STOP 横纵比是噪声，保存时的舍入就能让方向词翻过来。"""
    traj = np.asarray(traj, dtype=np.float32).astype(np.float64)
    return plan_from_track([(0.0, 0.0)] + [tuple(p[:2]) for p in traj])


_SYNTH = None


def _lc_gap(args):
    """换道样本：(cf 最小车距, 专家最小车距)，满 5 s。"""
    global _SYNTH
    from lane_change import LaneChangeSynth, clearance
    if _SYNTH is None:
        _SYNTH = LaneChangeSynth()
    tok, side, traj, gt = args
    mc = _SYNTH.fs.metric_cache(tok)
    e = mc.ego_state.rear_axle
    return (tok, side), clearance(mc, traj, e.x, e.y, e.heading)[0], clearance(mc, gt, e.x, e.y, e.heading)[0]


def lc_tight(split: str, json_dir: str) -> set:
    """比专家明显更贴着别的车的换道 (token, 方向)。用户 10-27 看图发现（a87edb35ddf85c5f：第 5 s 离车 0.1 m，PDM 只查 4 s）。
    同一帧左右可能都有换道，要带上方向，不然一边太贴会连带删掉另一边。"""
    from multiprocessing import Pool
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    jobs = []
    for line in open(f"{CF}/lane_change/{split}/ok.jsonl"):
        r = json.loads(line)
        jobs.append((r["token"], r["side"], r["traj"], json.load(open(f"{json_dir}/{r['token']}.json"))["gt_trajectory"]))
    with Pool(48) as pool:
        res = pool.map(_lc_gap, jobs, chunksize=8)
    return {t for t, cf, ex in res if cf < LC_GAP_ABS and ex - cf > LC_GAP_REL}


def rows(split: str, json_dir: str, drop: Counter):
    tight = lc_tight(split, json_dir)
    for kind in ("speed", "lane_change"):
        for line in open(f"{CF}/{kind}/{split}/ok.jsonl"):
            r = json.loads(line)
            if r["token_err"] > TOKEN_ERR_MAX:
                drop[f"{kind}: token_err > {TOKEN_ERR_MAX}"] += 1
                continue
            if r["pdm"]["ttc"] < 1.0:
                # 合成时只要求 NC / DAC / DDC。TTC = 0 是按当前车速 1 s 内会撞上前方的车：
                # 换道贴着目标车道的车并进去（f80f84c1127d5a59，最近 1.0 m）、KEEP 跟着减速的前车越贴越近
                drop[f"{kind}: PDM TTC = 0"] += 1
                continue
            _, speed, path = label(r["traj"])
            gt = json.load(open(f"{json_dir}/{r['token']}.json"))["gt_trajectory"]
            if kind == "lane_change" and abs(r["traj"][-1][1] - gt[-1][1]) < LC_MIN_SEP * abs(r["spacing"]):
                # t=0 专家还在本车道中间（过了 not_in_lane），但车头已经偏向目标车道，5 s 内自己换过去了
                drop[f"lane_change: 末端离专家 < {LC_MIN_SEP} 车道宽（专家自己在换道）"] += 1
                continue
            if kind == "lane_change" and (r["token"], r["side"]) in tight:
                drop[f"lane_change: 离他车 < {LC_GAP_ABS} m 且比专家近 > {LC_GAP_REL} m"] += 1
                continue
            expert_speed = label(gt)[1]
            if kind == "speed":
                cls = r["cls"]
                assert speed == cls, (r["token"], cls, speed)
                drop["(未丢) 方向词按 float32 轨迹重算后变了"] += path != r["plan"][1]
                start = bool(r["start_from_stop"])
            else:
                cls = path = f"LANE_{r['side']}"
                assert speed == r["speed"], (r["token"], speed, r["speed"])
                start = False
            yield {"token": r["token"], "cls": cls, "suggestion": SUGGESTION[cls],
                   # STOP 的动词按当前车速选，不按轨迹前 0.5 s 的平均速度：−2.5 刹车会把后者拉到 2 以下，
                   # v0 = 2.0–2.6 m/s 还在走的车会被写成 "remain stopped"（4,789 条）
                   "decision": decision(speed, path, r["v0"]), "plan_speed": speed, "plan_path": path,
                   "v0": r["v0"], "traj": r["traj"], "pdm": [r["pdm"][k] for k in PDM_KEYS],
                   "token_err": r["token_err"], "start_from_stop": start,
                   # 换道：专家没换道，都算反事实；速度：速度词和专家的不同才算
                   "differs_from_expert": kind == "lane_change" or speed != expert_speed,
                   "expert_speed": expert_speed}


def main():
    stats = {}
    for split, (src, json_dir) in SPLITS.items():
        drop = Counter()
        R = list(rows(src, f"{NUPLAN}/{json_dir}", drop))
        arr = {k: np.array([r[k] for r in R]) for k in R[0] if k not in ("traj", "pdm")}
        arr["traj"] = np.array([r["traj"] for r in R], dtype=np.float32)
        arr["pdm"] = np.array([r["pdm"] for r in R], dtype=np.float32)
        arr["v0"] = arr["v0"].astype(np.float32)
        arr["token_err"] = arr["token_err"].astype(np.float32)
        assert arr["traj"].shape[1:] == (10, 3)
        out = f"{CF}/cf_navtrain_v1_{split}.npz"
        np.savez_compressed(out, **arr)
        cls = Counter(arr["cls"].tolist())
        stats[split] = {
            "n": len(R), "frames": len(set(arr["token"].tolist())), "dropped": dict(drop),
            "cls": dict(cls.most_common()),
            "differs_from_expert": {c: int(((arr["cls"] == c) & arr["differs_from_expert"]).sum()) for c in cls},
            "start_from_stop": int(arr["start_from_stop"].sum()),
            "plan": dict(Counter(f"{s},{p}" for s, p in zip(arr["plan_speed"], arr["plan_path"])).most_common()),
            "suggestion": dict(Counter(arr["suggestion"].tolist()).most_common()),
        }
        print(f"{out}: {len(R)} 条（{stats[split]['frames']} 帧）  去掉 {dict(drop)}")
        print("  类别 " + "  ".join(f"{k} {v}" for k, v in cls.most_common()))
    json.dump(stats, open(f"{CF}/stats.json", "w"), indent=2, ensure_ascii=False)
    print(f"统计写到 {CF}/stats.json")


if __name__ == "__main__":
    main()
