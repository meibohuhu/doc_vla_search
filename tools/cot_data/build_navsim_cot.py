"""NAVSIM 规则版 CoT 生成器（零 VLM）。2026-10-27 重建。

brev 上的原版 build_navsim_cot.py 丢了，只剩它生成的 166k 数据（HF PhoenixHu/doc_vla_nuplan_166k_cot）。
这一版用那份数据反推出规则，再把反推不出来的三类重新定义。
在 60 个 log、8,621 帧上和 166k 对照：整句一致 88%，各子模式占比相差 ≤ 1 个百分点。

句式（四个槽，都可能为空）：
    [VRU]        I see pedestrians 12 meters ahead of me.
    [前车/切入]   The vehicle 8 meters ahead in my lane is stopped.
                 ／ A vehicle 23 meters on my left will move into my path.
    [理由+决策]   Because I need to stay behind it, I should come to a stop: <PLAN>STOP,STRAIGHT</PLAN>

和 166k 一致的部分（反推出来的）：
  - 弃权：t=0 往前 3 帧、往后 10 帧（2 Hz）中间缺帧 → cot 为空（166k 里 1.2%）
  - <PLAN> 和动词：plan_from_track(gt_trajectory)，和训练侧共用一个函数
  - 前车：正前 5–40 m、横向 < 4 m、同向（车 |yaw| < 90°，行人/骑行者 < 60°）里 x 最小的。
    距离写 int(x)；< 0.3 m/s stopped，< 2 m/s moving slowly，否则 moving at int(v) m/s
  - VRU：同一条走廊、20 m 内的行人/骑行者，距离写最近的 int(x)。前车是 VRU 时不再写前车句
  - 跟车 / 让行：只在 STOP / DECELERATE 时给理由。有前车 → stay behind；没有前车只有 VRU → yield

重新定义的三类（老规则反推不出来；老版切入里有不少停着不动的车）：
  - 切入：t=0 在自车路径带外（> 2 m），3 s 内真的开进路径带（< 1.5 m），进入点在自车前面、前车后面
  - 停车标志：STOP_SIGN 停车线多边形和自车路径线相交、前方 40 m 内，而且不在前车后面
  - 前方放空（只在 KEEP / ACCELERATE）：前车 ≥ 25 m、2.5 s 内提速 ≥ 1 m/s、2 s 后车距确实变大
自车路径 = GT 路径沿 GT 末点航向延长到 45 m。
理由优先级（STOP / DECELERATE）：停车标志 > 让行（无前车时）> 切入 > 跟车。
切入当理由时不再单独写切入句；有切入时第二个槽写切入，不写前车；切入的人已经在 VRU 句里时不写切入句。
示例图：docs/1026/cot_regen_examples/。说明：docs/1026/cot_regen_navtrain.md。

用法：
    python tools/cot_data/build_navsim_cot.py --json_dir /scratch/.../navtrain_nocot --out cot_navtrain.json --workers 64
然后用 merge_cot_into_samples.py 灌进样本 JSON。
"""
import argparse
import json
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
from pyquaternion import Quaternion

# 标签函数和训练侧共用（AutoVLA 仓库 models/utils/rl_rewards.py），保证 <PLAN> 和训练目标一致
VLA_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # 仓库根目录
sys.path.insert(0, VLA_REPO)
from models.utils.rl_rewards import plan_from_track  # noqa: E402

LOG_DIR = "/scratch/mh2803/vla/nuplan/navsim_logs/trainval"
MAPS_ROOT = os.environ.get("NUPLAN_MAPS_ROOT", "/scratch/mh2803/vla/nuplan/maps")
MAP_VERSION = os.environ.get("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")

HIST, FUT = 3, 10                     # 弃权窗口
VAL_STOP = 2.0
SPEED_VERB = {"DECELERATE": "slow down", "KEEP": "hold my current speed", "ACCELERATE": "speed up"}
CLS = {"vehicle": "vehicle", "pedestrian": "pedestrian", "bicycle": "cyclist"}
LEAD_X, LEAD_Y = (5.0, 40.0), 4.0
VRU_X = 20.0
STOPPED, SLOW = 0.3, 2.0
PATH_LEN, BAND_IN, BAND_OUT = 45.0, 1.5, 2.0
CUT_T, CUT_X, CUT_V = 6, 40.0, 0.5
SIGN_S = 40.0
GAP_X, GAP_DV, GAP_DT = 25.0, 1.0, 4
SIDE_OBS = {"left": "on my left", "right": "on my right", "center": "in front of me"}


# ── 坐标 ──────────────────────────────────────────────────────────────
def se2(fr):
    """和 navsim 一样只取平移 xy 和四元数的 yaw（用 4×4 矩阵会带进俯仰，停车时左右会判反）。"""
    x, y = fr["ego2global_translation"][:2]
    return float(x), float(y), Quaternion(*fr["ego2global_rotation"]).yaw_pitch_roll[0]


def to_local(o, p, xy):
    """位姿 p 的局部 xy → 全局 → 位姿 o 的局部。p = (0, 0, 0) 表示 xy 已经是全局坐标。"""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    c, s = np.cos(p[2]), np.sin(p[2])
    g = np.stack([p[0] + xy[:, 0] * c - xy[:, 1] * s, p[1] + xy[:, 0] * s + xy[:, 1] * c], axis=1) - o[:2]
    c, s = np.cos(o[2]), np.sin(o[2])
    return np.stack([g[:, 0] * c + g[:, 1] * s, -g[:, 0] * s + g[:, 1] * c], axis=1)


def window_ok(L, i):
    if i - HIST < 0 or i + FUT >= len(L):
        return False
    ts = np.array([L[j]["timestamp"] for j in range(i - HIST, i + FUT + 1)], dtype=np.float64)
    return bool(np.all(np.abs(np.diff(ts) / 1e6 - 0.5) <= 0.1))


def frame_record(L, i, m, gt):
    """t=0 的他车、他车未来 3 s（t=0 自车系）、50 m 内的 STOP_SIGN 停车线。gt 是样本 JSON 的 gt_trajectory。"""
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    fr, o = L[i], se2(L[i])
    a = fr["anns"]
    rec = {"names": np.asarray(a["gt_names"]).astype(str), "boxes": np.asarray(a["gt_boxes"], np.float64).reshape(-1, 7),
           "vel": np.asarray(a["gt_velocity_3d"], np.float64).reshape(-1, 3)[:, :2], "tracks": list(a["track_tokens"]),
           "ego_fut": [tuple(p[:2]) for p in gt], "ego_head": float(gt[-1, 2]), "fut": {}}
    for k in range(1, CUT_T + 1):
        g = L[i + k]
        if len(g["anns"]["track_tokens"]):
            xy = to_local(o, se2(g), np.asarray(g["anns"]["gt_boxes"])[:, :2])
            rec["fut"][k] = dict(zip(g["anns"]["track_tokens"], map(tuple, xy)))
    objs = m.get_proximal_map_objects(Point2D(o[0], o[1]), 50.0, [SemanticMapLayer.STOP_LINE])[SemanticMapLayer.STOP_LINE]
    rec["stop_lines"] = [to_local(o, (0.0, 0.0, 0.0), np.asarray(s.polygon.exterior.coords)[:, :2])
                         for s in objs if int(getattr(s.stop_line_type, "value", s.stop_line_type)) == 1]
    return rec


# ── 各个槽 ────────────────────────────────────────────────────────────
def decision(gt):
    v0, speed, path = plan_from_track([(0.0, 0.0)] + [tuple(p[:2]) for p in gt])
    verb = ("remain stopped" if v0 < VAL_STOP else "come to a stop") if speed == "STOP" else SPEED_VERB[speed]
    if path != "STRAIGHT":
        verb += f" and bear {path.lower()}"
    return verb, speed, path


def lead(r):
    B, nm = r["boxes"], r["names"]
    yaw = np.abs(np.arctan2(np.sin(B[:, 6]), np.cos(B[:, 6])))
    # 同向：车 |yaw| < 90°；行人 / 骑行者 < 60°（再斜就是在横穿，算让行，不算跟车）
    same_dir = np.where(nm == "vehicle", yaw < np.pi / 2, yaw < np.pi / 3)
    m = np.isin(nm, list(CLS)) & (B[:, 0] >= LEAD_X[0]) & (B[:, 0] < LEAD_X[1]) & (np.abs(B[:, 1]) < LEAD_Y) & same_dir
    idx = np.where(m)[0]
    if not len(idx):
        return None
    j = idx[np.argmin(B[idx, 0])]
    return j, CLS[nm[j]], int(B[j, 0]), float(np.hypot(*r["vel"][j]))


def lead_sentence(L):
    _, typ, x, v = L
    state = "stopped" if v < STOPPED else "moving slowly" if v < SLOW else f"moving at {int(v)} m/s"
    return f"The {typ} {x} meters ahead in my lane is {state}."


def vru(r):
    B, nm = r["boxes"], r["names"]
    m = np.isin(nm, ["pedestrian", "bicycle"]) & (B[:, 0] >= LEAD_X[0]) & (B[:, 0] < VRU_X) & (np.abs(B[:, 1]) < LEAD_Y)
    idx = np.where(m)[0]
    if not len(idx):
        return None
    n_ped, n_cyc = int((nm[idx] == "pedestrian").sum()), int((nm[idx] == "bicycle").sum())
    words = ([("a pedestrian" if n_ped == 1 else "pedestrians")] if n_ped else []) + \
            ([("a cyclist" if n_cyc == 1 else "cyclists")] if n_cyc else [])
    return set(idx.tolist()), " and ".join(words), int(B[idx, 0].min())


def ego_path(r):
    """自车 GT 路径 + 沿末端朝向直线延长到 PATH_LEN。
    朝向用 GT 末点自带的 heading，不用最后一段位移：快停下时最后一段只有几厘米，方向是噪声，路径带会歪出去。"""
    P = np.vstack([[0.0, 0.0], np.asarray(r["ego_fut"][:FUT])])
    P = P[np.concatenate([[True], np.linalg.norm(np.diff(P, axis=0), axis=1) > 0.05])]
    d = np.array([np.cos(r["ego_head"]), np.sin(r["ego_head"])])
    s_end = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) if len(P) >= 2 else 0.0
    if s_end < PATH_LEN:
        P = np.vstack([P, P[-1] + d * (PATH_LEN - s_end)])
    return P


def project(P, q):
    """点 q 到折线 P：(横向距离, 弧长)。"""
    best, s0 = (np.inf, 0.0), 0.0
    for a, b in zip(P[:-1], P[1:]):
        ab = b - a
        L2 = float(ab @ ab)
        t = 0.0 if L2 < 1e-9 else float(np.clip((q - a) @ ab / L2, 0.0, 1.0))
        dist = float(np.linalg.norm(q - (a + t * ab)))
        if dist < best[0]:
            best = (dist, s0 + t * np.sqrt(L2))
        s0 += np.sqrt(L2)
    return best


def cut_in(r, P, s_lead):
    """t=0 在路径带外、3 s 内真的开进自车前方路径带的最近他车。
    进入点要在自车前面、前车后面（被前车挡着时，横穿前车前面的人不是自车停下的原因）。"""
    B, nm = r["boxes"], r["names"]
    ego_s = [project(P, np.asarray(p))[1] for p in r["ego_fut"][:CUT_T]]
    best = None
    for j in np.where(np.isin(nm, list(CLS)) & (B[:, 0] > 0) & (B[:, 0] < CUT_X))[0]:
        if np.hypot(*r["vel"][j]) < CUT_V or project(P, B[j, :2])[0] <= BAND_OUT:
            continue
        for k in range(1, CUT_T + 1):
            q = r["fut"].get(k, {}).get(r["tracks"][j])
            if q is None:
                continue
            d, s = project(P, np.asarray(q))
            if d < BAND_IN and ego_s[k - 1] < s < min(CUT_X, s_lead):
                if best is None or B[j, 0] < B[best, 0]:
                    best = j
                break
    if best is None:
        return None
    y = float(B[best, 1])
    side = "left" if y > LEAD_Y / 2 else "right" if y < -LEAD_Y / 2 else "center"
    return best, CLS[nm[best]], int(B[best, 0]), side


def stop_sign(r, P, s_lead):
    """停车线多边形压到自车路径上（不看中心点：斜的大停车线中心会偏出车道），而且在前车前面。"""
    from shapely.geometry import LineString, Polygon
    from shapely.ops import nearest_points
    line = LineString(P)
    for poly in r["stop_lines"]:
        pg = Polygon(poly)
        if not pg.is_valid or line.distance(pg) > BAND_IN:
            continue
        s = line.project(nearest_points(line, pg)[0])
        if 0.0 < s <= min(SIGN_S, s_lead):
            return True
    return False


def gap_big(r, L):
    if L is None or L[1] != "vehicle" or L[2] < GAP_X:
        return False
    f = [r["fut"].get(k, {}).get(r["tracks"][L[0]]) for k in range(1, CUT_T + 1)]
    if not all(f):
        return False
    v_end = np.hypot(*(np.asarray(f[5]) - np.asarray(f[4]))) * 2
    gap2 = f[GAP_DT - 1][0] - r["ego_fut"][GAP_DT - 1][0]
    return v_end - L[3] >= GAP_DV and gap2 > float(r["boxes"][L[0], 0])


def generate(r):
    """→ (cot 文本, 子模式)。"""
    verb, speed, path = decision(r["ego_fut"])
    P = ego_path(r)
    V, L = vru(r), lead(r)
    s_lead = project(P, r["boxes"][L[0], :2])[1] if L else np.inf
    C = cut_in(r, P, s_lead)

    reason, mode = None, "none"
    if speed in ("STOP", "DECELERATE"):
        if stop_sign(r, P, s_lead):
            reason, mode = "there is a stop sign ahead", "stop_sign"
        elif V and not L:
            reason, mode = "I need to yield to them", "yield_vru"
        elif C:
            reason, mode = f"the {C[1]} on my {C[3]} will move into my path", "cross_path"
        elif L:
            reason, mode = "I need to stay behind it", "stay_behind"
    elif gap_big(r, L):
        reason, mode = "the gap in front of me is opening up", "gap_big"

    parts = []
    if V:
        parts.append(f"I see {V[1]} {V[2]} meters ahead of me.")
    if mode == "cross_path":
        pass                                   # 切入已经写在理由里
    elif C and not (V and C[0] in V[0]):       # 切入的人已经在 VRU 句里，不再重复
        parts.append(f"A {C[1]} {C[2]} meters {SIDE_OBS[C[3]]} will move into my path.")
    elif L and not (V and L[0] in V[0]):
        parts.append(lead_sentence(L))
    because = f"Because {reason}, " if reason else ""
    parts.append(f"{because}I should {verb}: <PLAN>{speed},{path}</PLAN>")
    return " ".join(parts), mode


# ── 批量 ──────────────────────────────────────────────────────────────
_MAPS = {}


def run_log(args):
    log, targets = args          # targets: {token: gt_trajectory}
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    L = pickle.load(open(f"{LOG_DIR}/{log}.pkl", "rb"))
    loc = L[0]["map_location"]
    if loc not in _MAPS:
        _MAPS[loc] = get_maps_api(MAPS_ROOT, MAP_VERSION, loc)
    out = {}
    for i, fr in enumerate(L):
        gt = targets.get(fr["token"])
        if gt is None:
            continue
        if not window_ok(L, i):
            out[fr["token"]] = ("", "abstain")
            continue
        out[fr["token"]] = generate(frame_record(L, i, _MAPS[loc], np.asarray(gt, dtype=np.float64)))
    return log, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", required=True, help="样本 JSON 目录（navtrain_nocot 等），决定生成哪些 token")
    ap.add_argument("--out", required=True, help="输出 {token: cot}")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit_logs", type=int, default=0, help="只跑前 N 个 log（试跑用）")
    args = ap.parse_args()

    t0 = time.time()
    by_log = defaultdict(dict)
    for f in os.listdir(args.json_dir):
        d = json.load(open(os.path.join(args.json_dir, f)))
        log = d["front_camera_paths"][-1].split("/trainval/")[1].split("/")[0]
        by_log[log][d["token"]] = d["gt_trajectory"]
    logs = sorted(by_log)[:args.limit_logs or None]
    n_target = sum(len(by_log[l]) for l in logs)
    print(f"log {len(logs)} 个   目标 token {n_target}   workers {args.workers}   读 JSON {time.time() - t0:.0f}s")

    cot, modes = {}, Counter()
    with Pool(args.workers) as pool:
        for k, (log, out) in enumerate(pool.imap_unordered(run_log, [(l, by_log[l]) for l in logs]), 1):
            for tok, (text, mode) in out.items():
                cot[tok] = text
                modes[mode] += 1
            if k % 100 == 0 or k == len(logs):
                print(f"  [{k}/{len(logs)}] 已产出 {len(cot)}   {time.time() - t0:.0f}s", flush=True)

    json.dump(cot, open(args.out, "w"))
    missing = n_target - len(cot)
    print(f"\n写出 {args.out}: {len(cot)} 条   用时 {time.time() - t0:.0f}s")
    print(f"  弃权（窗口缺帧）{modes['abstain']}   log 里找不到的 token {missing}")
    n = sum(v for k, v in modes.items() if k != "abstain")
    print("  子模式 " + "   ".join(f"{k} {100 * v / n:.1f}%" for k, v in modes.most_common() if k != "abstain"))
    plan = Counter(t.split("<PLAN>")[1].split("</PLAN>")[0] for t in cot.values() if t)
    print("  <PLAN> " + "   ".join(f"{k} {v}" for k, v in plan.most_common(8)))


if __name__ == "__main__":
    main()
