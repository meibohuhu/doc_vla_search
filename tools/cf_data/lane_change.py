"""
D3 合成换道（docs/1026/cf_sft_autovla_plan.md §3 D3）。

画面不动，只把这一帧未来 5 s 的轨迹换成"换到相邻同向车道"：

    路径  当前车道中心线 ──smoothstep，t = 0 开始──▶ 目标车道中心线
    快慢  专家每 0.5 s 走的距离（纵向不改）

然后逐关检查，任何一关不过就记下原因、丢掉:
    车道     自车在 lane 里（不在路口，也不在中心线起点之前）、有相邻同向车道、两条中心线间距合理
    几何     设计轨迹每 0.5 s 走的距离和专家对得上；横向加速度 ≤ A_LAT_MAX
    路线     目标车道前方第一个出口在路线上（不换进转弯专用道）
    路口     5 s 内的轨迹点都不进路口
    在动     5 s 总位移 ≥ MIN_DISP。否则就是"收到换道指令、车却不动"的样本
    PDM      NC / DAC / DDC 全为 1（PDM 只查前 4 s）
    跟踪     PDM 自带的 LQR + 自行车模型跟一遍，和设计轨迹偏差 ≤ TRACK_ERR_MAX
    间距     t = 0 目标车道里紧挨着没车；并入后和目标车道前后车的间距 ≥ max(2 m, 0.5 s × 车速)
    红灯     不往路线上亮红灯的 lane connector 里开（红绿灯从 navsim_logs 取，metric cache 里没有）
    tokenizer 编码成 10 个 action token 再解码，只记录误差，先看分布再定阈值

🔴 其他车按日志回放，不会对自车做出反应 —— 和 simlingo dreamer 一样。

用法:
    # 原型：随机 300 帧，出示例图
    python tools/cf_data/lane_change.py --n_tokens 300 --out /scratch/mh2803/vla/nuplan/cf_navtrain_v1/proto_lc --viz 12
    # 全量（val 换成 --json_dir /scratch/mh2803/vla/nuplan/navtrain_nocot_val）
    python tools/cf_data/lane_change.py --workers 48 --out /scratch/mh2803/vla/nuplan/cf_navtrain_v1/lane_change/navtrain
输出 candidates.jsonl（全部候选；失败的只留原因和标量）和 ok.jsonl（通过的，带轨迹和 token id）。
"""

import argparse
import json
import lzma
import math
import os
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from shapely.geometry import LineString, Point, Polygon

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.autovla_agent import TokenProcessor
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
from models.utils.feasible import FeasibleSet, _headings
from models.utils.rl_rewards import plan_from_track
from models.utils.teacher import EGO_L, EGO_W, RAC

NUPLAN = "/scratch/mh2803/vla/nuplan"
MAPS_ROOT = f"{NUPLAN}/maps"
MAP_VERSION = "nuplan-maps-v1.0"
JSON_DIR = f"{NUPLAN}/navtrain_nocot"
LOG_DIR = f"{NUPLAN}/navsim_logs/trainval"
METRIC_CACHE = f"{NUPLAN}/navtrain_metric_cache"
CODEBOOK = "codebook_cache/agent_vocab.pkl"

DT, N = 0.5, 10                  # 10 个 waypoint，每个 0.5 s
T_RANGE = (3.0, 4.5)             # smoothstep 总时长。下限:NavSafe I-3 要 ~3 s 横移 ≥ 3.8 m;
                                 # 上限:navtrain 专家换道中位 4.5 s（10→90% 用 2.76 s）
LANE_SPACING = (2.5, 5.0)        # 两条车道中心线的横向间距，m
MIN_DISP = 5.0                   # 5 s 总位移下限，m
TRACK_ERR_MAX = 1.0              # LQR 跟踪后偏离设计路径的最大【横向】距离，m
                                 # 🔴 不能用总偏差:PDM 的 LQR 纵向本来就跟不紧，专家 GT 自己的总偏差
                                 #    中位就有 0.53 m（减速时领先，4 s 处约 1 m）。横向偏差在专家 GT 上
                                 #    p50 0.17 / p90 0.80 / p95 0.91 m。换道时 LQR 普遍滞后 ~0.75 m，
                                 #    取 1.0（≈ 专家 p95）;PDM 的 NC / DAC 本来就是在跟踪后的轨迹上判的
MAX_SLOPE = math.tan(math.radians(20))  # 横移 / 前进 的上限。低速时 3 s 横移 3.4 m 只前进几米，等于横着挪
                                        # （原型先用 25°，5 m/s 减速中并入、斜率 25° 的样本看图后不要，收到 20°）
STRAIGHT_THROUGH = math.radians(20)     # 穿过路口时两条车道在 5 s 内的朝向变化上限
GAP_MIN, HEADWAY = 2.0, 0.5      # 间距下限 max(2 m, 0.5 s × 车速)
IN_LANE_HALF = 1.75              # 中心离目标车道中心线 < 1.75 m 才算"目标车道里的车"
HEAD_MISMATCH = math.radians(30) # 自车朝向和所在车道朝向差太多 → 不在这条车道上（或 metric cache 朝向翻转）
ROUTE_HORIZON = 100.0            # 目标车道在这段距离内的第一个出口必须有一条在路线上，m
SUB = 5                          # 0.1 s 一步，和日志红绿灯 / observation 对齐
RED_ENTER = 0.5                  # 和红灯 connector 的重叠面积比 t = 0 多出这么多（m²）算闯
RED_INSIDE = 1.0                 # t = 0 时重叠已经超过这么多（m²）= 已过停止线，不算
A_LAT_MAX = 3.0                  # m/s²。专家 GT 每条轨迹的最大横向加速度 p99 2.56 / p99.9 3.31（navtrain 3000 条）
SIDES = {"LEFT": 0, "RIGHT": 1}  # adjacent_edges 的下标


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


# ── 几何小工具（全局坐标）────────────────────────────────────────────────

def frame_at(ls: LineString, s: float) -> Tuple[np.ndarray, np.ndarray]:
    """折线上弧长 s 处的点和左法向量。"""
    L = ls.length
    a = max(0.0, min(s, L - 0.5))
    p0 = np.asarray(ls.interpolate(a).coords[0])
    p1 = np.asarray(ls.interpolate(a + 0.5).coords[0])
    t = (p1 - p0) / (np.linalg.norm(p1 - p0) + 1e-9)
    return np.asarray(ls.interpolate(s).coords[0]), np.array([-t[1], t[0]])


def project(ls: LineString, p) -> Tuple[float, float]:
    """点 → (弧长, 带符号横向距离，左正)。"""
    s = ls.project(Point(*p))
    base, n = frame_at(ls, s)
    return s, float(np.dot(np.asarray(p) - base, n))


def to_local(xy, ex, ey, eh):
    c, s = math.cos(eh), math.sin(eh)
    d = np.asarray(xy, dtype=np.float64) - [ex, ey]
    return np.stack([d[..., 0] * c + d[..., 1] * s, -d[..., 0] * s + d[..., 1] * c], axis=-1)


def to_global(xy, ex, ey, eh):
    c, s = math.cos(eh), math.sin(eh)
    xy = np.asarray(xy, dtype=np.float64)
    return np.stack([ex + xy[..., 0] * c - xy[..., 1] * s, ey + xy[..., 0] * s + xy[..., 1] * c], axis=-1)


def lateral_err(ref: np.ndarray, sim: np.ndarray) -> float:
    """跟踪后的点到参考路径的最大垂直距离。

    路径沿末端方向延长 20 m 再量 —— 否则纵向领先、跑出路径末端的点会被算成到端点的距离，
    纵向滞后就混进了横向误差。
    """
    d = ref[-1] - ref[-2]
    path = LineString(np.vstack([ref, ref[-1] + 20.0 * d / (np.linalg.norm(d) + 1e-9)]))
    return max(abs(project(path, p)[1]) for p in sim)


def upsample(traj) -> np.ndarray:
    """0.5 s 一个点的轨迹（不含 t = 0）→ 0.1 s 一步、含 t = 0 的局部坐标点（51 个）。"""
    p = np.vstack([[0.0, 0.0], np.asarray(traj, dtype=np.float64)[:, :2]])
    t5, t1 = np.arange(N + 1) * DT, np.arange(N * SUB + 1) * DT / SUB
    return np.stack([np.interp(t1, t5, p[:, 0]), np.interp(t1, t5, p[:, 1])], axis=1)


def max_lat_acc(traj) -> float:
    """0.5 s 一个点的轨迹 → 最大横向加速度（相邻两段的朝向变化率 × 车速）。一段 < 0.5 m 的不算（低速朝向噪声大）。"""
    p = np.vstack([[0.0, 0.0], np.asarray(traj)[:, :2]])
    d = np.diff(p, axis=0)
    step = np.linalg.norm(d, axis=1)
    h = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    out = [abs(h[i] - h[i - 1]) / DT * 0.5 * (step[i] + step[i - 1]) / DT
           for i in range(1, len(d)) if step[i] > 0.5 and step[i - 1] > 0.5]
    return max(out) if out else 0.0


def ego_box(x, y, h) -> Polygon:
    """后轴位姿 → 车身四角（Pacifica 尺寸，同 teacher.py）。"""
    cx, cy = x + RAC * math.cos(h), y + RAC * math.sin(h)
    c, s = math.cos(h), math.sin(h)
    hl, hw = EGO_L / 2, EGO_W / 2
    return Polygon([(cx + c * dx - s * dy, cy + s * dx + c * dy)
                    for dx, dy in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))])


MOVERS = ("VEHICLE", "BICYCLE")


def clearance(mc, traj, ex, ey, eh, types=MOVERS) -> Tuple[float, float]:
    """每 0.5 s 自车框到最近的车 / 自行车框的距离（满 5 s，PDM 只查 4 s）→ (最小值, 发生时刻 s)。
    metric cache 的 observation 是 0.1 s 一帧。周围没有时是 (inf, 0.5)。
    只算 types：observation 里还有路边的杆子、花盆、锥桶（GENERIC_OBJECT 等），离它们近由 DAC 管，
    算进来会把正常的换道当成"贴车"（用户 10-27 看 a87edb35ddf85c5f、2332cf379a5f5bf9 指出）。"""
    traj = np.asarray(traj, dtype=np.float64)
    g = to_global(traj[:, :2], ex, ey, eh)
    uo = mc.observation.unique_objects
    keep = {t for t, o in uo.items() if o.tracked_object_type.name in types}
    d = []
    for k in range(len(traj)):
        box, obs = ego_box(g[k][0], g[k][1], eh + traj[k][2]), mc.observation[5 * (k + 1)]
        d.append(min((box.distance(geom) for t, geom in zip(obs.tokens, obs._geometries) if t in keep), default=np.inf))
    k = int(np.argmin(d))
    return float(d[k]), 0.5 * (k + 1)


def heading_change(ls: LineString, s_from: float, s_to: float) -> float:
    """折线在 [s_from, s_to] 上的朝向变化范围（rad）。"""
    hs = []
    for s in np.linspace(s_from, s_to, 12):
        _, n = frame_at(ls, s)
        hs.append(math.atan2(-n[0], n[1]))          # 左法向量转回切向
    hs = np.unwrap(hs)
    return float(hs.max() - hs.min())


def chain(edge, need: float) -> LineString:
    """从 edge 起沿后继接 baseline，直到长度 ≥ need。岔路取方向变化最小的那条。"""
    pts = [(p.x, p.y) for p in edge.baseline_path.discrete_path]
    length = edge.baseline_path.length
    for _ in range(20):
        if length >= need or not edge.outgoing_edges:
            break
        end_h = math.atan2(pts[-1][1] - pts[-2][1], pts[-1][0] - pts[-2][0])

        def turn(e):
            q = e.baseline_path.discrete_path
            h = math.atan2(q[1].y - q[0].y, q[1].x - q[0].x)
            return abs((h - end_h + math.pi) % (2 * math.pi) - math.pi)

        edge = min(edge.outgoing_edges, key=turn)
        pts += [(p.x, p.y) for p in edge.baseline_path.discrete_path][1:]
        length += edge.baseline_path.length
    return LineString(pts)


def extend_back(ls: LineString, edge, p) -> LineString:
    """车道多边形包住了自车，但中心线可能从自车前方才开始（刚出路口）。这时接上离自车最近的
    前驱（多半是路口里的 lane connector），否则投影被夹在起点上，设计轨迹第一步就会跳出去
    （val fe06df4a8eb45023：中心线从前方 12.8 m 开始，0.5 s 走了 17.7 m）。"""
    if ls.project(Point(*p)) >= 0.1 or not edge.incoming_edges:
        return ls
    prev = min(edge.incoming_edges, key=lambda e: e.baseline_path.linestring.distance(Point(*p)))
    return LineString([(q.x, q.y) for q in prev.baseline_path.discrete_path][:-1] + list(ls.coords))


def first_exits(edge, s_from: float, horizon: float) -> list:
    """沿唯一的 lane 后继往前走，返回 horizon 内第一个出口（分岔，或进路口的 lane connector）。没有就返回 []。"""
    dist = edge.baseline_path.length - s_from
    for _ in range(20):
        outs = edge.outgoing_edges
        if dist > horizon or not outs:
            return []
        if len(outs) > 1 or type(outs[0]) is not type(edge):
            return list(outs)
        edge = outs[0]
        dist += edge.baseline_path.length
    return []


# ── 主体 ────────────────────────────────────────────────────────────────

class LaneChangeSynth:
    def __init__(self, metric_cache=METRIC_CACHE, codebook=CODEBOOK, json_dir=JSON_DIR):
        self.fs = FeasibleSet(Path(metric_cache))     # metric cache 读取 + PDM simulator / scorer
        self.tp = TokenProcessor(codebook, DictConfig({"num_k": 1, "temp": 1.0}))
        self.json_dir = json_dir
        self._maps: Dict[str, object] = {}
        self._log_map: Dict[str, str] = {}
        # log → ({token: 帧号}, 每帧的 [(lane_connector_id, is_red)])。日志是 2 Hz
        self._log_tl: Dict[str, Tuple[Dict[str, int], list]] = {}

    def map_api(self, log: str):
        if log not in self._log_map:
            frames = pickle.load(open(f"{LOG_DIR}/{log}.pkl", "rb"))
            self._log_map[log] = frames[0]["map_location"]
            self._log_tl[log] = ({f["token"]: i for i, f in enumerate(frames)},
                                 [f["traffic_lights"] for f in frames])
        name = self._log_map[log]
        if name not in self._maps:
            self._maps[name] = get_maps_api(MAPS_ROOT, MAP_VERSION, name)
        return self._maps[name]

    def tokenize(self, traj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """→ (10 个 token id, 解码后的 10×3 轨迹)。和训练时 TrajectoryTargetBuilder 同一个 TokenProcessor。"""
        out = self.tp(torch.tensor(traj, dtype=torch.float32))
        dec = torch.cat([out["gt_pos"][0], out["gt_heading"][0][:, None]], dim=-1).numpy()
        return out["gt_idx"][0].numpy(), dec

    def boxes(self, xy_local: np.ndarray, ex, ey, eh) -> List[Polygon]:
        """0.1 s 一步的局部点（含 t = 0）→ 自车多边形（全局）。"""
        g = to_global(xy_local, ex, ey, eh)
        hs = np.concatenate([[eh], _headings(xy_local[1:], (0.0, 0.0)) + eh])
        return [ego_box(x, y, h) for (x, y), h in zip(g, hs)]

    def red_light(self, mc, m, log: str, token: str, boxes) -> Optional[dict]:
        """boxes：t = 0, 0.1, …, 5.0 s 的自车多边形（全局）。闯红灯返回 {t, connector}，否则 None。

        🔴 不能用 metric cache 的 observation：navsim v1 建 metric cache 时没把红绿灯传进
           PDMObservation（metric_cache_processor.py 只调了 update_detections_tracks），
           里面一个红灯都没有，PDM 打分也从来不管红灯。这里从 navsim_logs 取每帧（2 Hz）的
           红绿灯，状态在 [t_k, t_k + 0.5 s) 内保持；只看路线上的 connector（同 PDM 的设计，
           否则横穿路口时会撞上横向车流那些红灯 connector）。往红灯 connector 里开得比 t = 0
           时更深就算闯；t = 0 已经过了停止线的不算。
        """
        idx, tls = self._log_tl[log]
        i0 = idx[token]
        route = {str(x) for x in mc.route_lane_ids}
        polys: Dict[str, object] = {}
        base: Dict[str, float] = {}
        for step, box in enumerate(boxes):
            k = i0 + step // SUB
            if k >= len(tls):
                break
            for lc, is_red in tls[k]:
                lc = str(lc)
                if not is_red or lc not in route:
                    continue
                if lc not in polys:
                    obj = m.get_map_object(lc, SemanticMapLayer.LANE_CONNECTOR)
                    polys[lc] = obj.polygon if obj is not None else None
                if polys[lc] is None:
                    continue
                if lc not in base:
                    base[lc] = boxes[0].intersection(polys[lc]).area
                if base[lc] > RED_INSIDE:
                    continue
                if box.intersection(polys[lc]).area > base[lc] + RED_ENTER:
                    return {"t": round(0.1 * step, 1), "connector": lc}
        return None

    def ego_lane(self, m, ex, ey, eh):
        """自车所在的 lane（不含 lane connector）。多条重叠时取朝向最接近的。"""
        lanes = m.get_all_map_objects(Point2D(ex, ey), SemanticMapLayer.LANE)
        best, best_dh = None, HEAD_MISMATCH
        for ln in lanes:
            ls = ln.baseline_path.linestring
            s = ls.project(Point(ex, ey))
            p0, p1 = ls.interpolate(max(0.0, s - 0.5)), ls.interpolate(min(ls.length, s + 0.5))
            dh = abs((math.atan2(p1.y - p0.y, p1.x - p0.x) - eh + math.pi) % (2 * math.pi) - math.pi)
            if dh < best_dh:
                best, best_dh = ln, dh
        return best

    def run(self, token: str, rng: random.Random) -> List[dict]:
        """一帧 → 左右两侧各一条候选（不论成败，都带 status / reason）。"""
        d = json.load(open(f"{self.json_dir}/{token}.json"))
        log = d["front_camera_paths"][-1].split("/trainval/")[1].split("/")[0]
        mc = self.fs.metric_cache(token)
        e = mc.ego_state.rear_axle
        ex, ey, eh = float(e.x), float(e.y), float(e.heading)
        gt = np.asarray(d["gt_trajectory"], dtype=np.float64)
        v0 = float(np.hypot(*d["velocity"]))
        base = {"token": token, "cmd": d["instruction"], "v0": v0, "log": log}

        m = self.map_api(log)
        lane = self.ego_lane(m, ex, ey, eh)
        if lane is None:
            return [dict(base, side=s, status="skip", reason="not_in_lane") for s in SIDES]

        # 专家每个时刻走过的弧长（纵向照抄专家）
        xy = np.vstack([[0.0, 0.0], gt[:, :2]])
        s_exp = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
        if s_exp[-1] < MIN_DISP:
            return [dict(base, side=s, status="skip", reason="not_moving") for s in SIDES]
        t = np.arange(N + 1) * DT
        v_exp = np.gradient(s_exp, t)

        need = s_exp[-1] + 30.0
        C = extend_back(chain(lane, lane.baseline_path.linestring.project(Point(ex, ey)) + need), lane, (ex, ey))
        s0, o0 = project(C, (ex, ey))
        # 往回接了还在起点之前（没有前驱），或者离中心线太远（骑线、专家自己正在换道）
        if s0 < 0.1 or abs(o0) > IN_LANE_HALF:
            return [dict(base, side=s, status="skip", reason="not_in_lane") for s in SIDES]
        route = {str(x) for x in mc.route_lane_ids}
        inters = [o.polygon for o in m.get_proximal_map_objects(
            Point2D(ex, ey), s_exp[-1] + 30.0, [SemanticMapLayer.INTERSECTION])[SemanticMapLayer.INTERSECTION]]

        out = []
        for side, k in SIDES.items():
            rec = dict(base, side=side, status="fail", reason="")
            tgt = lane.adjacent_edges[k]
            if tgt is None:
                out.append(dict(rec, status="skip", reason="no_adjacent")); continue
            Tc = extend_back(chain(tgt, tgt.baseline_path.linestring.project(Point(ex, ey)) + need), tgt, (ex, ey))
            if C.length < s0 + s_exp[-1] + 1.0 or Tc.length < Tc.project(Point(ex, ey)) + s_exp[-1] + 1.0:
                out.append(dict(rec, reason="path_short")); continue

            # 两条中心线在自车处的横向间距（左正）
            spacing = project(C, Tc.interpolate(Tc.project(Point(ex, ey))).coords[0])[1]
            rec["spacing"] = spacing
            if not (LANE_SPACING[0] <= abs(spacing) <= LANE_SPACING[1]) or (spacing > 0) != (side == "LEFT"):
                out.append(dict(rec, reason="spacing")); continue

            # ── 设计轨迹:沿当前车道走专家的弧长，横向按 smoothstep 从当前偏移移到目标车道 ──
            T = rng.uniform(*T_RANGE)
            w = smoothstep(t / T)
            pts_g, lat, dlts = [], [], []
            for i in range(N + 1):
                b, n = frame_at(C, s0 + s_exp[i])
                q = np.asarray(Tc.interpolate(Tc.project(Point(*b))).coords[0])
                dlt = float(np.dot(q - b, n))
                dlts.append(dlt)
                lat.append((1 - w[i]) * o0 + w[i] * dlt)
                pts_g.append(b + lat[-1] * n)
            pts_g = np.asarray(pts_g)
            # 目标车道要在 5 s 内一直和当前车道平行。右侧车道常常是右转专用道，
            # 沿后继接下去会拐进匝道 —— 那就成了右转，不是换道
            if not all(LANE_SPACING[0] <= abs(x) <= LANE_SPACING[1] and (x > 0) == (side == "LEFT")
                       for x in dlts):
                out.append(dict(rec, reason="target_diverges")); continue
            P = to_local(pts_g, ex, ey, eh)
            traj = np.concatenate([P[1:], _headings(P[1:], (0.0, 0.0))[:, None]], axis=1)
            rec.update(T=T, traj=traj.round(4).tolist())

            # 几何兜底：每 0.5 s 走的距离要和专家对得上（横移和弯道上的偏移会让它略长，给 25% 余量）
            step = np.linalg.norm(np.diff(P, axis=0), axis=1)
            if np.any(np.abs(step - np.diff(s_exp)) > np.maximum(0.5, 0.25 * np.diff(s_exp))):
                out.append(dict(rec, reason="geometry")); continue

            # 目标车道前方第一个出口要有一条在路线上。否则是换进了转弯专用道（路线要直行却换进
            # 左转道），或者路线要转弯却换进了直行道。PDM 不罚这种，画面里常有 "ONLY" 地标和实线，
            # 而 nuPlan 地图分不出实线虚线（车道之间的 boundary_type 都是 0）
            exits = first_exits(tgt, tgt.baseline_path.linestring.project(Point(ex, ey)), ROUTE_HORIZON)
            if exits and not any(str(o.id) in route for o in exits):
                out.append(dict(rec, reason="off_route")); continue

            ds = np.diff(s_exp)
            slope = np.abs(np.diff(lat))[ds > 0.1] / ds[ds > 0.1]
            rec["max_slope_deg"] = float(math.degrees(math.atan(slope.max()))) if len(slope) else 90.0
            if len(slope) == 0 or slope.max() > MAX_SLOPE:
                out.append(dict(rec, reason="too_sharp")); continue
            rec["a_lat"] = max_lat_acc(traj)
            if rec["a_lat"] > A_LAT_MAX:
                out.append(dict(rec, reason="too_fast_for_curve")); continue

            # 路口:keep forward 且两条车道在这 5 s 里都基本是直的，允许直行穿过（路边丁字口、小路口）；
            # 否则一律不在路口附近换道
            if any(poly.contains(Point(*p)) for p in pts_g for poly in inters):
                straight = (heading_change(C, s0, s0 + s_exp[-1]) < STRAIGHT_THROUGH and
                            heading_change(Tc, Tc.project(Point(ex, ey)),
                                           Tc.project(Point(ex, ey)) + s_exp[-1]) < STRAIGHT_THROUGH)
                if not (straight and d["instruction"] == "keep forward"):
                    out.append(dict(rec, reason="junction")); continue
                rec["through_junction"] = True

            # ── PDM 子分 + LQR 跟踪 ──
            tj = Trajectory(traj.astype(np.float32), TrajectorySampling(num_poses=N, interval_length=DT))
            met = self.fs.score(token, tj)
            if met is None:
                out.append(dict(rec, reason="pdm_error")); continue
            rec["pdm"] = met
            states = get_trajectory_as_array(transform_trajectory(tj, mc.ego_state),
                                             self.fs.future_sampling, mc.ego_state.time_point)
            sim = self.fs.simulator.simulate_proposals(states[None], mc.ego_state)[0]
            # 状态数组前两维是 x, y（pdm_enums.StateIndex）
            terr = lateral_err(states[:, :2], sim[:, :2])
            rec["track_err"] = terr
            rec["track_err_total"] = float(np.linalg.norm(sim[:, :2] - states[:, :2], axis=1).max())
            rec["tracked"] = to_local(sim[::5, :2], ex, ey, eh)[1:].round(3).tolist()

            # ── tokenizer 往返（只记录）──
            idx, dec = self.tokenize(traj)
            rec["token_ids"] = idx.tolist()
            rec["decoded"] = dec.round(4).tolist()
            rec["token_err"] = float(np.linalg.norm(dec[:, :2] - traj[:, :2], axis=1).max())
            rec["token_err_lat_end"] = float(abs(dec[-1, 1] - traj[-1, 1]))

            pft = plan_from_track([(0.0, 0.0)] + [tuple(p) for p in traj[:, :2]])
            rec["speed"] = pft[1] if pft else None

            if met["nc"] < 1 or met["dac"] < 1 or met["ddc"] < 1:
                out.append(dict(rec, reason="pdm_" + "_".join(
                    k for k in ("nc", "dac", "ddc") if met[k] < 1))); continue
            if terr > TRACK_ERR_MAX:
                out.append(dict(rec, reason="tracking")); continue
            gap = self.gap_check(mc, pts_g, t, w, v_exp, Tc)
            rec["gap"] = gap
            if gap is not None:
                out.append(dict(rec, reason=gap["reason"])); continue
            # 纵向照抄专家，专家在灯刚变时进路口的情况会原样带过来（全量 4942 条里 43 条，40 条专家自己也算闯）
            rec["red"] = self.red_light(mc, m, log, token, self.boxes(upsample(traj), ex, ey, eh))
            if rec["red"] is not None:
                out.append(dict(rec, reason="red_light")); continue
            out.append(dict(rec, status="ok"))
        return out

    def gap_check(self, mc, pts_g, t, w, v_exp, Tc) -> Optional[dict]:
        """目标车道里的车（日志回放）和自车的前后间距。通过返回 None。"""
        obs = mc.observation
        heads = np.concatenate([[mc.ego_state.rear_axle.heading],
                                np.arctan2(np.diff(pts_g[:, 1]), np.diff(pts_g[:, 0]))])
        n_obs = min(len(obs._global_to_local_idcs), 51)
        for i in range(n_obs):
            ti = 0.1 * i
            if i > 0 and float(np.interp(ti, t, w)) < 0.1:
                continue                                       # 还没开始往目标车道去
            x, y = np.interp(ti, t, pts_g[:, 0]), np.interp(ti, t, pts_g[:, 1])
            h = heads[min(math.ceil(ti / DT - 1e-9), N)]       # (t_{k-1}, t_k] 段用第 k 段的朝向
            box = ego_box(x, y, h)
            e_s = [Tc.project(Point(*c)) for c in box.exterior.coords[:4]]
            v_ego = float(np.interp(ti, t, v_exp))
            occ = obs[i]
            for tok, geom in zip(occ.tokens, occ._geometries):
                if obs.red_light_token in tok:
                    continue
                if abs(project(Tc, (geom.centroid.x, geom.centroid.y))[1]) > IN_LANE_HALF:
                    continue
                a_s = [Tc.project(Point(*c)) for c in geom.exterior.coords]
                ahead = min(a_s) >= max(e_s)
                gap = (min(a_s) - max(e_s)) if ahead else (min(e_s) - max(a_s))
                if i == 0:
                    if gap < GAP_MIN:
                        return {"reason": "alongside_t0", "t": 0.0, "gap": float(gap)}
                    continue
                obj = obs.unique_objects.get(tok)
                v_other = float(obj.velocity.magnitude()) if obj is not None and hasattr(obj, "velocity") else v_ego
                need = max(GAP_MIN, HEADWAY * (v_ego if ahead else v_other))
                if gap < need:
                    return {"reason": "gap", "t": round(ti, 1), "gap": float(gap), "need": need}
        return None


# ── 画图 ────────────────────────────────────────────────────────────────

def render(rec: dict, synth: LaneChangeSynth, out_png: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.image import imread

    d = json.load(open(f"{synth.json_dir}/{rec['token']}.json"))
    mc = synth.fs.metric_cache(rec["token"])
    e = mc.ego_state.rear_axle
    ex, ey, eh = float(e.x), float(e.y), float(e.heading)
    gt = np.asarray(d["gt_trajectory"])

    fig = plt.figure(figsize=(14, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.7, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(imread(d["front_camera_paths"][-1]))
    ax.set_title("front camera, t = 0", fontsize=10)
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    dm = mc.drivable_area_map
    for geom, typ in zip(dm._geometries, dm._map_types):
        if typ != SemanticMapLayer.LANE and typ != SemanticMapLayer.LANE_CONNECTOR:
            continue
        loc = to_local(np.asarray(geom.exterior.coords), ex, ey, eh)
        ax.plot(-loc[:, 1], loc[:, 0], color="0.6" if typ == SemanticMapLayer.LANE else "0.85", lw=0.7)
    for i, alpha in ((0, 0.9), (25, 0.35)):                  # 日志车辆:t = 0 实色，t = 2.5 s 淡色
        for tok, geom in zip(mc.observation[i].tokens, mc.observation[i]._geometries):
            if mc.observation.red_light_token in tok:
                continue
            loc = to_local(np.asarray(geom.exterior.coords), ex, ey, eh)
            ax.fill(-loc[:, 1], loc[:, 0], color="tab:purple", alpha=alpha * 0.5, lw=0)
    ax.plot(-gt[:, 1], gt[:, 0], "o-", color="0.45", ms=3, lw=1, label="expert GT")
    if "traj" in rec:
        tr = np.asarray(rec["traj"])
        ax.plot(-tr[:, 1], tr[:, 0], "o-", color="tab:red", ms=3.5, lw=1.5, label=f"lane change (T={rec['T']:.1f}s)")
    if "tracked" in rec:
        tk = np.asarray(rec["tracked"])
        ax.plot(-tk[:, 1], tk[:, 0], "--", color="tab:orange", lw=1.2, label="LQR tracked (4 s)")
    if "decoded" in rec:
        dc = np.asarray(rec["decoded"])
        ax.plot(-dc[:, 1], dc[:, 0], "x", color="tab:green", ms=6, label="token decoded")
    ax.plot([0], [0], "s", color="k", ms=6)
    ax.set_xlim(-15, 15)
    ax.set_ylim(-12, max(50, gt[-1, 0] + 8))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.legend(loc="lower right", fontsize=7)

    pdm = rec.get("pdm", {})
    info = (f"{rec['status'].upper()} {rec['reason']}  |  side {rec['side']}  v0 {rec['v0']:.1f} m/s  "
            f"cmd '{rec['cmd']}'  speed word {rec.get('speed')}\n"
            f"NC {pdm.get('nc', '-')}  DAC {pdm.get('dac', '-')}  DDC {pdm.get('ddc', '-')}  "
            f"track err {rec.get('track_err', float('nan')):.2f} m  "
            f"token err {rec.get('token_err', float('nan')):.2f} m  gap {rec.get('gap')}")
    fig.suptitle(f"token {rec['token']}\n{info}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=90)
    plt.close(fig)


_SYNTH: Optional[LaneChangeSynth] = None
_HEAVY = ("traj", "tracked", "decoded", "token_ids")    # 失败的候选不存这些，全量时文件小一个量级


def _init_worker(json_dir: str):
    global _SYNTH
    torch.set_num_threads(1)
    _SYNTH = LaneChangeSynth(json_dir=json_dir)


def _run_one(args) -> List[dict]:
    tok, seed = args
    try:
        # 每个 token 自己的随机数（T 的抽样），结果和进程数、调度顺序无关
        recs = _SYNTH.run(tok, random.Random(f"{seed}:{tok}"))
    except Exception as ex:  # noqa: BLE001 —— 记下来，不中断
        recs = [{"token": tok, "status": "error", "reason": repr(ex)[:200]}]
    _SYNTH.fs._cache.clear()
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", default=JSON_DIR, help="navtrain_nocot 或 navtrain_nocot_val")
    ap.add_argument("--n_tokens", type=int, default=0, help="随机抽多少帧；0 = 全部")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--viz", type=int, default=0, help="最多画几张 ok 的；失败的另画同样多张")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tokens = sorted(f[:-5] for f in os.listdir(args.json_dir))
    if args.n_tokens:
        tokens = random.Random(args.seed).sample(tokens, args.n_tokens)
    jobs = [(tok, args.seed) for tok in tokens]

    recs = []
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers, initializer=_init_worker, initargs=(args.json_dir,)) as pool:
            for i, rs in enumerate(pool.imap(_run_one, jobs, chunksize=16), 1):
                recs += rs
                if i % 5000 == 0:
                    print(f"  {i}/{len(jobs)}", flush=True)
    else:
        _init_worker(args.json_dir)
        for job in jobs:
            recs += _run_one(job)

    ok = [r for r in recs if r["status"] == "ok"]
    with open(f"{args.out}/candidates.jsonl", "w") as f:
        for r in recs:
            if r["status"] != "ok":
                r = {k: v for k, v in r.items() if k not in _HEAVY}
            f.write(json.dumps(r) + "\n")
    with open(f"{args.out}/ok.jsonl", "w") as f:
        for r in ok:
            f.write(json.dumps(r) + "\n")

    from collections import Counter
    print(f"{len(tokens)} tokens -> {len(recs)} candidates")
    print("status/reason:", Counter((r["status"], r["reason"].split("(")[0]) for r in recs).most_common())
    if ok:
        te = np.array([r["token_err"] for r in ok])
        print("ok token_err max-pos p50/p90/max:", np.percentile(te, [50, 90, 100]).round(2))
        print("ok track_err p50/p90/max:", np.percentile([r["track_err"] for r in ok], [50, 90, 100]).round(2))
        print("ok speed words:", Counter(r["speed"] for r in ok))

    if args.viz:
        synth = _SYNTH or LaneChangeSynth(json_dir=args.json_dir)
        fails = [r for r in recs if r["status"] == "fail" and "traj" in r]
        for j, r in enumerate(ok[:args.viz] + fails[:args.viz]):
            render(r, synth, f"{args.out}/{r['status']}_{j:02d}_{r['side'].lower()}_{r['token']}.png")


if __name__ == "__main__":
    main()
