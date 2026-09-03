"""
反事实速度剖面 → 可行集 D（flow §6.2）。

teacher 的【知识】那一半:给定学生这一帧的**走法**，沿它自己的路径重新配速，
逐条送 PDMScorer，看哪几档在物理和规则上说得通。

    D ⊆ {HARD_BRAKE, BRAKE, KEEP, ACCEL}

⚠️ 这里只回答"哪几档可行"，**不**回答"该选哪一档"。判决（触发器 + 快慢侧）是另一半。

────────────────────────────────────────────────────────────────────────────
🔴 剖面常数是**在 NAVSIM 上重标的**，不是 CARLA 那套。

  simlingo/CARLA 用的是 a_brake=3.0、hard=8.0。在 NAVSIM 上实测（navtest 4000 帧、
  36000 个逐步加速度）：

      |a| p50=0.45  p90=1.29  p95=1.50  p99=1.93  p99.9=2.44   m/s²
      P(|a| ≥ 2.0) = 0.68%     P(|a| ≥ 3.0) = 0.04%     P(|a| ≥ 8.0) = 0.03%

  照搬 3.0/8.0 会造出一批**物理上根本不出现**的反事实 —— 可行集要么全空要么全满，
  teacher 等于没判。所以取:

      BRAKE       1.5   ≈ 减速侧 p95
      HARD_BRAKE  2.5   ≈ 减速侧 p99.9，NAVSIM 上的实际天花板
      ACCEL      +1.5   对称（加速侧分布与减速侧接近）

────────────────────────────────────────────────────────────────────────────
🔴 路径不做外插。

  加速档要走得比学生远，学生轨迹不够长。**沿最后朝向直线外插会撒谎** ——
  弯道上会直接插出可行驶区域，DAC 判假违规。
  这里改成:学生轨迹末点投影到 metric_cache 的 centerline 上，从那里沿 centerline 续。
  centerline 也不够长时，该档标 `uncheckable`，**弃权**，不进 D 也不算"不可行"
  （flow §4.3:未来越界必须返回不可评，不能 fail-open 当成"安全"）。
"""

from __future__ import annotations

import lzma
import math
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np

from navsim.common.dataclasses import Trajectory
from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

# 档位 → 纵向加速度 (m/s²)。见文件头的重标依据。
PROFILES: Dict[str, float] = {
    "HARD_BRAKE": -2.5,
    "BRAKE": -1.5,
    "KEEP": 0.0,
    "ACCEL": +1.5,
}
# 与 <PLAN> 的速度词表的对应（teacher 判决要落回那 4 个词）
PLAN_OF_PROFILE = {"HARD_BRAKE": "STOP", "BRAKE": "DECELERATE",
                   "KEEP": "KEEP", "ACCEL": "ACCELERATE"}

V_MAX = 16.0            # m/s，navtest v0 最大 14.6，留一点余量
PATH_SLACK = 0.5        # m，允许的路径末端容差


def _resample_path(xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """折线 → (点, 累积弧长)。"""
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return xy, np.concatenate([[0.0], np.cumsum(seg)])


def _interp_xy(xy: np.ndarray, s: np.ndarray, q: np.ndarray) -> Optional[np.ndarray]:
    """按弧长 q 在折线上取点。q 超出 s[-1]+PATH_SLACK → None（不可评）。"""
    if float(q.max()) > float(s[-1]) + PATH_SLACK:
        return None
    return np.stack([np.interp(q, s, xy[:, 0]), np.interp(q, s, xy[:, 1])], axis=1)


def _headings(pts: np.ndarray, prev: Tuple[float, float]) -> np.ndarray:
    """逐点朝向:指向下一点;最后一点沿用上一段。几乎不动时沿用前一个朝向。"""
    out, last = [], 0.0
    allp = np.vstack([np.asarray(prev, dtype=np.float64)[None, :], pts])
    for i in range(1, len(allp)):
        d = allp[i] - allp[i - 1]
        if float(np.hypot(*d)) > 0.05:
            last = math.atan2(d[1], d[0])
        out.append(last)
    return np.asarray(out)


def speed_schedule(v0: float, a: float, n: int, dt: float) -> np.ndarray:
    """恒定加速度下每个采样时刻的累积弧长。速度夹在 [0, V_MAX]，停住就不再前进。"""
    v, s, out = float(v0), 0.0, []
    for _ in range(n):
        v_next = min(max(v + a * dt, 0.0), V_MAX)
        s += 0.5 * (v + v_next) * dt         # 梯形积分
        v = v_next
        out.append(s)
    return np.asarray(out)


class FeasibleSet:
    """给一帧算可行集。metric_cache 逐 token 现取现用（lzma 解压，~10ms）。"""

    def __init__(self, metric_cache_path, num_poses: int = 10, interval: float = 0.5):
        self.loader = MetricCacheLoader(metric_cache_path)
        self.num_poses, self.interval = num_poses, interval
        self.model_sampling = TrajectorySampling(num_poses=num_poses, interval_length=interval)
        self.future_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
        self.simulator = PDMSimulator(self.future_sampling)
        self.scorer = PDMScorer(self.future_sampling)
        self._cache: Dict[str, object] = {}

    def metric_cache(self, token: str):
        if token not in self._cache:
            with lzma.open(self.loader.metric_cache_paths[token], "rb") as f:
                self._cache[token] = pickle.load(f)
        return self._cache[token]

    # ── 几何:学生路径 + centerline 续接，全部在 ego 局部系 ──────────────
    def _path(self, token: str, base_poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        xy = np.vstack([[0.0, 0.0], np.asarray(base_poses, dtype=np.float64)[:, :2]])
        mc = self.metric_cache(token)
        try:
            ego = mc.ego_state.rear_axle
            ex, ey, eh = float(ego.x), float(ego.y), float(ego.heading)
            cl = mc.centerline
            # 学生末点 → 全局 → 投影到 centerline → 沿 centerline 再取 60m
            c, s_ = math.cos(eh), math.sin(eh)
            gx = ex + xy[-1, 0] * c - xy[-1, 1] * s_
            gy = ey + xy[-1, 0] * s_ + xy[-1, 1] * c
            from shapely.geometry import Point
            s0 = float(cl.project(Point(gx, gy)))
            q = np.arange(s0 + 2.0, min(s0 + 62.0, float(cl.length)), 2.0)
            if len(q) >= 2:
                arr = np.asarray(cl.interpolate(q, as_array=True))[:, :2]
                dx, dy = arr[:, 0] - ex, arr[:, 1] - ey
                loc = np.stack([dx * c + dy * s_, -dx * s_ + dy * c], axis=1)
                xy = np.vstack([xy, loc])
        except Exception:
            pass                                   # 续不上就只用学生路径，超长的档会弃权
        return _resample_path(xy)

    # ── 合成一档 ────────────────────────────────────────────────────────
    def synth(self, token: str, base_poses: np.ndarray, v0: float,
              mode: str) -> Optional[Trajectory]:
        xy, s = self._path(token, base_poses)
        q = speed_schedule(v0, PROFILES[mode], self.num_poses, self.interval)
        pts = _interp_xy(xy, s, q)
        if pts is None:
            return None                            # 路径不够长 → 不可评
        head = _headings(pts, (0.0, 0.0))
        return Trajectory(np.concatenate([pts, head[:, None]], axis=1).astype(np.float32),
                          self.model_sampling)

    # ── 评一条轨迹 ──────────────────────────────────────────────────────
    def score(self, token: str, traj: Trajectory) -> Optional[dict]:
        try:
            r = pdm_score(metric_cache=self.metric_cache(token), model_trajectory=traj,
                          future_sampling=self.future_sampling,
                          simulator=self.simulator, scorer=self.scorer)
        except Exception:
            return None
        return {"score": float(r.score), "nc": float(r.no_at_fault_collisions),
                "dac": float(r.drivable_area_compliance), "ep": float(r.ego_progress),
                "ttc": float(r.time_to_collision_within_bound),
                "ddc": float(r.driving_direction_compliance)}

    @staticmethod
    def is_feasible(m: dict) -> bool:
        """可行 = 无归责碰撞 且 在可行驶区域内。

        ⚠️ 故意**不**把 ego_progress 写进这里:STOP 的 EP 天然接近 0，
           加进来会让"停"永远不可行。推进那一条是判决侧的条件，不是知识侧的。
        """
        return m["nc"] >= 1.0 and m["dac"] >= 1.0

    def feasible_set(self, token: str, base_poses: np.ndarray, v0: float) -> dict:
        """→ {'D': [...可行档...], 'uncheckable': [...], 'metrics': {档: 子判据}}"""
        D, unck, met = [], [], {}
        for mode in PROFILES:
            traj = self.synth(token, base_poses, v0, mode)
            if traj is None:
                unck.append(mode); continue
            m = self.score(token, traj)
            if m is None:
                unck.append(mode); continue
            met[mode] = m
            if self.is_feasible(m):
                D.append(mode)
        return {"D": D, "uncheckable": unck, "metrics": met}
