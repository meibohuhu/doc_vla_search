"""teacher 的判决与定价（flow §6.3–6.5）—— **默认关闭**，由 `rl.teacher.enable` 打开。

三块:
    触发器   学生轨迹不可行 → 这一条要不要管
    splice   把 <PLAN>SPEED,PATH</PLAN> 里的 SPEED 换成可行集里最近的一档 → z*
    Δ 定价   Δ = nll(z*) − nll(z)，clip 后进 reward（**不是 loss**）

────────────────────────────────────────────────────────────────────────────
🔴 触发器的闸门是量出来的，不是拍的。navtest 600 帧（ep2 greedy）实测:

    触发器定义                              触发率     chg
    NC=0 或 DAC=0（原始）                   15.67%   13.50%
    只认 NC=0                               3.67%    3.00%
    NC=0 或 (DAC=0 且 深度>0.3m 且 首次≤3s)   7.67%    6.83%

   ⚠️ 69.5% 的触发是**纯横向失效**（DAC=0 而 NC=1）。沿学生路径改速度救不了它们 ——
      D 之所以非空，只是因为"刹车让车在 4s 窗口内走不到出界的那一段"。
      teacher 会学到"你转向不准，那就别开那么快"。见 logs/0902/step0_gonogo.md。

   → 默认 trigger = `nc_only`。要放宽就用 `nc_or_deep_dac`（带深度/时刻闸门），
     `nc_or_dac` 只为复现原始行为保留，**不建议用**。
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from models.utils.feasible import PLAN_OF_PROFILE, PROFILES, FeasibleSet

# SPEED 轴的顺序（慢 → 快）。判决就是在这条数轴上取最近的可行档。
SPEED_AXIS = ["STOP", "DECELERATE", "KEEP", "ACCELERATE"]
PROFILE_OF_PLAN = {v: k for k, v in PLAN_OF_PROFILE.items()}

# 决策从句的模板 —— 与 navsim_agent_qa.py::_speed_verb 同步。
#   head = f"{reason}, I should {verb}" 或 f"I should {verb}"
#   verb = _SPEED_VERB[SPEED]（+ " and bear left/right"）
#   STOP 的动词取决于当前是否已经停着:v0 < VAL_STOP → "remain stopped"，否则 "come to a stop"
VAL_STOP = 2.0
SPEED_VERB = {"DECELERATE": "slow down", "KEEP": "hold my current speed",
              "ACCELERATE": "speed up"}
STOP_VERBS = ("remain stopped", "come to a stop")
_ALL_VERBS = sorted(list(SPEED_VERB.values()) + list(STOP_VERBS), key=len, reverse=True)

_PLAN_RE = re.compile(r"(<PLAN>\s*)([A-Z_]+)(\s*,\s*[A-Z_]+\s*</PLAN>)")
# "I should <verb>[ and bear left/right]: <PLAN>...
_DECISION_RE = re.compile(
    r"(I should )(" + "|".join(re.escape(v) for v in _ALL_VERBS) + r")"
    r"((?: and bear (?:left|right))?)(\s*:\s*<PLAN>)")


def speed_verb(speed: str, v0: float) -> str:
    if speed != "STOP":
        return SPEED_VERB[speed]
    return "remain stopped" if v0 < VAL_STOP else "come to a stop"


# ego 车身（Pacifica）。DAC 判的是四角，不是中心线。
EGO_L, EGO_W, RAC = 5.176, 2.297, 1.461


# ── 出界深度 / 时刻 ─────────────────────────────────────────────────────

def _corners(x, y, h, ex, ey, eh):
    c, s = math.cos(eh), math.sin(eh)
    gx, gy = ex + x * c - y * s, ey + x * s + y * c
    gh = eh + h
    cx, cy = gx + RAC * math.cos(gh), gy + RAC * math.sin(gh)
    hl, hw = EGO_L / 2, EGO_W / 2
    cc, ss = math.cos(gh), math.sin(gh)
    return [(cx + cc * dx - ss * dy, cy + ss * dx + cc * dy)
            for dx, dy in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]


def dac_penetration(metric_cache, poses, dt: float = 0.5,
                    horizon: float = 4.0) -> Tuple[float, float]:
    """→ (前 horizon 秒内的最大出界深度 m, 首次出界时刻 s)。没出界返回 (0.0, inf)。

    ⚠️ 这里量的是**模型输出的 waypoint**，而 PDM 评的是 LQR 跟踪仿真出来的轨迹，
       两者不完全重合（实测约 17% 的 DAC=0 在 waypoint 上看不到出界）。
       所以这只是闸门用的近似，别当成 PDM 自己的读数。
    """
    from shapely.geometry import Point
    from shapely.ops import unary_union
    e = metric_cache.ego_state.rear_axle
    ex, ey, eh = float(e.x), float(e.y), float(e.heading)
    U = unary_union(list(metric_cache.drivable_area_map._geometries))
    P = np.asarray(poses)
    depth, first = 0.0, float("inf")
    for k in range(len(P)):
        t = dt * (k + 1)
        d = max(U.distance(Point(*c)) for c in _corners(P[k, 0], P[k, 1], P[k, 2], ex, ey, eh))
        if d > 1e-6:
            first = min(first, t)
            if t <= horizon:
                depth = max(depth, d)
    return depth, first


# ── splice ─────────────────────────────────────────────────────────────

def plan_span(text: str) -> Optional[Tuple[int, int, str]]:
    """→ (SPEED 词在 text 里的 char 起止, 该词)。找不到返回 None。"""
    m = _PLAN_RE.search(text or "")
    if m is None:
        return None
    return m.start(2), m.end(2), m.group(2)


def decision_span(text: str) -> Optional[Tuple[int, int]]:
    """决策从句的 char 区间:从 "I should" 到 "</PLAN>" 结束。

    ⚠️ Δ 要在**这整段**上算，不能只取 tag 里那个 SPEED 词 ——
       splice 同时改了散文动词和 tag，只量一处会漏掉另一处的似然变化。
       实测这段约 10 个 token，而整条 completion 是 107 个，仍然足够集中。
    """
    m = _DECISION_RE.search(text or "")
    if m is None:
        return None
    end = text.find("</PLAN>", m.end())
    if end < 0:
        return None
    return m.start(), end + len("</PLAN>")


def splice(text: str, new_speed: str, v0: float = 0.0) -> Optional[str]:
    """把决策改写成 new_speed —— **散文动词短语和 <PLAN> tag 必须一起改。**

    🔴 只改 tag 是错的。句子长这样:

        "I should speed up: <PLAN>ACCELERATE,STRAIGHT</PLAN>"
                  ↑ 散文里已经把决策说了一遍

       只把 tag 换成 KEEP，就造出一个自相矛盾的句子，而且给定前缀
       "I should speed up: <PLAN>" 之后 ACCELERATE 几乎是确定的 ——
       实测 nll(z)≈0、nll(z*)≈23.9，Δ 量到的是"矛盾程度"，不是"模型对被纠正的
       判决有多抗拒"。这正是 simlingo 那 28% 病句的同一个坑，AutoVLA 并没有豁免。

    改法:散文动词按模板替换（" and bear left/right" 的 PATH 后缀原样保留），
         tag 里的 SPEED 同步替换。两处都改不到就返回 None（弃权，不硬拼）。
    """
    m = _DECISION_RE.search(text or "")
    if m is None:
        return None
    verb = speed_verb(new_speed, v0)
    out = text[:m.start()] + f"{m.group(1)}{verb}{m.group(3)}{m.group(4)}" + text[m.end():]
    sp = plan_span(out)
    if sp is None:
        return None
    a, b, _ = sp
    return out[:a] + new_speed + out[b:]


def nearest_feasible_speed(said: str, D: List[str]) -> Optional[str]:
    """在 SPEED 数轴上取离 said 最近的可行档。

    ⚠️ 快慢两侧等距都可行 → 返回 None（弃权），不硬选一侧。
       这是 simlingo 上踩出来的规则（flow §4.3）。
    """
    if said not in SPEED_AXIS:
        return None
    words = [PLAN_OF_PROFILE[m] for m in D if m in PLAN_OF_PROFILE]
    cand = [w for w in words if w in SPEED_AXIS]
    if not cand:
        return None
    i = SPEED_AXIS.index(said)
    best = min(abs(SPEED_AXIS.index(w) - i) for w in cand)
    tie = [w for w in cand if abs(SPEED_AXIS.index(w) - i) == best]
    if len(tie) > 1:
        return None                      # 两侧都有，弃权
    return tie[0]


# ── 判决 ───────────────────────────────────────────────────────────────

class TeacherVerdict:
    """一帧一条 rollout 的判决。**不碰模型**，只吃轨迹和文本。"""

    def __init__(self, metric_cache_path, trigger: str = "nc_only",
                 dac_depth_min: float = 0.3, dac_time_max: float = 3.0):
        assert trigger in ("nc_only", "nc_or_dac", "nc_or_deep_dac"), trigger
        self.fs = FeasibleSet(metric_cache_path)
        self.trigger = trigger
        self.dac_depth_min = dac_depth_min
        self.dac_time_max = dac_time_max

    def _fires(self, token, poses, m: dict) -> Tuple[bool, str]:
        if m["nc"] < 1.0:
            return True, "collision"
        if m["dac"] >= 1.0:
            return False, "feasible"
        if self.trigger == "nc_only":
            return False, "lateral_ignored"
        if self.trigger == "nc_or_dac":
            return True, "offroad"
        depth, first = dac_penetration(self.fs.metric_cache(token), poses)
        if depth > self.dac_depth_min and first <= self.dac_time_max:
            return True, "offroad_deep"
        return False, "offroad_marginal"

    def judge(self, token: str, poses, v0: float, text: str,
              metrics: Optional[dict] = None) -> Dict:
        """→ dict(fire, reason, D, said, target, z_star, abstain)

        `target is None` 一律表示**弃权**:不产生 w_say 的代价。
        """
        out = {"fire": False, "reason": "", "D": [], "said": None,
               "target": None, "z_star": None, "abstain": None}
        m = metrics
        if m is None:
            from navsim.common.dataclasses import Trajectory
            from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
            m = self.fs.score(token, Trajectory(np.asarray(poses, dtype=np.float32),
                                                TrajectorySampling(num_poses=10, interval_length=0.5)))
        if m is None:
            out["abstain"] = "uncheckable"
            return out

        fire, reason = self._fires(token, poses, m)
        out["fire"], out["reason"] = fire, reason
        if not fire:
            return out

        sp = plan_span(text)
        if sp is None:
            out["abstain"] = "unparseable"          # 说不出决策词 → 没得改，交给 w_fmt 管
            return out
        said = sp[2]
        out["said"] = said

        D = self.fs.feasible_set(token, np.asarray(poses), float(v0))
        out["D"] = D["D"]
        if not D["D"]:
            out["abstain"] = "empty_D"              # 造不出可行档 → 弃权，不是"随便挑一个"
            return out
        if PROFILE_OF_PLAN.get(said) in D["D"]:
            out["abstain"] = "said_in_D"            # 说得对，是开得不准 → 不改
            return out

        target = nearest_feasible_speed(said, D["D"])
        if target is None:
            out["abstain"] = "ambiguous"            # 快慢两侧都可行
            return out
        z = splice(text, target, float(v0))
        if z is None or z == text:
            out["abstain"] = "splice_failed"
            return out
        out["target"], out["z_star"] = target, z
        return out
