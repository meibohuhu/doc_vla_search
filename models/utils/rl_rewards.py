"""
GRPO 的 reward 分项 —— 除 PDMS 之外的部分。

对应 simlingo 侧的 `simlingo_training/rl/rewards.py`。这里只有一项**进 reward**:

    format      格式/退化惩罚      —— 堵住"薅 reward 的废话"

另有一项**只作为指标、不进 reward**:

    consistency reasoning 里的 <PLAN> ↔ 它自己轨迹算出来的档
                🔴 写进 reward 等于用手把 reasoning 和轨迹焊在一起，而
                   "reasoning 变好 → 轨迹变好"正是本方法要【论证】的东西(paper §7)。
                   焊完再去测这条因果，测到的是自己刚加的那一项。
                   留着当观测量:开跑前基线多少、DiRL 训完涨没涨。

teacher 相关的两项(`w_viol` 可行性罚、`w_say` 的 Δ 定价)要等
反事实剖面→可行集 D 落地,见 docs/0806/vcrd_flow_autovla.md §6.2/6.4/6.5。

────────────────────────────────────────────────────────────────────────────
⚠️ `plan_from_track` 是从
   other_repo/Impromptu-VLA/data_qa_generate/data_engine/datasets/navsim/navsim_agent_qa.py
   **抄过来的 calib=True 分支**(连同 4 个阈值常数)。跨 repo import 太脆,所以复制。

   **两边必须保持一致** —— 它既是 CoT 训练标签的产生规则,又是这里判"轨迹实际在做什么"
   的规则。一致性 reward 的全部意义就在于两者同规则:模型说 STOP、而用同一把尺子量它自己
   的轨迹也是 STOP,才叫一致。改了那边一定要同步改这边。
   (已验证:用本文件的实现在 GT 轨迹上重算,与 cot_navtrain.json 的标签逐条一致。)
"""

import math
import re
from collections import Counter
from typing import List, Optional, Sequence, Tuple

import numpy as np

# ── 标签规则的常数(与 navsim_agent_qa.py 同步)────────────────────────────
NUM_FUT = 6                      # SPEED 判据窗口:6 帧 @2Hz = 3.0s
FUT_HORIZON = NUM_FUT / 2.0      # 秒
VAL_STOP = 2.0                   # m/s
A_DEC = 0.4                      # m/s²
A_ACC = 0.4
PATH_HEAD_DEG = 30.0
PATH_LAT_RATIO = 0.20

SPEED_WORDS = ("STOP", "DECELERATE", "KEEP", "ACCELERATE")
PATH_WORDS = ("LEFT", "STRAIGHT", "RIGHT")

_PEDAL = {"const": "KEEP", "accelerate": "ACCELERATE",
          "decelerate": "DECELERATE", "stop": "STOP"}
_PATH = {"left": "LEFT", "right": "RIGHT", "straight": "STRAIGHT"}


def plan_from_track(xy: Sequence[Tuple[float, float]],
                    fps: float = 2.0) -> Optional[Tuple[float, str, str]]:
    """一段 ego 局部系轨迹 → (v0, SPEED, PATH)。点数不足返回 None(= 不可评,应弃权)。

    xy[0] 必须是当前位置 (0,0),后面是未来位姿。
    """
    if len(xy) < NUM_FUT + 1:
        return None
    pts = np.asarray(xy, dtype=np.float64)
    step = np.linalg.norm(np.diff(pts, axis=0), axis=1) * fps
    speeds = np.append(step, [0.0])                 # 与标签生成侧对齐:末点补 0
    if len(speeds) <= NUM_FUT:
        return None

    v0, vT = float(speeds[0]), float(speeds[NUM_FUT])
    a = (vT - v0) / FUT_HORIZON
    # STOP:窗口末慢，【且】整段确实停住过 或 末了仍然慢（否则会把起步初期误判成 STOP）
    v_min = float(speeds[:NUM_FUT + 1].min())
    v_end = float(speeds[-2]) if len(speeds) >= 2 else vT
    if vT < VAL_STOP and (v_min < 0.5 or v_end < VAL_STOP):
        speed_plan = "stop"
    elif a <= -A_DEC:
        speed_plan = "decelerate"
    elif a >= A_ACC:
        speed_plan = "accelerate"
    else:
        speed_plan = "const"

    lat = float(pts[-1][1] - pts[0][1])
    lon = float(pts[-1][0] - pts[0][0])
    ratio = abs(lat) / max(abs(lon), 1e-6)
    d = pts[-1] - pts[-3] if len(pts) >= 3 else pts[-1] - pts[0]
    head = math.degrees(math.atan2(d[1], d[0])) if float(np.hypot(*d)) > 0.5 else 0.0
    if abs(head) >= PATH_HEAD_DEG or ratio >= PATH_LAT_RATIO:
        path_plan = "left" if lat > 0 else "right"
    else:
        path_plan = "straight"

    return v0, _PEDAL[speed_plan], _PATH[path_plan]


def trajectory_plan(trajectory_poses) -> Optional[Tuple[str, str]]:
    """把 action token 解出来的轨迹(N×≥2，ego 局部系，不含当前位姿)算成 (SPEED, PATH)。"""
    if trajectory_poses is None or len(trajectory_poses) < NUM_FUT:
        return None
    p = np.asarray(trajectory_poses, dtype=np.float64)[:, :2]
    out = plan_from_track([(0.0, 0.0)] + [tuple(q) for q in p])
    return None if out is None else (out[1], out[2])


# ── reasoning 侧的解析与格式闸门 ─────────────────────────────────────────

_PLAN_RE = re.compile(r"<PLAN>\s*([A-Z_]+)\s*,\s*([A-Z_]+)\s*</PLAN>")
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)
# 模型偶尔跑偏输出中文 → 判 off-format（CJK 扩展A + 基本区）
_CJK_RE = re.compile("[㐀-䶿一-鿿]")


def parse_plan(text: str) -> Optional[Tuple[str, str]]:
    """从 completion 里抽 <PLAN>SPEED,PATH</PLAN>。抽不到或词不在词表里 → None。"""
    if not text:
        return None
    m = _PLAN_RE.search(text)
    if not m:
        return None
    sp, pa = m.group(1), m.group(2)
    return (sp, pa) if sp in SPEED_WORDS and pa in PATH_WORDS else None


def is_degenerate(text: str, max_words: int = 60) -> bool:
    """退化 reasoning:空 / 过长 / 单词刷屏 / 短语循环。

    真实 reasoning p50 = 32 token、最长 4 个从句，max_words=60 已经很松。
    这一项针对的是 memory `grpo-reasoning-reward-hacking` 记的那类退化
    （reasoning 塌成空串或复读，而轨迹照样拿分）。
    """
    if not text or not text.strip():
        return True
    w = text.lower().split()
    if len(w) > max_words:
        return True
    if len(w) >= 6:
        if Counter(w).most_common(1)[0][1] / len(w) > 0.4:      # 单词刷屏
            return True
        run = 1
        for i in range(1, len(w)):
            run = run + 1 if w[i] == w[i - 1] else 1
            if run >= 4:                                        # 连续同词 ≥4
                return True
        for n in (2, 3):                                        # n-gram 循环 ≥4
            g = Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
            if g and g.most_common(1)[0][1] >= 4:
                return True
    return False


def format_violation(text: str, expect_think: bool) -> bool:
    """这条 completion 是否 off-format(→ 吃 w_fmt 的常数罚)。

    expect_think:该样本的 prompt 是否要求先说理由(reason_then_act)。
      True  → 必须有非空 <think>，且里面能解析出 <PLAN>
      False → 不该有 <think>；只要求有 <answer>
    """
    if not text:
        return True
    if _CJK_RE.search(text):
        return True
    if _ANSWER_RE.search(text) is None:
        return True
    m = _THINK_RE.search(text)
    if not expect_think:
        return m is not None                    # 让直答的时候不要偷偷加推理
    if m is None:
        return True
    body = m.group(1).strip()
    return is_degenerate(body) or parse_plan(body) is None


def reward_consistency(text: str, trajectory_poses) -> float:
    """reasoning 说的 SPEED 和它自己轨迹算出来的 SPEED 是否一致。

    一致 +1 / 矛盾 −1 / 有一边判不出 0。

    🔴 **这是指标,不是 reward** —— 调用方只把它打到 wandb(`m_consistency`),
       不加进 R。名字里的 reward_ 是从 simlingo 那边搬来的历史包袱。

    ⭐ 两边走的是**同一个** plan_from_track —— 这正是它作为指标的意义:
      它不拿 reasoning 去比 GT，而是量 reasoning 和【模型自己的轨迹】有多一致，
      纯 self-consistency，不引入 GT 监督，也不惩罚"和 GT 不同但自洽"的开法。
    """
    m = _THINK_RE.search(text or "")
    plan = parse_plan(m.group(1) if m else (text or ""))
    traj = trajectory_plan(trajectory_poses)
    if plan is None or traj is None:
        return 0.0
    return 1.0 if plan[0] == traj[0] else -1.0
