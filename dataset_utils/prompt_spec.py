"""
prompt 的**唯一**出处 —— SFT 与 RL 共用。

🔴 为什么要有这个文件:2026-09-02 之前，system / user 文案在两处各写了一份 ——
   dataset_utils/sft_dataset.py（训练）和 models/autovla.py::get_prompt（RL rollout / 推理）。
   两份已经漂开了:RL 那份还停在旧版 system prompt（要求"考虑红绿灯、车道线"、
   "If necessary, use step-by-step reasoning"），连撇号都是 ’ 而不是 '。
   **prompt 一漂，RL 就跑在训练分布外**，而这种偏差不会报错、只会让指标莫名其妙。
   现在两边都从这里取。

模式（见 docs/0901/sft_cot_sampling.md）:
    reason_then_act   prompt 要求先说理由   target = <think>…</think><answer>…
    reasoning_given   reasoning 给在 prompt 里，target = 只有 <answer>
    act_directly      prompt 只要动作       target = 只有 <answer>
    counterfactual    prompt 带 scene report + suggested action，不加结尾指令
                      target = <PLAN>S,D</PLAN> + <answer>（没有 <think>，见 docs/1026/cf_sft_autovla_plan.md §2）

⚠️ using_cot=False（纯 action 训练）时**不加任何后缀** —— 那条路径必须与
   引入 CoT 之前逐字节一致。
"""

SYSTEM_NOCOT = (
    "You are an Advanced Driver Assistance and Full Self-Driving System. "
    "You will be provided with video observations from the ego vehicle's surrounding cameras, "
    "along with the vehicle's current dynamic states. "
    "Your task is to predict the most appropriate driving action for the next five seconds."
)

# 🔴 CoT 分支的 system prompt 是重写过的。原文三处与本项目设定冲突:
#   ① 让模型"考虑红绿灯、车道线" —— 红绿灯已整条砍掉，车道线我们从来不说;
#   ② "If necessary, use CoT … Otherwise, you may directly predict" —— 把要不要推理
#      交给模型自己决定，而模式已在 user prompt 里逐样本指定，两者打架，
#      且"让模型自己决定"正是 fast-thinking 塌陷的成因（models/autovla.py:820）;
#   ③ "step-by-step reasoning" —— 我们的 reasoning 是一句话（p50=32 token）。
SYSTEM_COT = (
    "You are an autonomous driving system. "
    "You receive camera observations from the ego vehicle and its current dynamic state. "
    "Your task is to predict the driving action for the next five seconds.\n\n"
    "When asked to explain, say briefly what constrains you right now and what you will do. "
    "Then give the final action."
)


# ── 相机输入 ────────────────────────────────────────────────────────────────
# [mh 2026-10-05] 每路相机用几帧可配：model.video.num_frames（1-4，默认 4 = 原行为）。
#   JSON 里每路存 4 帧（旧→新，最后一帧 = 当前帧，2Hz：-1.5s/-1.0s/-0.5s/0s），这里取最后 N 帧。
#   🔴 SFT（dataset_utils/sft_dataset.py）与推理/RL（models/autovla.py::get_prompt）都从这里拼，
#      两边帧数和文案不可能对不上。num_frames=4 时与改动前逐字节一致。
#   N=1 仍按 video 喂（Qwen 按 temporal_patch_size=2 把单帧复制成 2 帧）：模型 forward / RFT
#      只认 pixel_values_videos / video_grid_thw，换成 image 会断。
CAMERA_INTRO = (
    "The autonomous vehicle is equipped with three cameras mounted at the front, left, and right, "
    "enabling a comprehensive perception of the surrounding environment."
)
_CAMERAS = (("front_camera", "first", "front"),
            ("front_left_camera", "second", "front-left"),
            ("front_right_camera", "third", "front-right"))
_NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four"}


def camera_content(camera_paths: dict, num_frames: int, min_pixels: int, max_pixels: int) -> list:
    """user content 里相机部分（开头介绍 + 3 路 video）。

    camera_paths[cam] 是该路 4 帧的完整路径（旧→新）；取最后 num_frames 帧，当前帧一定在。
    """
    if num_frames not in _NUM_WORDS:
        raise ValueError(f"model.video.num_frames 只能是 1-4，收到 {num_frames}")
    content = [{"type": "text", "text": CAMERA_INTRO}]
    for cam, ordinal, view in _CAMERAS:
        frames = camera_paths[cam][-num_frames:]
        if num_frames == 1:
            desc = (f"The {ordinal} video presents the {view} view of the vehicle, "
                    f"comprising only the current frame.")
        else:
            desc = (f"The {ordinal} video presents the {view} view of the vehicle, "
                    f"comprising {_NUM_WORDS[num_frames]} sequential frames sampled at 2 Hz.")
        content.append({"type": "text", "text": desc})
        content.append({"type": "video", "min_pixels": min_pixels, "max_pixels": max_pixels,
                        "video": [f"file://{p}" for p in frames]})
    return content


def system_text(using_cot: bool) -> str:
    return SYSTEM_COT if using_cot else SYSTEM_NOCOT


# ── 反事实指令（docs/1026/cf_sft_autovla_plan.md §2）─────────────────────────
# suggestion 只有 5 个；KEEP 不当 suggestion（KEEP 的 cf 样本只当解耦样本用）。
CF_SUGGESTIONS = {"STOP": "come to a stop", "DECELERATE": "slow down", "ACCELERATE": "speed up",
                  "LANE_LEFT": "change lanes to the left", "LANE_RIGHT": "change lanes to the right"}
# scene report：固定 7 句，按答案的动作随机配一句相符的，不看画面（simlingo 第一版的做法）。
CF_REPORTS = {
    "pedestrian": "A pedestrian is on the road ahead.",
    "cyclist": "A cyclist is on the road ahead.",
    "animal": "An animal is on the road ahead.",
    "stopped_vehicle": "A stopped vehicle is ahead in the ego lane.",
    "obstacle": "An obstacle is ahead in the ego lane.",
    "cut_in": "A vehicle is cutting into the ego lane.",
    "no_hazard": "No hazard is in the ego lane.",
}
_SLOW = ("pedestrian", "cyclist", "animal", "stopped_vehicle", "obstacle", "cut_in")
CF_ACTION_REPORTS = {"STOP": _SLOW, "DECELERATE": _SLOW, "ACCELERATE": ("no_hazard",), "KEEP": ("no_hazard",),
                     "LANE_LEFT": ("stopped_vehicle", "obstacle"), "LANE_RIGHT": ("stopped_vehicle", "obstacle")}
# <PLAN> 词表：方向在原来的 LEFT / STRAIGHT / RIGHT（跟着路转弯）之外加了换道。
# models/utils/rl_rewards.py 的 PATH_WORDS 没改（那份只读），评测解析换道时用这里的。
PLAN_SPEED_WORDS = ("STOP", "DECELERATE", "KEEP", "ACCELERATE")
PLAN_PATH_WORDS = ("LEFT", "STRAIGHT", "RIGHT", "LANE_LEFT", "LANE_RIGHT")


def user_text(velocity: float, acceleration: float, instruction: str,
              using_cot: bool, cot_mode: str = "act_directly",
              gt_cot: str = "", report: str = "", suggestion: str = "") -> str:
    """user prompt 正文。

    ⚠️ reasoning_given 与 act_directly 的指令尾巴**故意完全相同**（都是
       "Answer directly."），差别只有前者多了一段 reasoning 文本 —— 这样模型没法靠
       指令词分流，只能靠"prompt 里到底有没有 reasoning"来决定行为，
       要建立的正是这条 reasoning → trajectory 的依赖。simlingo 同样这么做
       （dataloader/dataset_driving.py:316）。
    """
    # ⚠️ base 结尾**没有空格** —— using_cot=False 时它就是最终 prompt，
    #    必须与引入 CoT 之前逐字节一致（原文以 "next five seconds." 结束）。
    #    模式后缀自带前导空格。
    base = (
        f"The current velocity of the vehicle is {velocity:.3f} m/s, "
        f"and the current acceleration is {acceleration:.3f} m/s². "
        f"The driving instruction is: {instruction}. Based on this information, "
        f"plan the action trajectory for the autonomous vehicle over the next five seconds."
    )
    if not using_cot:
        return base
    if cot_mode == "counterfactual":
        # 结尾不加 "Explain your reasoning first." / "Answer directly."：答案只有 <PLAN> 和动作，
        # 和普通样本靠 prompt 里有没有 Scene report / Suggested action 区分。
        return base + f" Scene report: {report} Suggested action: {suggestion}."
    if cot_mode == "reason_then_act":
        return base + " Explain your reasoning first."
    if cot_mode == "reasoning_given":
        return base + f" Here is the situation: {gt_cot} Answer directly."
    return base + " Answer directly."
