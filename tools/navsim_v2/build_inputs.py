"""NAVSIM v2 两阶段 split → AutoVLA 的输入，一行一个 token。只取图片路径，不解码图片。

要在 v2 devkit 的 PYTHONPATH 下跑（run.sh 里是 `v2py`）：
    PYTHONPATH=$V2_DEVKIT python tools/navsim_v2/build_inputs.py --split warmup_two_stage --out <dir>

token 来自 v2 的 SceneLoader，和官方 run_create_submission_pickle.py 一样：
  第一阶段 = loader.tokens_stage_one（真实场景）
  第二阶段 = loader.reactive_tokens_stage_two（合成场景，3DGS 渲染）
每个 token 用 loader.get_agent_input_from_token 拿到 AgentInput，字段和 v1 navtest JSON 的来源一致
（navsim/navsim/agents/vla_agent.py 的 VlaAgentFeatureBuilder）：
  velocity / acceleration = ego_statuses[-1] 的二维向量（get_prompt 里取模）
  instruction            = driving_command one-hot → turn left / keep forward / turn right / unknown
  cams                   = 每路 4 帧绝对路径（旧 → 新），存 F0 / L0 / R0 / L1 / R1，推理时按 ckpt 选侧视
输出 <out>/inputs.jsonl 和 <out>/inputs_summary.json（token 数、补图情况）。

缺图：某帧图在本地找不到时，用同一路相机最近的一帧代替（优先更新的帧），记在 "filled" 里。
  warmup 第一阶段有 5 个 token 缺最早一帧（-1.5 s）：本地 sensor_blobs/test 只留了 navtest 用到的帧，
  而 warmup 包本身不带第一阶段的图。navhard 的包自带第一阶段的图（curr / hist sensors）。
"""
import argparse
import dataclasses
import json
import os
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

import navsim
from navsim.common import dataclasses as nd
from navsim.common.dataloader import SceneLoader

CAMS = ("cam_f0", "cam_l0", "cam_r0", "cam_l1", "cam_r1")
COMMANDS = ("turn left", "keep forward", "turn right", "unknown")


def paths_only(cls, sensor_blobs_path, camera_dict, sensor_names):
    """替换 Cameras.from_camera_dict：只记相对路径，不 Image.open（原实现每帧解码 8 张 1920x1080）。"""
    cams = {}
    for name, spec in camera_dict.items():
        ident = name.lower()
        cams[ident] = nd.Camera(camera_path=spec["data_path"]) if ident in sensor_names else nd.Camera()
    return cls(**{f.name: cams.get(f.name, nd.Camera()) for f in dataclasses.fields(cls)})


nd.Cameras.from_camera_dict = classmethod(paths_only)


def split_cfg(devkit, split):
    base = f"{devkit}/navsim/planning/script/config/common/train_test_split"
    data_split = OmegaConf.load(f"{base}/{split}.yaml").data_split
    return instantiate(OmegaConf.load(f"{base}/scene_filter/{split}.yaml")), data_split


def resolve(rel, roots):
    for r in roots:
        p = os.path.join(r, rel)
        if os.path.exists(p):
            return p
    return None


def cam_paths(agent_input, roots):
    """每路相机 4 帧绝对路径；缺的帧用最近的帧补。返回 (cams, filled)。"""
    cams, filled = {}, []
    for c in CAMS:
        ps = [resolve(str(getattr(f, c).camera_path), roots) for f in agent_input.cameras]
        have = [i for i, p in enumerate(ps) if p]
        if not have:
            raise FileNotFoundError(f"{c} 4 帧都找不到：{getattr(agent_input.cameras[-1], c).camera_path}")
        for i, p in enumerate(ps):
            if p is None:
                j = min(have, key=lambda k: (abs(k - i), -k))
                ps[i] = ps[j]
                filled.append(f"{c}@{i}<-{j}")
        cams[c] = ps
    return cams, filled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="warmup_two_stage")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=os.environ.get("OPENSCENE_DATA_ROOT"))
    ap.add_argument("--devkit", default=os.path.dirname(os.path.dirname(navsim.__file__)))
    a = ap.parse_args()
    assert hasattr(SceneLoader, "tokens_stage_one"), f"import 到的 navsim 不是 v2：{navsim.__file__}（PYTHONPATH 要设成 v2 devkit）"

    scene_filter, data_split = split_cfg(a.devkit, a.split)
    syn_sensors = f"{a.data}/{a.split}/sensor_blobs"
    orig_sensors = f"{a.data}/sensor_blobs/{data_split}"
    loader = SceneLoader(
        data_path=Path(f"{a.data}/navsim_logs/{data_split}"),
        original_sensor_path=Path(orig_sensors),
        scene_filter=scene_filter,
        synthetic_sensor_path=Path(syn_sensors),
        synthetic_scenes_path=Path(f"{a.data}/{a.split}/synthetic_scene_pickles"),
        sensor_config=nd.SensorConfig(**{f.name: f.name in CAMS for f in dataclasses.fields(nd.SensorConfig)}),
    )
    stages = [(1, t) for t in sorted(loader.tokens_stage_one)] + [(2, t) for t in sorted(loader.reactive_tokens_stage_two)]
    print(f"{a.split}：第一阶段 {len(loader.tokens_stage_one)}，第二阶段 {len(loader.reactive_tokens_stage_two)}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    n_filled, bad, cmds = {1: 0, 2: 0}, [], {}
    with open(f"{a.out}/inputs.jsonl", "w") as f:
        for i, (stage, tok) in enumerate(stages):
            ai = loader.get_agent_input_from_token(tok)
            # 第一阶段的图先在原始 sensor_blobs 找，再到 split 自带的包里找（navhard 的包带第一阶段的图）
            roots = [orig_sensors, syn_sensors] if stage == 1 else [syn_sensors]
            try:
                cams, filled = cam_paths(ai, roots)
            except FileNotFoundError as e:
                bad.append({"token": tok, "stage": stage, "error": str(e)})
                continue
            ego = ai.ego_statuses[-1]
            cmd = COMMANDS[int(np.argmax(ego.driving_command))]
            cmds[cmd] = cmds.get(cmd, 0) + 1
            n_filled[stage] += bool(filled)
            f.write(json.dumps({
                "token": tok, "stage": stage,
                "velocity": [float(x) for x in ego.ego_velocity],
                "acceleration": [float(x) for x in ego.ego_acceleration],
                "instruction": cmd, "cams": cams, "filled": filled,
            }) + "\n")
            if i % 500 == 0:
                print(f"  {i + 1}/{len(stages)}", flush=True)

    summary = {"split": a.split, "n_stage1": len(loader.tokens_stage_one), "n_stage2": len(loader.reactive_tokens_stage_two),
               "tokens_with_filled_frames": n_filled, "unusable": bad, "instruction": cmds}
    json.dump(summary, open(f"{a.out}/inputs_summary.json", "w"), indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
