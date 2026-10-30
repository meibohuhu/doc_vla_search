import os
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from models.action_tokenizer import ActionTokenizer
from navsim.agents.autovla_agent import AutoVLAAgent
from dataset_utils import prompt_spec
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

import random

IGNORE_INDEX = -100

class SFTDataset(Dataset):
    def __init__(self, data_config, model_config, processor, using_cot=True):
        data_paths = data_config['json_dataset_path']
        sensor_data_paths = data_config['sensor_data_path']

        # Handle both single path (string/Path) and multiple paths (list)
        if isinstance(data_paths, (str, Path)):
            self.data_paths = [Path(data_paths)]
        else:
            self.data_paths = [Path(path) for path in data_paths]
        
        # Handle sensor_data_path - can be single value or list matching data_paths
        if isinstance(sensor_data_paths, list):
            self.sensor_data_paths = sensor_data_paths
        else:
            # Single value - use for all data paths
            self.sensor_data_paths = [sensor_data_paths] * len(self.data_paths)
        
        # Validate lengths match
        if len(self.sensor_data_paths) != len(self.data_paths):
            raise ValueError(
                f"Number of sensor_data_paths ({len(self.sensor_data_paths)}) must match "
                f"number of json_dataset_paths ({len(self.data_paths)})"
            )
            
        self.processor = processor
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer, model_config=model_config)
        # Flag to control whether to use CoT in training data
        self.using_cot = using_cot
        # ── CoT 采样比例（simlingo 式）────────────────────────────────────
        # 🔴 原版是【按数据有没有 CoT】决定走 think 分支，且两条分支各带一句
        #    "This is a complex scenario…" / "This is a straightforward scenario…"。
        #    后果记在 models/autovla.py:820 —— 发布的 RFT ckpt 在 navtest 上
        #    100% 走 fast thinking，<think> 里恒为 "straightforward"，一次真实推理都不做。
        #
        # 改成 simlingo 的做法（dataloader/dataset_driving.py:312 起）:
        #    每条样本【随机】选模式，而且 **prompt 侧也区分**:
        #      reason_then_act  prompt 要求先分析再给动作   answer: reasoning + <answer>
        #      act_directly     prompt 只要动作             answer: 只有 <answer>
        #    ⚠️ prompt 必须区分。只在 answer 侧随机、prompt 不变的话，模型无从判断
        #       该走哪条，必然塌向便宜的那条——就是上面那个 fast-thinking 坑。
        #
        # 三种模式，比例对齐 simlingo（commentary_ratio=0.75，其中 in-prompt 18%）:
        #   reason_then_act   0.75 × 0.82 = 61.5%   prompt 要求先说，target = reasoning + action
        #   reasoning_given   0.75 × 0.18 = 13.5%   reasoning【给在 prompt 里】，target = 只有 action
        #   act_directly                    25.0%   prompt 只要动作，target = 只有 action
        #
        # ⭐ reasoning_given 不是可有可无的一档。它教的是【给定 reasoning → 产出轨迹】
        #    这条通路，强迫轨迹去条件于 reasoning 文本，而不是绕过它。
        #    没有它，waypoint head 会直接绕开 reasoning（simlingo 上实测过:reasoning
        #    退化成空/重复而轨迹照样好），那本项目要证明的
        #    "improving reasoning benefits trajectory generation" 就无从谈起；
        #    §7 的因果验证（同一帧喂不同 reasoning、看轨迹变不变）也会落在分布外。
        #
        # 参数（config 的 model: 段）:
        #   use_cot: false                → 纯 action 训练。**与改动前逐字节一致**
        #   use_cot: true
        #     cot_ratio: 0.75             有 reasoning 参与的比例（其余为 act_directly）
        #     cot_in_prompt_ratio: 0.18   上者之中，reasoning 给在 prompt 里的比例
        #     cot_ratio: 1.0, cot_in_prompt_ratio: 0.0   → 全部 reason_then_act（消融）
        self.cot_ratio = float(model_config.get('cot_ratio', 0.75))
        self.cot_in_prompt_ratio = float(model_config.get('cot_in_prompt_ratio', 0.18))
        # [mh 2026-10-05] 每路相机用最后几帧（1-4），默认 4 = 原行为。见 prompt_spec.camera_content
        self.num_frames = int(model_config.get('video', {}).get('num_frames', 4))

        trajectory_sampling = TrajectorySampling(time_horizon=model_config['trajectory']['time_horizon'], 
                                                interval_length=model_config['trajectory']['interval_length'])
        # Use first sensor_data_path for agent init (will be overridden per-scene)
        nuplan_agent = AutoVLAAgent(trajectory_sampling=trajectory_sampling,
                                    sensor_data_path=self.sensor_data_paths[0],
                                   codebook_cache_path=model_config['codebook_cache_path'],
                                   skip_model_load=True)

        self._agent = nuplan_agent
        
        # Get all JSON files from all data paths, tracking which sensor_data_path to use
        # Store as tuples: (scene_path, sensor_data_path)
        self.scenes = []
        for data_path, sensor_path in zip(self.data_paths, self.sensor_data_paths):
            path_scenes = sorted(list(data_path.glob('*.json')))
            for scene in path_scenes:
                self.scenes.append((scene, sensor_path))
            
        if not self.scenes:
            raise ValueError(f"No JSON files found in any of the provided data paths: {self.data_paths}")

        # ── 像素数读 config 的 model.video（docs/1026/cf_sft_autovla_plan.md §4 C4）──────
        # 原来写死 28*28*128，而推理（models/autovla.py get_prompt）读 model.video（=140），
        # 训练和推理的图像 token 数对不上。没写 model.video 时保持 128，和改之前一致。
        video_conf = model_config.get('video') or {}
        self.min_pixels = int(video_conf.get('min_pixels', 28 * 28 * 128))
        self.max_pixels = int(video_conf.get('max_pixels', 28 * 28 * 128))

        # ── 反事实指令样本（docs/1026/cf_sft_autovla_plan.md §2、§4 C2）────────────────
        # model.cf.ratio 的概率把这一条换成 cf 样本：按类别比例抽一条 cf，按 token 读原 JSON，
        # 把 gt_trajectory 换成 cf 轨迹（compute_targets 会现场编码成 action token）。
        # 没有 data.<split>.cf_index 或 ratio 为 0 时完全不走这条路，也不多消耗随机数。
        cf_conf = model_config.get('cf') or {}
        cf_index = data_config.get('cf_index')
        self.cf_ratio = float(cf_conf.get('ratio', 0.0)) if cf_index else 0.0
        if self.cf_ratio > 0:
            if not using_cot:
                raise ValueError("cf 样本用 CoT 的 system prompt，需要 model.use_cot: true")
            z = np.load(cf_index)
            self.cf_token, self.cf_traj = z['token'], z['traj']
            self.cf_speed, self.cf_path, cls = z['plan_speed'], z['plan_path'], z['cls']
            self.cf_pools = {str(c): np.where(cls == c)[0] for c in np.unique(cls)}

            def _norm(probs, name):
                bad = [c for c in probs if c not in self.cf_pools]
                if bad:
                    raise ValueError(f"model.cf.{name} 里的类别 {bad} 在 {cf_index} 里没有")
                s = float(sum(probs.values()))
                return {c: float(p) / s for c, p in probs.items() if p > 0}

            # 照做样本：只有 5 个 suggestion 对应的类别（KEEP 不当 suggestion）
            self.cf_type_probs = _norm(cf_conf['type_probs'], 'type_probs')
            if 'KEEP' in self.cf_type_probs:
                raise ValueError("KEEP 不当 suggestion，只能放在 model.cf.decouple_type_probs 里")
            # 解耦样本：suggestion 换成别的动作、只训 action token；KEEP 只出现在这里
            self.cf_decouple_ratio = float(cf_conf.get('decouple_ratio', 0.1))
            self.cf_decouple_probs = _norm(cf_conf.get('decouple_type_probs', cf_conf['type_probs']),
                                           'decouple_type_probs')
            self.scene_by_token = {Path(p).stem: (p, s) for p, s in self.scenes}
            missing = set(self.cf_token.tolist()) - set(self.scene_by_token)
            if missing:
                raise ValueError(f"{cf_index} 里有 {len(missing)} 个 token 在 {self.data_paths} 里找不到 JSON")
            print(f"[cf] {cf_index}: {len(self.cf_token)} 条  ratio={self.cf_ratio}  "
                  f"decouple={self.cf_decouple_ratio}  type_probs={self.cf_type_probs}  "
                  f"decouple_type_probs={self.cf_decouple_probs}")

    def _sample_cf(self) -> Dict[str, Any]:
        """抽一条 cf 样本：类别 → 池子里随机一条 → report（和答案的动作相符）和 suggestion。"""
        decouple = random.random() < self.cf_decouple_ratio
        probs = self.cf_decouple_probs if decouple else self.cf_type_probs
        cls = random.choices(list(probs), weights=list(probs.values()))[0]
        pool = self.cf_pools[cls]
        j = int(pool[random.randrange(len(pool))])
        report = prompt_spec.CF_REPORTS[random.choice(prompt_spec.CF_ACTION_REPORTS[cls])]
        own = prompt_spec.CF_SUGGESTIONS.get(cls)        # KEEP 没有
        if decouple:
            suggestion = random.choice([s for s in prompt_spec.CF_SUGGESTIONS.values() if s != own])
        else:
            suggestion = own
        return {"row": j, "token": str(self.cf_token[j]), "cls": cls,
                "traj": self.cf_traj[j].astype(np.float64).tolist(),
                "plan": f"{self.cf_speed[j]},{self.cf_path[j]}",
                "report": report, "suggestion": suggestion, "decouple": decouple}

    def __len__(self):
        # return len(self._scene_loader.tokens)
        return len(self.scenes)

    def __getitem__(self, idx):
        # Load data from JSON file
        input_features: Dict[str, torch.Tensor] = {}
        target_trajectory: Dict[str, torch.Tensor] = {}
        
        # cf 样本：不用 idx 这一帧，按类别另抽一条，读它的原 JSON 再换掉 gt_trajectory
        cf = self._sample_cf() if self.cf_ratio > 0 and random.random() < self.cf_ratio else None

        # Unpack scene path and its corresponding sensor_data_path
        scene_path, sensor_data_path = self.scene_by_token[cf["token"]] if cf else self.scenes[idx]
        with open(scene_path, 'r') as f:
            scene_data = json.load(f)
        if cf:
            scene_data['gt_trajectory'] = cf["traj"]

        for builder in self._agent.get_feature_builders():
            input_features.update(builder.compute_features(scene_data))
        for builder in self._agent.get_target_builders():
            target_trajectory.update(builder.compute_targets(scene_data))
        
        # Override sensor_data_path with the correct one for this scene
        input_features['sensor_data_path'] = sensor_data_path
        
        # image sensor
        images = input_features['images']
        camera_images = {}
        
        # List of camera types to load
        camera_types = ['front_camera', 'front_left_camera', 'front_right_camera']
        
        # When sensor_data_path is set, image paths in the json are relative and
        # need to be prefixed with it. When it is null/empty (e.g. nuScenes stores
        # full paths during preprocessing), use the stored path as-is.
        for camera_type in camera_types:
            camera_images[camera_type] = []
            for i in range(4):
                img = images[camera_type][i]
                if input_features['sensor_data_path']:
                    camera_images[camera_type].append(
                        os.path.join(input_features['sensor_data_path'], img))
                else:
                    camera_images[camera_type].append(img)
        # 4 帧都留着，取最后 self.num_frames 帧在 prompt_spec.camera_content 里做

        # vehicle state
        velocity = input_features["vehicle_velocity"]
        if isinstance(velocity, list) or isinstance(velocity, np.ndarray):
            velocity_x = velocity[0]
            velocity_y = velocity[1]
            velocity = np.sqrt(velocity_x**2 + velocity_y**2)
            
        acceleration = input_features["vehicle_acceleration"]
        if isinstance(acceleration, list) or isinstance(acceleration, np.ndarray):
            acceleration_x = acceleration[0]
            acceleration_y = acceleration[1]
            acceleration = np.sqrt(acceleration_x**2 + acceleration_y**2)

        instruction = input_features["driving_command"].lower()

        # trajectory
        gt_action_idx = target_trajectory['gt_idx']
        gt_raw_trajectory = target_trajectory['gt_pos_raw']


        # get assistent content for different datasets
        gt_cot = scene_data['cot_output']

        
        if not self.using_cot:
            assistant_content = [
                {
                    "type": "text",
                    "text": (
                        "<answer>\n"
                        "The final output action is: " + self.action_tokenizer(gt_action_idx[0]) + "\n"
                        "</answer>"
                    )
                }
            ]
            # prompt 文案统一放在 dataset_utils/prompt_spec.py（RL 侧共用同一份）
            system_content = [{"type": "text", "text": prompt_spec.system_text(False)}]

            has_cot = False
            cot_mode = 'act_directly'
        else:
            system_content = [{"type": "text", "text": prompt_spec.system_text(True)}]

            # Otherwise, check dataset type and CoT availability
            if cf:
                # cf 样本：没有 <think>、没有决策句，只有 <PLAN> 和动作（docs/1026/cf_sft_autovla_plan.md §2）。
                # 解耦样本的文字由 DataCollator 按 skip_text_loss 屏蔽，只训 10 个 action token。
                cot_mode = "counterfactual"
                has_cot = False
                assistant_content = [
                    {
                        "type": "text",
                        "text": (
                            f"<PLAN>{cf['plan']}</PLAN>\n"
                            "<answer>\n"
                            "The final output action is: " + self.action_tokenizer(gt_action_idx[0]) + "\n"
                            "</answer>"
                        )
                    }
                ]
            elif scene_data["dataset_name"] == "nuplan" or scene_data["dataset_name"] == "waymo" :
                # simlingo 式:每条样本随机选模式，prompt 侧同步区分（见 __init__ 的说明）
                _avail = bool(isinstance(gt_cot, str) and gt_cot)
                _use = _avail and random.random() < self.cot_ratio
                cot_mode = ("reasoning_given"
                            if _use and random.random() < self.cot_in_prompt_ratio
                            else "reason_then_act" if _use else "act_directly")
                has_cot = (cot_mode == "reason_then_act")
                if has_cot:
                    assistant_content = [
                        {
                            "type": "text",
                            "text": (
                                "<think>\n"
                                f"{gt_cot}\n"
                                "</think>\n"
                                "<answer>\n"
                                "The final output action is: " + self.action_tokenizer(gt_action_idx[0]) + "\n"
                                "</answer>"
                            )
                        }
                    ]
                else:
                    # 直接给动作，【不写空的 <think>】——写了等于教模型"想一下但什么都不说"
                    assistant_content = [
                        {
                            "type": "text",
                            "text": (
                                "<answer>\n"
                                "The final output action is: " + self.action_tokenizer(gt_action_idx[0]) + "\n"
                                "</answer>"
                            )
                        }
                    ]
            elif scene_data['dataset_name'] == "nuscenes":
                has_cot = False
                cot_mode = "act_directly"
                if len(gt_cot) == 5:  # If CoT is available
                    if gt_cot[4] == "STOP\n":
                        gt_cot[4] = "stop"
                    assistant_content = [
                        {
                            "type": "text",
                            "text":
                                "<think>\n"
                                "This is a complex scenario requiring additional reasoning.\n" 
                                f"### Scene Description:\n{gt_cot[0]}\n\n" 
                                f"### Critical Object Description:\n{gt_cot[1] + gt_cot[2]}\n\n" 
                                f"### Reasoning on Intent:\n{gt_cot[3]}\n\n"
                                f"### Best Driving Action:\n{gt_cot[4]}\n" 
                                "</think>\n"
                                "<answer>\n"
                                "The final output action is: " + self.action_tokenizer(gt_action_idx[0]) + "\n"
                                "</answer>"
                        }
                    ]
                    has_cot = True
                else:  # Only return the final action
                    assistant_content = [
                        {
                            "type": "text",
                            "text": (
                                "<think>\n"
                                "This is a straightforward scenario, and a direct decision can be made.\n"
                                "</think>\n"
                                "<answer>\n"
                                "The final output action is: " + self.action_tokenizer(gt_action_idx[0]) + "\n"
                                "</answer>"
                            )
                        }
                    ]
            else:
                print(scene_data['dataset_name'])
                exit()


        # 相机部分（介绍 + 3 路 video）统一由 prompt_spec 拼，推理侧 models/autovla.py::get_prompt 用同一个函数。
        # min/max_pixels 读 config 的 model.video（见 __init__，没写时仍是 28*28*128，与改动前一致）。
        user_content = prompt_spec.camera_content(
            camera_images, self.num_frames, min_pixels=self.min_pixels, max_pixels=self.max_pixels) + [
            {
                "type": "text",
                "text": prompt_spec.user_text(
                    velocity, acceleration, instruction, self.using_cot, cot_mode,
                    gt_cot if isinstance(gt_cot, str) else "",
                    **({"report": cf["report"], "suggestion": cf["suggestion"]} if cf else {})),
            },
        ]

        # create messages
        messages = [
            {   
                "role": "system",
                "content": system_content
            },

            {
                "role": "user",
                "content": user_content
            },

            # assistant response
            {
                "role": "assistant",
                "content": assistant_content
            }
        ]


        # ── 调试:PRINT_SFT_DEBUG=N 时，每个 dataloader worker 打印前 N 条样本 ──
        # 用来肉眼确认三档采样、prompt 后缀、target 里到底喂了什么。
        # 训练时 num_workers>0，每个 worker 各打 N 条，所以总量 = N x num_workers。
        # ⚠️ 只打印，不改任何返回值；N 设小一点（1~3），否则 log 会被刷爆。
        _dbg = int(os.environ.get("PRINT_SFT_DEBUG", "0"))
        if _dbg and getattr(self, "_dbg_printed", 0) < _dbg:
            self._dbg_printed = getattr(self, "_dbg_printed", 0) + 1
            _w = torch.utils.data.get_worker_info()
            print(
                f"\n[SFT_DEBUG] worker={_w.id if _w else 0} "
                f"#{self._dbg_printed}/{_dbg}  mode={cot_mode}  has_cot={has_cot}  "
                f"token={scene_data.get('token')}"
                + (f"  cf={cf['cls']} decouple={cf['decouple']}" if cf else "") + "\n"
                f"  SYSTEM : {system_content[0]['text']}\n"
                f"  USER   : {user_content[-1]['text']}\n"
                f"  TARGET : {assistant_content[0]['text']}\n"
                f"  ↑ TARGET 是 teacher forcing 的标签:reasoning 用的是 GT，"
                f"不是模型自己生成的\n",
                flush=True,
            )

        # process the images and messages
        image_inputs, video_inputs = process_vision_info(messages)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
        )
    
        inputs = {'text': text, 'image_inputs': image_inputs, 'video_inputs': video_inputs}

        # trajectory information
        inputs['gt_trajectory'] = gt_raw_trajectory
        inputs['gt_action'] = gt_action_idx
        inputs['has_cot'] = has_cot
        inputs['data_path'] = scene_path
        # 解耦 cf 样本只训 action token（DataCollator 读这个，不放进 batch）
        inputs['skip_text_loss'] = bool(cf and cf["decouple"])
        inputs['cf'] = cf
        # Force garbage collection to free temporary objects
        # gc.collect()

  
        return inputs


@dataclass
class DataCollator:
    processor: AutoProcessor
    ignore_index: int = -100
    assistant_id: Optional[List[int]] = None
    action_start_id: int = 151665
    
    def __post_init__(self):
        if self.assistant_id is None:
            self.assistant_id = [151644, 77091]  # default value for Qwen2.5-VL

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract text inputs
        text = [batch["text"] for batch in features]
        
        # Process video and image inputs
        video_inputs = []
        image_inputs = []
        has_cot = []
        for batch in features:
            video_inputs.extend(batch["video_inputs"])
            image_inputs.append(batch["image_inputs"])
            has_cot.append(batch["has_cot"])
        
        
        batch = self.processor(
            text=text,
            images=image_inputs if image_inputs[0] is not None else None,
            videos=video_inputs if video_inputs[0] is not None else None,
            padding=True,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()

        # Find the start of the assistant response
        assistant_id = torch.tensor(self.assistant_id)
        for i in range(labels.shape[0]):
            for j in range(len(labels[i]) - len(assistant_id) + 1):
                if torch.equal(labels[i][j:j + len(assistant_id)], assistant_id):
                    start_idx = j
                    break
            
            # [CRITICAL] We take the losses on the assistant response only
            labels[i, :start_idx] = self.ignore_index

            # 解耦 cf 样本（suggestion 被换成别的动作）：文字一律不算 loss，只留 10 个 action token。
            # 文字上模型只学照做；轨迹上同一个 suggestion 配不同答案，逼它跟着 <PLAN> 走。
            if features[i].get('skip_text_loss'):
                labels[i][labels[i] < self.action_start_id] = self.ignore_index

        # add labels and gt action to the batch
        batch['labels'] = labels
        batch['gt_trajectory'] = torch.stack([batch['gt_trajectory'] for batch in features])
        batch['gt_action'] = torch.stack([batch['gt_action'] for batch in features])
        batch['has_cot'] = torch.tensor(has_cot)

        return batch


if __name__ == "__main__":
    # C6 smoke test（docs/1026/cf_sft_autovla_plan.md §4 C6）：
    #   python -m dataset_utils.sft_dataset --config training/1026/<name> [--n 200] [--split train]
    # 检查 cf 样本的 prompt / 答案 / 轨迹 / 解耦，以及 DataCollator 对解耦样本的 label 屏蔽；打印各类占比。
    import argparse
    import re
    from collections import Counter

    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    config = yaml.safe_load(open(f"config/{args.config}.yaml"))
    processor = AutoProcessor.from_pretrained(config['model']['pretrained_model_path'], use_fast=True)
    ds = SFTDataset(config['data'][args.split], config['model'], processor, using_cot=config['model']['use_cot'])
    collator = DataCollator(processor=processor, ignore_index=config['model']['tokens']['ignore_index'],
                            assistant_id=config['model']['tokens']['assistant_id'],
                            action_start_id=config['model']['tokens']['action_start_id'])
    allowed = {c: {prompt_spec.CF_REPORTS[k] for k in v} for c, v in prompt_spec.CF_ACTION_REPORTS.items()}
    tail_re = re.compile(r"Scene report: (.+?) Suggested action: (.+?)\.<\|im_end\|>")
    plan_re = re.compile(r"<\|im_start\|>assistant\n<PLAN>([A-Z_]+),([A-Z_]+)</PLAN>\n<answer>\n")
    random.seed(args.seed)
    stat, errors, shown = Counter(), [], Counter()
    for k in range(args.n):
        item = ds[random.randrange(len(ds))]
        cf, text = item['cf'], item['text']
        if cf is None:
            stat['normal'] += 1
            if "Scene report:" in text or "Suggested action:" in text:
                errors.append(("normal sample has cf text", item['data_path']))
            continue
        stat['cf'] += 1
        stat[f"cls={cf['cls']}{' (decouple)' if cf['decouple'] else ''}"] += 1
        stat[f"report={cf['report']}"] += 1
        m, p = tail_re.search(text), plan_re.search(text)
        own = prompt_spec.CF_SUGGESTIONS.get(cf['cls'])
        checks = {
            "prompt 结尾是 report + suggestion，后面没有别的指令": m is not None,
            "report 和答案的动作相符": m is not None and m.group(1) in allowed[cf['cls']],
            "suggestion 在 5 个里": m is not None and m.group(2) in prompt_spec.CF_SUGGESTIONS.values(),
            "照做 ⇔ suggestion 等于本类": m is not None and ((m.group(2) == own) != cf['decouple']),
            "KEEP 只当解耦样本": cf['cls'] != 'KEEP' or cf['decouple'],
            "skip_text_loss ⇔ 解耦": item['skip_text_loss'] == cf['decouple'],
            "答案是 <PLAN> + <answer>，没有 <think>": p is not None and "<think>" not in text.split("assistant\n", 1)[-1],
            "<PLAN> 在词表里": p is not None and p.group(1) in prompt_spec.PLAN_SPEED_WORDS
                               and p.group(2) in prompt_spec.PLAN_PATH_WORDS,
            "<PLAN> 和类别一致": p is not None and (p.group(2) == cf['cls'] if cf['cls'].startswith('LANE')
                                                 else p.group(1) == cf['cls']),
            "gt 轨迹就是 cf 轨迹": np.allclose(np.asarray(item['gt_trajectory'])[:, :2],
                                          ds.cf_traj[cf['row']][:, :2], atol=1e-4),
        }
        for name, ok in checks.items():
            if not ok:
                errors.append((name, cf['token'], cf['cls']))
        # collator：解耦样本只剩 action token 有 label；照做样本 <PLAN> 也有 label
        lab = collator([item])['labels'][0]
        kept = lab[lab != -100]
        only_action = bool((kept >= collator.action_start_id).all())
        if only_action != cf['decouple'] or (cf['decouple'] and len(kept) != 10):
            errors.append(("collator label 屏蔽不对", cf['token'], cf['decouple'], len(kept)))
        key = "decouple" if cf['decouple'] else cf['cls']
        if shown[key] < 1:
            shown[key] += 1
            # user 段的最后一句；答案取第一个 assistant 段（chat template 末尾还会再接一个空的 assistant 开头）
            print(f"\n── {key}  token={cf['token']}"
                  f"\n  USER  : …{text.split('vehicle over the next five seconds.')[1].split('<|im_end|>')[0]}"
                  f"\n  TARGET: {text.split('<|im_start|>assistant' + chr(10))[1].split('<|im_end|>')[0][:110]}…"
                  f"\n  label 里非 -100 的 token 数 {len(kept)}（只剩 action: {only_action}）")
    print(f"\n{args.n} 条：normal {stat['normal']}  cf {stat['cf']}（{100 * stat['cf'] / args.n:.1f}%，config {ds.cf_ratio}）")
    for k, v in sorted(stat.items()):
        if k.startswith(("cls=", "report=")):
            print(f"  {k:60s} {v}")
    print(f"\n错误 {len(errors)} 条" + ("" if not errors else ":"))
    for e in errors[:20]:
        print("  ", e)
    sys_exit = 1 if errors else 0
    raise SystemExit(sys_exit)