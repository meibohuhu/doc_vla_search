"""第 0 步 go/no-go 的【需要模型】那一半:在学生自己的 rollout 上量触发率。

    P(学生轨迹不可行)              触发器点火率
    P(说法也不在 D 里 | 不可行)     才是 chg —— 只有这部分会真的产生代价
    组内 PDMS 跨度                  GRPO 的 advantage 有没有信号

    /data/autovla_data/envs/autovla/bin/python logs/0902/step0_rollout.py \
        --ckpt /data/autovla_data/checkpoints/sft/2026-09-02_05-47-39/epoch=1-loss=0.5147.ckpt \
        --n 600 --G 5 --shard 0 --of 3 --out /tmp/step0_r0.jsonl

⚠️ 用的是**训练中途**的 CoT-SFT 权重。ckpt 文件名里的 epoch= 是 0-indexed，
   epoch=1 是【第 2 个】epoch。数会随 SFT 收敛而变，
   这一版只用来回答"值不值得做"，不能当最终数字写进论文。
"""
import argparse, json, os, sys, time
sys.path.insert(0, '.'); sys.path.insert(0, './navsim')
os.environ.setdefault("NUPLAN_MAPS_ROOT", "/data/autovla_data/nuplan/maps")
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
os.environ.setdefault("OPENSCENE_DATA_ROOT", "/data/autovla_data/nuplan")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import yaml
from pathlib import Path
from navsim.common.dataclasses import Trajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from models.autovla import GRPOAutoVLA
from models.utils import rl_rewards
from models.utils.feasible import FeasibleSet, PLAN_OF_PROFILE
from dataset_utils.rft_dataset import RFTDataset

PROF_OF_PLAN = {v: k for k, v in PLAN_OF_PROFILE.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--G", type=int, default=5)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", default=None,
                    help="只跑这个文件里列的 token（每行一个）。给可视化挑样本用。")
    ap.add_argument("--poses", action="store_true",
                    help="把预测轨迹的 10 个位姿一起存进 jsonl（画 BEV 需要，默认不存以免文件太大）")
    ap.add_argument("--temp", type=float, default=None,
                    help="覆盖采样温度。默认用 config 的 training.sample.temperature(0.9,=RL rollout)。"
                         "设 0.01 = 贪心，用来对比【采样探索】和【模型本身】各贡献了多少不可行")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cfg['model']['lora']['use'] = False        # SFT ckpt 是全参的，别再套 LoRA
    cfg['rl']['group']['per_prompt_G'] = args.G
    if args.temp is not None:
        cfg['training']['sample']['temperature'] = args.temp
    dev = "cuda"

    model = GRPOAutoVLA(cfg, inference=True)
    sd = torch.load(args.ckpt, map_location="cpu")["state_dict"]
    msg = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}", flush=True)
    model = model.to(torch.bfloat16).to(dev).eval()

    ds = RFTDataset(cfg['data']['val'], cfg['model'])
    rng = np.random.RandomState(23)
    if args.tokens:
        want = {l.strip() for l in open(args.tokens) if l.strip()}
        idx = [i for i in range(len(ds)) if ds.scenes[i].stem in want]
    else:
        idx = rng.permutation(len(ds))[: args.n]
    idx = idx[args.shard::args.of]
    print(f"[shard {args.shard}/{args.of}] {len(idx)} 帧", flush=True)

    fs = FeasibleSet(Path(cfg['data']['val']['metric_cache_path']))
    samp = TrajectorySampling(num_poses=10, interval_length=0.5)
    fout = open(args.out, "w")
    t0 = time.time()
    for c, i in enumerate(idx, 1):
        item = ds[int(i)]
        item['input_features'].setdefault('sensor_data_path', None)
        tok = item['token']
        try:
            smp = model.generate_sample({'input_features': item['input_features'], 'token': tok},
                                        model=model.autovla, device=dev)
        except Exception as e:
            print(f"  [skip] {tok} generate: {e}", flush=True); continue

        rec = {"token": tok, "rollouts": []}
        for g, (text, poses) in enumerate(zip(smp['completion_texts'], smp['trajectory_poses'])):
            m = fs.score(tok, Trajectory(np.asarray(poses, dtype=np.float32), samp))
            if m is None:
                continue
            feas = FeasibleSet.is_feasible(m)
            think = rl_rewards._THINK_RE.search(text)
            plan = rl_rewards.parse_plan(think.group(1) if think else text)
            D = fs.feasible_set(tok, np.asarray(poses), float(np.hypot(*item['input_features']['vehicle_velocity'][:2])))
            rec["rollouts"].append({
                "pdms": m["score"], "nc": m["nc"], "dac": m["dac"], "ep": m["ep"],
                "feasible": bool(feas),
                "plan": None if plan is None else list(plan),
                "fmt_viol": bool(rl_rewards.format_violation(text, expect_think=True)),
                "consistency": rl_rewards.reward_consistency(text, poses),
                "D": D["D"], "uncheckable": D["uncheckable"],
                "say_in_D": (None if plan is None else PROF_OF_PLAN.get(plan[0]) in D["D"]),
                "text": text[:400],
                **({"poses": np.asarray(poses)[:, :3].tolist()} if args.poses else {}),
            })
        fout.write(json.dumps(rec) + "\n"); fout.flush()
        if c % 20 == 0:
            print(f"  {c}/{len(idx)}  {time.time()-t0:.0f}s  ({(time.time()-t0)/c:.1f}s/帧)", flush=True)
    fout.close()
    print(f"done {len(idx)} 帧 {time.time()-t0:.0f}s → {args.out}", flush=True)


if __name__ == "__main__":
    main()
