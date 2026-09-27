"""action tokenizer 的天花板 —— GT 轨迹量化再还原之后，还能不能通过 DAC？

为什么要测:navtest 上 69.5% 的失效是横向(DAC=0)，而且全是逐渐漂出去的。
"模型方向学得不好"是一个解释，但还有一个更基础的:**轨迹是量化成 codebook token
输出的**，如果 codebook 本身表达不了那条路的横向精度，天花板就不在模型。

做法:把 GT 轨迹过一遍 tokenize → detokenize（和训练时给模型的监督目标完全一致），
      再送 PDMScorer。**这是模型即使 100% 预测对 token 也只能拿到的分数。**
"""
import json, os, sys, time
sys.path.insert(0, '.'); sys.path.insert(0, './navsim')
os.environ.setdefault("NUPLAN_MAPS_ROOT", "/data/autovla_data/nuplan/maps")
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
os.environ.setdefault("OPENSCENE_DATA_ROOT", "/data/autovla_data/nuplan")
import numpy as np
import torch
import yaml
from pathlib import Path
from navsim.common.dataclasses import Trajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.agents.autovla_agent import AutoVLAAgent
from models.action_tokenizer import transform_to_global
from models.utils.feasible import FeasibleSet

N = int(sys.argv[1]) if len(sys.argv) > 1 else 600
cfg = yaml.safe_load(open("config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml"))
samp = TrajectorySampling(num_poses=10, interval_length=0.5)
agent = AutoVLAAgent(trajectory_sampling=samp, sensor_data_path=None,
                     codebook_cache_path=cfg['model']['codebook_cache_path'], skip_model_load=True)
tb = agent.get_target_builders()

# 直接用 codebook 做 rollout（等价于 ActionTokenizer.decode_token_ids_to_trajectory，
# 但不用为了解 "<action_N>" 这个字符串去加载整个 HF tokenizer）
import pickle
with open(cfg['model']['codebook_cache_path'], "rb") as f:
    CODE = torch.tensor(pickle.load(f)['token_all']['veh'])   # (n_bins, 6, 4, 2)


def rollout(idx):
    at = CODE[idx]
    pos = torch.tensor([[[0.0, 0.0]]]); head = torch.tensor([[0.0]])
    for t in range(at.shape[0]):
        nt = at[None, t]
        g = transform_to_global(pos_local=nt.flatten(1, 2), head_local=None,
                                pos_now=pos[:, t], head_now=head[:, t])[0].view(*nt.shape)
        pn = g[:, -1].mean(dim=1)
        dxy = g[:, -1, 0] - g[:, -1, 3]
        hn = torch.arctan2(dxy[:, 1], dxy[:, 0])
        pos = torch.cat([pos, pn.unsqueeze(1)], dim=1)
        head = torch.cat([head, hn.unsqueeze(1)], dim=1)
    return torch.cat([pos, head.unsqueeze(-1)], dim=-1)[0, 1:].numpy()
fs = FeasibleSet(Path("/data/autovla_data/nuplan/navtest_metric_cache"))

d = '/data/autovla_data/nuplan/navtest_cot'
names = sorted(os.listdir(d))
rng = np.random.RandomState(23)
names = [names[i] for i in rng.permutation(len(names))[:N]]

gt_m, q_m, lat = [], [], []
t0 = time.time()
for k, nm in enumerate(names, 1):
    tokn = os.path.splitext(nm)[0]
    s = json.load(open(os.path.join(d, nm)))
    gt = np.asarray(s['gt_trajectory'], dtype=np.float32)
    tt = {}
    for b in tb:
        tt.update(b.compute_targets(s))
    idx = torch.as_tensor(np.asarray(tt['gt_idx']).reshape(-1)).long()
    rec = np.asarray(rollout(idx), dtype=np.float32)
    a = fs.score(tokn, Trajectory(gt, samp))
    b_ = fs.score(tokn, Trajectory(rec, samp))
    if a and b_:
        gt_m.append(a); q_m.append(b_)
        lat.append(float(np.abs(rec[:, 1] - gt[:, 1]).max()))
    if k % 150 == 0:
        print(f"  {k}/{len(names)}  {time.time()-t0:.0f}s", flush=True)


def rep(name, M):
    print(f"  {name:26s} PDMS {np.mean([m['score'] for m in M]):.4f}   "
          f"NC=0 {100*np.mean([m['nc'] < 1 for m in M]):5.2f}%   "
          f"DAC=0 {100*np.mean([m['dac'] < 1 for m in M]):5.2f}%")


print(f"\nn = {len(gt_m)}")
rep("GT 原始轨迹", gt_m)
rep("GT 量化再还原", q_m)
lat = np.array(lat)
print(f"\n量化引入的最大横向误差:  p50={np.percentile(lat,50):.2f}  p90={np.percentile(lat,90):.2f}  "
      f"p99={np.percentile(lat,99):.2f}  max={lat.max():.2f} m")
bad = [i for i in range(len(q_m)) if q_m[i]['dac'] < 1 and gt_m[i]['dac'] >= 1]
print(f"GT 本来 DAC=1、量化后变成 DAC=0:  {len(bad)} / {len(q_m)}  ({100*len(bad)/len(q_m):.2f}%)")
print("  → 模型即使把 token 全预测对，也拿不到的那部分")
