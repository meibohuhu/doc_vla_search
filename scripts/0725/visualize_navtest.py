#!/usr/bin/env python3
"""
把 navtest 评测结果可视化：每个场景一张图 = 8 路相机 + BEV（预测轨迹 vs 人类 GT）+ 分数。

用途：肉眼看模型到底怎么开的，尤其归零/低分场景到底错在哪
（冲出边界？转向错误？开太慢？）——比只看 PDMS 数字直观得多。

复用 navsim 自带的 plot_cameras_frame_with_bev_agent_cot（评测脚本里被注释掉的那个），
但因为 no-CoT agent 是 requires_scene=False、只吃 JSON，scene 没被加载，
所以这里单独用 SceneLoader 重新加载 scene 来画图。

用法:
    # 指定 token（逗号分隔）
    python scripts/0725/visualize_navtest.py --ckpt <ckpt> --tokens t1,t2,t3

    # 从某次评测结果里挑 PDMS 最低的 N 个
    python scripts/0725/visualize_navtest.py --ckpt <ckpt> \
        --from_csv /data/autovla_data/nuplan/sft_eval/epoch_4-loss_0_6966_merged.csv \
        --worst 20

    # 随机 N 个
    python scripts/0725/visualize_navtest.py --ckpt <ckpt> --from_csv <csv> --sample 20
"""
import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "navsim"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

DATA = "/data/autovla_data/nuplan"
os.environ.setdefault("NUPLAN_MAPS_ROOT", f"{DATA}/maps")
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
os.environ.setdefault("OPENSCENE_DATA_ROOT", DATA)
os.environ.setdefault("NAVSIM_DEVKIT_ROOT", str(REPO / "navsim"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from omegaconf import OmegaConf
from hydra.utils import instantiate
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.agents.autovla_agent import AutoVLAAgent
from navsim.visualization.plots import plot_cameras_frame_with_bev_agent_cot
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

NAVTEST = REPO / "navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"


def pick_tokens(args):
    if args.tokens:
        return [t.strip() for t in args.tokens.split(",") if t.strip()]
    df = pd.read_csv(args.from_csv)
    df = df[df.token.notna()]
    if args.worst:
        df = df.sort_values("score").head(args.worst)
    elif args.sample:
        df = df.sample(min(args.sample, len(df)), random_state=42)
    return df.token.tolist(), df.set_index("token")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval_config",
                    default=str(REPO / "config/eval/0723/autovla-navtest-eval-nocot.yaml"))
    ap.add_argument("--tokens", default=None, help="逗号分隔的 token；优先于 --from_csv")
    ap.add_argument("--from_csv", default=None, help="评测结果 merged.csv，用于挑 token + 标注分数")
    ap.add_argument("--worst", type=int, default=0, help="从 csv 挑 PDMS 最低的 N 个")
    ap.add_argument("--sample", type=int, default=0, help="从 csv 随机挑 N 个")
    ap.add_argument("--out", default="/data/autovla_data/nuplan/viz")
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    scores = None
    if args.tokens:
        tokens = [t.strip() for t in args.tokens.split(",") if t.strip()]
    else:
        assert args.from_csv, "需要 --tokens 或 --from_csv"
        tokens, scores = pick_tokens(args)
    os.makedirs(args.out, exist_ok=True)
    print(f"要可视化 {len(tokens)} 个场景 -> {args.out}")

    # 只加载这些 token 对应的 scene（scene_filter 按 token 过滤）
    sf: SceneFilter = instantiate(OmegaConf.load(NAVTEST))
    sf.tokens = tokens
    loader = SceneLoader(
        sensor_blobs_path=Path(f"{DATA}/sensor_blobs/test"),
        data_path=Path(f"{DATA}/navsim_logs/test"),
        scene_filter=sf,
        # 画图要 8 路相机，但【不要 lidar】——我们下载时没下 lidar（lidar_pc=False 一路贯穿），
        # build_all_sensors() 会开 lidar 导致 FileNotFoundError(.pcd)。手动只开 8 相机。
        sensor_config=SensorConfig(
            cam_f0=True, cam_l0=True, cam_l1=True, cam_l2=True,
            cam_r0=True, cam_r1=True, cam_r2=True, cam_b0=True,
            lidar_pc=False,
        ),
    )
    print(f"SceneLoader 命中 {len(loader.tokens)} / {len(tokens)} 个 token")

    # agent（加载模型，用来实际预测轨迹）
    cfg = yaml.safe_load(open(args.eval_config))
    agent = AutoVLAAgent(
        trajectory_sampling=TrajectorySampling(time_horizon=5, interval_length=0.5),
        checkpoint_path=args.ckpt,
        sensor_data_path=f"{DATA}/sensor_blobs/test",
        config_path=args.eval_config,
        lora_conf={"use_lora": False},
    )
    agent.initialize()

    import json as _json
    for i, token in enumerate(loader.tokens):
        scene = loader.get_scene_from_token(token)   # 只用来画 BEV/相机
        # 预测走跟评测【完全一样】的路径：读 navtest_nocot 的 JSON 喂 feature builder
        scene_data = _json.load(open(f"{DATA}/navtest_nocot/{token}.json"))
        feats = {}
        for b in agent.get_feature_builders():
            feats.update(b.compute_features(scene_data))
        feats["sensor_data_path"] = f"{DATA}/sensor_blobs/test"
        with torch.no_grad():
            traj, cot = agent.autovla.predict(feats)
        from navsim.common.dataclasses import Trajectory
        traj_obj = Trajectory(traj[:10, :], agent._trajectory_sampling)

        frame_idx = scene.scene_metadata.num_history_frames - 1
        cap = cot or ""
        if scores is not None and token in scores.index:
            r = scores.loc[token]
            cap = (f"PDMS={r.score*100:.1f}  NC={r.no_at_fault_collisions:.0f} "
                   f"DAC={r.drivable_area_compliance:.0f} EP={r.ego_progress*100:.0f} "
                   f"TTC={r.time_to_collision_within_bound:.0f}\n") + cap
        fig, _ = plot_cameras_frame_with_bev_agent_cot(scene, frame_idx, traj_obj, cot=cap)
        out = Path(args.out) / f"{token}.png"
        fig.savefig(out, bbox_inches="tight", dpi=100)
        plt.close(fig)
        print(f"  [{i+1}/{len(loader.tokens)}] {token} -> {out.name}")

    print(f"\n完成。图片在 {args.out}")


if __name__ == "__main__":
    main()
