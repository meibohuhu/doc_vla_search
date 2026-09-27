"""
NuScenes Evaluation Script for AutoVLA.

This script evaluates the AutoVLA model on NuScenes validation data,
computing planning metrics such as L2 distance and collision rate.
"""
import argparse
import json
import sys
import time
from pathlib import Path

# Add project root and navsim to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "navsim"))

import yaml
import torch
import numpy as np
from tqdm import tqdm
from prettytable import PrettyTable
from transformers import AutoProcessor

from dataset_utils.sft_dataset import SFTDataset
from models.autovla import SFTAutoVLA
from tools.eval.planning_metrics import PlanningMetric



def load_config(file_path):
    """Load configuration from YAML file."""
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate AutoVLA on NuScenes validation data")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint")
    parser.add_argument("--seg_data_path", type=str, required=True,
                        help="Path to the segmentation data directory for evaluation")
    parser.add_argument("--output", type=str, default="planning_table.txt",
                        help="Output file for results (default: planning_table.txt)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use (default: cuda:0)")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to evaluate (default: all)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print verbose output for each sample")
    # --- 分片评测（多 job 并行）：第 shard_id 片处理 idx % num_shards == shard_id 的样本 ---
    parser.add_argument("--num_shards", type=int, default=1,
                        help="把 val 集切成几片并行（默认 1=不分片）")
    parser.add_argument("--shard_id", type=int, default=0,
                        help="本 job 跑第几片 [0, num_shards)")
    parser.add_argument("--dump_raw", type=str, default=None,
                        help="把原始累加和(obj_col/obj_box_col/L2/total)存到此 .pt，供多片合并")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize processor
    processor = AutoProcessor.from_pretrained(config['model']['pretrained_model_path'], use_fast=True)
    
    # Build data config for SFTDataset from config
    
    train_dataset = SFTDataset(config['data']['val'], config['model'], processor)

    # Load model
    checkpoint_path = Path(args.checkpoint)
    model = SFTAutoVLA(config)
    model.autovla.vlm.resize_token_embeddings(len(processor.tokenizer))
    
    state_dict = torch.load(checkpoint_path, map_location=args.device)['state_dict']
    # ⚠️ Lightning ckpt 的 key 带 "autovla." 前缀（如 autovla.vlm.visual...），
    # 而这里 load 进的是 model.autovla（子模块，key 应为 vlm.visual...）。
    # 不 strip 前缀 → strict=False 下【全部 key 静默对不上】→ 加载随机权重跑出垃圾指标。
    # 对齐 NAVSIM AutoVLAAgent(autovla_agent.py:391) 的做法：先去掉 autovla. 前缀。
    stripped = {k.replace("autovla.", "", 1): v for k, v in state_dict.items()}
    missing, unexpected = model.autovla.load_state_dict(stripped, strict=False)
    # 防静默失败：真正加载进去的 key 必须占绝大多数（ckpt 是全参，825 key）。
    loaded = len(stripped) - len(unexpected)
    print(f"[ckpt] loaded={loaded}/{len(stripped)}  missing={len(missing)}  unexpected={len(unexpected)}")
    assert loaded > 0.9 * len(stripped), (
        f"ckpt 加载异常：只匹配上 {loaded}/{len(stripped)} 个 key，"
        f"极可能是 key 前缀不对（会静默跑垃圾分数）。unexpected 样例: {unexpected[:3]}"
    )
    # 断言 action token 的 embedding 确实被加载（这 2048 行是轨迹能力的关键）
    assert not any("embed_tokens" in k or "lm_head" in k for k in unexpected), \
        "embed_tokens/lm_head 没加载进去——action token 会是随机的"

    model.to(args.device)
    model.autovla.device = args.device  # Update the device attribute for predict()
    model.eval()

    # Initialize planning metrics
    planning_metrics = PlanningMetric(n_future=6)
    
    # Determine number of samples to evaluate
    sample_num = len(train_dataset.scenes)
    if args.num_samples is not None:
        sample_num = min(args.num_samples, sample_num)

    # 分片：本 job 只处理 idx % num_shards == shard_id 的样本。
    # 用取模而非连续切块，保证各片难度分布均匀（相邻样本常来自同一 scene）。
    assert 0 <= args.shard_id < args.num_shards, \
        f"shard_id={args.shard_id} 必须在 [0,{args.num_shards})"
    indices = [i for i in range(sample_num) if i % args.num_shards == args.shard_id]
    print(f"Evaluating {len(indices)} samples "
          f"(shard {args.shard_id}/{args.num_shards} of {sample_num} total)...")

    # Evaluate each sample
    for idx in tqdm(indices, desc=f"Processing samples [shard {args.shard_id}/{args.num_shards}]"):
        # scenes is a list of tuples: (scene_path, sensor_data_path)
        scene_path, _ = train_dataset.scenes[idx]
        
        # Load scene data
        with open(scene_path, 'r') as f:
            scene_data = json.load(f)
        
        # Get features and targets
        input_features = {}
        target_trajectory = {}
        
        start_time = time.time()

        for builder in train_dataset._agent.get_feature_builders():
            input_features.update(builder.compute_features(scene_data))
        for builder in train_dataset._agent.get_target_builders():
            target_trajectory.update(builder.compute_targets(scene_data))

        # Model prediction
        pred_trajectory, output_text = model.autovla.predict(input_features)
        if pred_trajectory == [] or len(pred_trajectory) == 0:
            continue
        
        if args.verbose:
            print(f"Output: {output_text}")
            print(f"Predicted trajectory: {pred_trajectory}")
            print(f"Time taken: {time.time() - start_time:.3f} seconds")
       
        # Get ground truth trajectory
        gt_raw_trajectory = target_trajectory["gt_pos_raw"]
        pred_xy = pred_trajectory[:, :2].to(gt_raw_trajectory.device)

        # Load segmentation data for collision evaluation
        seg_path = Path(args.seg_data_path) / f"{scene_data['token']}.pt"
        if not seg_path.exists():
            print(f"Warning: Segmentation file not found: {seg_path}")
            continue
            
        uniad_data = torch.load(seg_path, map_location="cpu")
        sdc_planning_mask = uniad_data['sdc_planning_mask'].to(gt_raw_trajectory.dtype)
        segmentation = uniad_data['segmentation'].to(gt_raw_trajectory.dtype)

        # Transform GT trajectory to UniAD coordinate system
        gt_traj_uniadcoord = gt_raw_trajectory.unsqueeze(0).clone()
        gt_traj_uniadcoord[:, :, [0, 1]] = gt_traj_uniadcoord[:, :, [1, 0]]
        gt_traj_uniadcoord[:, :, 0] = -gt_traj_uniadcoord[:, :, 0]
        gt_traj_uniadcoord = gt_traj_uniadcoord.unsqueeze(0)

        # Transform predicted trajectory to UniAD coordinate system
        pred_traj_uniadcoord = pred_xy.unsqueeze(0).clone()
        pred_traj_uniadcoord[:, :, [0, 1]] = pred_traj_uniadcoord[:, :, [1, 0]]
        pred_traj_uniadcoord[:, :, 0] = -pred_traj_uniadcoord[:, :, 0]
        pred_traj_uniadcoord = pred_traj_uniadcoord.unsqueeze(0)

        # Validate future mask consistency
        cache_future_mask = torch.tensor(scene_data['future_mask'][:6])
        sdc_mask = sdc_planning_mask[0, 0, :, 0]
        if not torch.allclose(cache_future_mask, sdc_mask):
            print(f"Warning: Mismatch between mask values for sample {idx}")
            continue

        # Compute planning metrics
        planning_metrics(
            pred_traj_uniadcoord[0, :, :6, :], 
            gt_traj_uniadcoord[0, :, :6, :], 
            sdc_planning_mask[0, :, :6, :2], 
            segmentation[:, [1, 2, 3, 4, 5, 6]]
        )
    
    # 分片模式：把【原始累加和】存出去，供合并（不能存平均后的表——平均没法再合并）。
    # PlanningMetric 的状态 obj_col/obj_box_col/L2 是逐 timestep 的求和，total 是样本计数。
    if args.dump_raw is not None:
        raw = {
            "obj_col": planning_metrics.obj_col.detach().cpu(),
            "obj_box_col": planning_metrics.obj_box_col.detach().cpu(),
            "L2": planning_metrics.L2.detach().cpu(),
            "total": planning_metrics.total.detach().cpu(),
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
            "checkpoint": str(args.checkpoint),
        }
        Path(args.dump_raw).parent.mkdir(parents=True, exist_ok=True)
        torch.save(raw, args.dump_raw)
        print(f"[dump] 原始累加和已存: {args.dump_raw}  (total={int(planning_metrics.total)})")

    # Calculate and print overall statistics
    eval_result = planning_metrics.compute()

    # Create table with STP3's definition (cumulative average)
    planning_tab_stp3 = PrettyTable()
    planning_tab_stp3.title = "STP3's Definition Planning Metrics (Cumulative Average)"
    planning_tab_stp3.field_names = ["metrics", "0.5s", "1.0s", "1.5s", "2.0s", "2.5s", "3.0s"]
    
    for key, value in eval_result.items():
        row_value = [key]
        for i in range(min(len(value), 6)):
            row_value.append("%.4f" % float(value[:i + 1].mean()))
        planning_tab_stp3.add_row(row_value)
    print(planning_tab_stp3)

    # Create table with UniAD's definition (per-timestep)
    planning_tab_uniad = PrettyTable()
    planning_tab_uniad.title = "UniAD's Definition Planning Metrics (Per-Timestep)"
    planning_tab_uniad.field_names = ["metrics", "0.5s", "1.0s", "1.5s", "2.0s", "2.5s", "3.0s"]
    
    for key, value in eval_result.items():
        row_value = [key]
        for i in range(min(len(value), 6)):
            row_value.append("%.4f" % float(value[i]))
        planning_tab_uniad.add_row(row_value)
    print(planning_tab_uniad)

    # Save results to file
    with open(args.output, 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Evaluation Results - {sample_num} samples\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"{'='*60}\n\n")
        f.write(str(planning_tab_stp3) + "\n\n")
        f.write(str(planning_tab_uniad) + "\n")
    
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
