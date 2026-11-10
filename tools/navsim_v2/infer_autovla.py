"""AutoVLA 对 build_inputs.py 生成的 inputs.jsonl 逐个 token 推理，输出轨迹和生成的文本。

不用 v2 的 PYTHONPATH（用环境里装的仓库版 navsim 和 models/）。单卡一片，可以中断后接着跑：
    CUDA_VISIBLE_DEVICES=0 python tools/navsim_v2/infer_autovla.py --inputs <dir>/inputs.jsonl --out <dir>/preds \
        --ckpt ... --config .../hparams.yaml --base .../qwen2.5-vl-3b-instruct --shard 0 --nshard 2

输入和 navtest 评测（run_pdm_score_cot.py → AutoVLAAgentFeatureBuilder）一样：
  前视 F0 + 两路侧视（按 config 的 model.video.nuplan_side_cams 选 L0R0 或 L1R1）× 4 帧，
  速度 / 加速度二维向量，导航指令，默认 prompt（reason_then_act）。
输出 <out>/preds_s{shard}.jsonl：{token, stage, traj: 10×(x, y, heading)（0.5 s 间隔，5 s）, text}；出错的行带 error。
"""
import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
os.chdir(REPO)  # config 里的 codebook_cache_path 是相对仓库根目录的

SIDES = {"L0R0": ("cam_l0", "cam_r0"), "L1R1": ("cam_l1", "cam_r1")}


def load_cfg(path, base):
    import yaml
    cfg = yaml.safe_load(open(path))
    cfg = cfg.get("config", cfg)  # lightning 的 hparams.yaml 外面包了一层 config
    cfg["model"]["pretrained_model_path"] = base
    return cfg


def load_model(cfg, ckpt):
    import torch
    from models.autovla import AutoVLA
    model = AutoVLA(cfg, device="cuda:0")
    sd = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=False)["state_dict"]
    res = model.load_state_dict({k.replace("autovla.", ""): v for k, v in sd.items()}, strict=False)
    print(f"[v2-vla] 载入 {ckpt}：missing {len(res.missing_keys)}，unexpected {len(res.unexpected_keys)}", flush=True)
    for name, ks in (("missing", res.missing_keys), ("unexpected", res.unexpected_keys)):
        if ks:
            print(f"  {name} 例：{ks[:5]}", flush=True)
    del sd
    model.eval()
    return model


def features(row, sides):
    left, right = SIDES[sides]
    return {
        "images": {"front_camera": row["cams"]["cam_f0"],
                   "front_left_camera": row["cams"][left],
                   "front_right_camera": row["cams"][right]},
        "vehicle_velocity": row["velocity"],
        "vehicle_acceleration": row["acceleration"],
        "driving_command": row["instruction"],
        "sensor_data_path": None,  # 路径已经是绝对路径
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True, help="训练时的 config（hparams.yaml 或 config/training/*.yaml）")
    ap.add_argument("--base", required=True, help="Qwen2.5-VL-3B-Instruct 目录")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help=">0 时只跑前 N 个（试跑用）")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.inputs)]
    rows = rows[a.shard::a.nshard]
    if a.limit:
        rows = rows[:a.limit]
    os.makedirs(a.out, exist_ok=True)
    fn = os.path.join(a.out, f"preds_s{a.shard}.jsonl")
    done = set()
    for f in os.listdir(a.out):
        if f.startswith("preds_s") and f.endswith(".jsonl"):
            done |= {json.loads(l)["token"] for l in open(os.path.join(a.out, f)) if '"error"' not in l}
    todo = [r for r in rows if r["token"] not in done]
    cfg = load_cfg(a.config, a.base)
    sides = (cfg["model"].get("video") or {}).get("nuplan_side_cams", "L1R1")
    print(f"[v2-vla] 片 {a.shard}/{a.nshard}：{len(rows)} 个 token，待跑 {len(todo)}，侧视 {sides}，"
          f"use_cot={cfg['model']['use_cot']} → {fn}", flush=True)
    if not todo:
        return

    import torch
    torch.manual_seed(0)
    model = load_model(cfg, a.ckpt)
    t0 = time.time()
    with open(fn, "a") as f, torch.no_grad():
        for n, r in enumerate(todo):
            rec = {"token": r["token"], "stage": r["stage"]}
            try:
                traj, text = model.predict(features(r, sides))
                traj = np.asarray(traj.float().cpu() if hasattr(traj, "cpu") else traj, dtype=np.float64)[:, :3]
                rec.update(traj=traj.round(4).tolist(), text=text)
            except Exception as e:  # action token 不够 10 个等
                rec.update(traj=None, text=locals().get("text"), error=repr(e))
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 20 == 0 or n == len(todo) - 1:
                el = time.time() - t0
                print(f"[v2-vla] {n + 1}/{len(todo)}  {el / (n + 1):.1f} s/个  {r['token']} 第{r['stage']}阶段"
                      f"{'  出错 ' + rec['error'] if 'error' in rec else ''}", flush=True)
    print(f"[v2-vla] 片 {a.shard} 完成", flush=True)


if __name__ == "__main__":
    main()
