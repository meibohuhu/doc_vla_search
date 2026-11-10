"""Senna 在 CoLT-Drive 上：问规划问答（SPEED / PATH），读成 CoLT 动作对 —— docs/1104/colt_eval.md §8.1。

要在 Senna 自己的环境里跑（torch 2.0.1+cu118 / transformers 4.31，Blackwell 卡跑不了），所以只依赖
Senna 的代码和同目录的 speed_path_plan.py（只用标准库）。换机器时把 tools/colt/ 下这两个文件拷过去就行。

权重：HF rb93dett/Senna（7B，LLaVA + vicuna-7b-v1.5）。代码：github.com/hustvl/Senna。
推理照官方 eval_tools/senna_plan_cmd_eval_multi_img.py：load_senna_pretrained_model + eval_multi_img_model_wo_init，
conv_mode llava_v1，temperature 0，max_new_tokens 512。prompt 照 data_tools/senna_nusc_data_converter.py 原文。
输入（--views）：
  6（默认）  训练格式：6 个 <VIEW>:\n<image>，前视图 = CoLT 的图，其余 5 路 = 同尺寸全黑图
  1          只给前视图一个 <image>（官方 eval_tools/senna_plan_visualization.py 也是单张前视图问规划）

    export PYTHONPATH=/path/to/Senna:$PYTHONPATH
    python tools/colt/eval_senna.py --senna /path/to/Senna --model /path/to/Senna-weights --data /path/to/CoLT-Drive/data \
        --out /path/to/out --dry-run                     # 打印第一个样本的 prompt 和图片列表，不加载权重
    CUDA_VISIBLE_DEVICES=0 python tools/colt/eval_senna.py ... --every 50              # 冒烟：36 个样本
    CUDA_VISIBLE_DEVICES=0 python tools/colt/eval_senna.py ... --views 1 --every 50    # 单图对照
    CUDA_VISIBLE_DEVICES=0 python tools/colt/eval_senna.py ... --shard 0 --nshard 2    # 全量分两片
    python tools/colt/speed_path_plan.py agg --data /path/to/CoLT-Drive/data --run-dir /path/to/out/vfull/senna_views6 [--every 50]
中断后重跑会接着做。结果目录拷回来后，在这台机器上 agg 也行（--data 用默认）。
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import speed_path_plan as S  # noqa: E402


def images_for(root, sid, views, run_dir):
    front = f"{root}/{sid}/front_camera.jpg"
    if views == 1:
        return [front]
    from PIL import Image
    w, h = Image.open(front).size
    black = os.path.join(run_dir, f"black_{w}x{h}.jpg")
    if not os.path.exists(black):
        Image.new("RGB", (w, h)).save(black, quality=95)
    return [front] + [black] * (views - 1)


def main(a):
    root, ids, meta, acc = S.load_split(a.split, a.data)
    ids = S.select(ids, a.every, a.shard, a.nshard)
    run_dir = os.path.join(a.out, a.split, a.tag)
    done, f = S.open_run(run_dir, a.shard)
    todo = [s for s in ids if s not in done]
    print(f"[senna] {a.split} 片 {a.shard}/{a.nshard}：{len(ids)} 个样本，待跑 {len(todo)} → {run_dir}", flush=True)
    if a.dry_run:
        s = (todo or ids)[0]
        print(f"--- {s}\n{S.senna_prompt(meta[s], a.views)}\n--- images: {images_for(root, s, a.views, run_dir)}")
        return
    if not todo:
        return

    sys.path.insert(0, a.senna)
    from llava.model.builder import load_senna_pretrained_model
    from data_tools.senna_qa_utils import eval_multi_img_model_wo_init
    tokenizer, model, image_processor, _ = load_senna_pretrained_model(a.model, None, model_name="llava", device_map=0)
    for n, s in enumerate(todo):
        args = type("Args", (), {
            "model_path": a.model, "model_base": None, "query": S.senna_prompt(meta[s], a.views),
            "conv_mode": "llava_v1", "image_file": images_for(root, s, a.views, run_dir), "sep": ",",
            "temperature": 0, "top_p": None, "num_beams": 1, "max_new_tokens": 512})()
        text = eval_multi_img_model_wo_init(args, tokenizer, model, image_processor)
        plan = S.parse(text)
        f.write(json.dumps({"sample_id": s, "text": text, "plan": plan}) + "\n")
        f.flush()
        if n % 20 == 0 or plan is None:
            print(f"[senna] {n + 1}/{len(todo)} {s} → {plan}  | {text[:120]!r}", flush=True)
    print(f"[senna] 片 {a.shard} 完成", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--senna", default=os.environ.get("SENNA_ROOT", "/home/mh2803/vla/Senna"), help="Senna 代码目录")
    ap.add_argument("--model", required=True, help="Senna 权重目录（HF rb93dett/Senna）")
    ap.add_argument("--data", default=S.DATA, help="CoLT-Drive 的 data 目录（下面是 vfull / vclean）")
    ap.add_argument("--out", default="/scratch/mh2803/vla/colt/senna")
    ap.add_argument("--split", default="vfull", choices=["vfull", "vclean"])
    ap.add_argument("--views", type=int, default=6, choices=[1, 6])
    ap.add_argument("--tag", default=None, help="输出子目录名，默认 senna_views{N}")
    ap.add_argument("--every", type=int, default=1, help="每 k 个样本取 1 个（冒烟用 50）")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    a.tag = a.tag or f"senna_views{a.views}"
    main(a)
