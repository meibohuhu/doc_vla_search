"""Impromptu VLA 在 CoLT-Drive 上：问 q6（<SPEED PATH PLAN>），读成 CoLT 动作对 —— docs/1104/colt_eval.md §8.1。

权重：HF aaaaaap/ImpromptuVLAModel 的 3B_AD（Base+Impromptu，训练时有 q6 问答），Qwen2.5-VL-3B。
prompt、解析、打分都在 speed_path_plan.py。输入就是 CoLT 的单张前视图，和 Impromptu 训练数据里 Mapillary 的 q6 一样。
对话模板照 LLaMA-Factory 的 qwen2_vl（system = "You are a helpful assistant."，<image> → vision 占位），
图片像素上限 262144（它训练 config 里的 image_max_pixels），贪心解码。

    PY=/home/mh2803/miniconda3/envs/autovla_codeclean/bin/python
    $PY tools/colt/eval_impromptu.py --dry-run                         # 打印第一个样本的完整输入，不加载权重
    CUDA_VISIBLE_DEVICES=2 $PY tools/colt/eval_impromptu.py --every 50  # 冒烟：36 个样本
    CUDA_VISIBLE_DEVICES=2 $PY tools/colt/eval_impromptu.py --shard 0 --nshard 2
    $PY tools/colt/speed_path_plan.py agg --run-dir /scratch/mh2803/vla/colt/impromptu/vfull/3B_AD [--every 50]
中断后重跑会接着做。
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import speed_path_plan as S  # noqa: E402

CKPT = "/scratch/mh2803/vla/checkpoints/impromptu/3B_AD"
OUT = "/scratch/mh2803/vla/colt/impromptu"
SYSTEM = "You are a helpful assistant."
VISION = "<|vision_start|><|image_pad|><|vision_end|>"


def chat(question):
    """LLaMA-Factory qwen2_vl 模板。"""
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{question.replace('<image>', VISION)}<|im_end|>\n"
            "<|im_start|>assistant\n")


def main(a):
    root, ids, meta, acc = S.load_split(a.split, a.data)
    ids = S.select(ids, a.every, a.shard, a.nshard)
    run_dir = os.path.join(a.out, a.split, a.tag)
    done, f = S.open_run(run_dir, a.shard)
    todo = [s for s in ids if s not in done]
    print(f"[impromptu] {a.split} 片 {a.shard}/{a.nshard}：{len(ids)} 个样本，待跑 {len(todo)} → {run_dir}", flush=True)
    if a.dry_run:
        s = (todo or ids)[0]
        print(f"--- {s}\n{chat(S.impromptu_prompt(meta[s]))}")
        return
    if not todo:
        return

    import torch
    from PIL import Image
    from transformers import (AutoTokenizer, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor,
                              Qwen2VLImageProcessor)
    # ckpt 的 preprocessor_config 写的是新版类名 Qwen2_5_VLImageProcessor（transformers 4.49 不认），参数相同
    proc = Qwen2_5_VLProcessor(image_processor=Qwen2VLImageProcessor(min_pixels=3136, max_pixels=a.max_pixels),
                               tokenizer=AutoTokenizer.from_pretrained(a.ckpt))
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        a.ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    for n, s in enumerate(todo):
        img = Image.open(f"{root}/{s}/front_camera.jpg").convert("RGB")
        inp = proc(text=[chat(S.impromptu_prompt(meta[s]))], images=[img], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=a.max_new_tokens, do_sample=False)
        text = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        plan = S.parse(text)
        rec = {"sample_id": s, "text": text, "plan": plan}
        f.write(__import__("json").dumps(rec) + "\n")
        f.flush()
        if n % 20 == 0 or plan is None:
            print(f"[impromptu] {n + 1}/{len(todo)} {s} → {plan}  | {text[:120]!r}", flush=True)
    print(f"[impromptu] 片 {a.shard} 完成", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="vfull", choices=["vfull", "vclean"])
    ap.add_argument("--data", default=S.DATA)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--tag", default=None, help="输出子目录名，默认取 ckpt 目录名")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--max-pixels", type=int, default=262144)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--every", type=int, default=1, help="每 k 个样本取 1 个（冒烟用 50）")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    a.tag = a.tag or os.path.basename(os.path.normpath(a.ckpt))
    main(a)
