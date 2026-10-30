"""把 0926_tiger 三个 no-CoT ViT 解冻 SFT 的 run 目录传到 HuggingFace（public model repo）。

每个 run 一个 repo：整个目录（ckpt + lightning_logs + wandb 离线数据）原样上传，不加 README。

    HF_TOKEN=hf_xxx python scripts/1001/upload_runs_hf.py
    HF_TOKEN=hf_xxx python scripts/1001/upload_runs_hf.py --only navfiltbal   # 只传一个

token 只从环境变量 HF_TOKEN 读。upload_large_folder 可断点续传：中断后原命令重跑即可
（续传状态存在各 run 目录下的 .cache/huggingface/）。
"""
import argparse
import os

from huggingface_hub import HfApi

RUNS = "/shared/rc/llm-gen-agent/mhu/auto_vla/runs/sft"
USER = "hellooPheonix"

# key -> (run 目录, repo 名)
SPECS = {
    # navtrain (101,288) + trainval_166k_navfilt_bal (42,473)，4 节点 x accum 8
    "navfiltbal": ("2026-09-27_01-20-47", "autovla-nocot-sft-navtrain-navfiltbal-vit-unfreeze-adamw8bit-tigris4n"),
    # navtrain only (101,288)，8 节点 x accum 4（对照组）
    "navtrain": ("2026-09-27_01-25-34", "autovla-nocot-sft-navtrain-vit-unfreeze-adamw8bit-tigris8n"),
    # navtrain (101,288) + trainval_166k_navfilt_turn40 (19,322)，4 节点 x accum 8
    "navfiltturn40": ("2026-09-27_01-25-35", "autovla-nocot-sft-navtrain-navfiltturn40-vit-unfreeze-adamw8bit-tigris4n"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(SPECS), help="只传其中一个")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    api = HfApi()  # token 取自 HF_TOKEN
    for key in ([args.only] if args.only else SPECS):
        ts, repo = SPECS[key]
        run_dir, repo_id = os.path.join(RUNS, ts), f"{USER}/{repo}"
        print(f"\n===== {key}: {run_dir} -> https://huggingface.co/{repo_id}")
        api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
        api.upload_large_folder(repo_id=repo_id, folder_path=run_dir, repo_type="model",
                                num_workers=args.workers)
        print(f"===== {key} done")


if __name__ == "__main__":
    main()
