#!/usr/bin/env python3
"""
把 LoRA 训练的 Lightning ckpt merge 成普通全参 ckpt，让评测脚本（AutoVLAAgent）能直接加载。

为什么需要
----------
LoRA 训出的 ckpt 是 peft 结构：key 带 base_model.model. 前缀 + lora_A/lora_B。
而 AutoVLAAgent 用 plain load_state_dict(strict=False)（autovla_agent.py:389）——
直接喂 LoRA ckpt 会 key 对不上、静默部分加载、跑出垃圾分数但不报错。
所以评测前必须先 merge。

做什么
------
1. 用训练 config 重建 SFTAutoVLA（会应用同样的 peft 包装）
2. 把 Lightning ckpt 的 state_dict 加载进去（strict=False，但会打印匹配情况）
3. peft_model.merge_and_unload() 把 LoRA 权重融进 base
4. 存成 {"state_dict": {autovla.vlm.<普通 qwen 结构>}} —— 与 AutoVLAAgent 完全兼容

用法
----
    python scripts/0725/merge_lora_ckpt.py \
        --config training/qwen2.5-vl-3B-nuplan-nocot-sft-navtrain-brev-vit-lora \
        --ckpt  /path/to/lora_epoch=X.ckpt \
        --out   /path/to/merged.ckpt
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "navsim"))

import torch
import yaml
from models.autovla import SFTAutoVLA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="训练 config 名（相对 config/，不带 .yaml）")
    ap.add_argument("--ckpt", required=True, help="LoRA 训练存的 Lightning ckpt")
    ap.add_argument("--out", required=True, help="merge 后的输出 ckpt")
    ap.add_argument("--device", default="cpu", help="merge 在哪算，cpu 省显存（默认）")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(REPO / "config" / f"{args.config}.yaml"))
    assert str(cfg["model"]["train_lm_backbone"]).lower() == "lora", \
        f"config 的 train_lm_backbone 不是 lora（{cfg['model']['train_lm_backbone']}），无需 merge"

    print(f"[1/4] 重建 SFTAutoVLA（应用 peft 包装）...")
    # AutoVLA.__init__ 默认 device='cpu'，重建时不占显存
    model = SFTAutoVLA(cfg)

    print(f"[2/4] 加载 LoRA ckpt: {args.ckpt}")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    # 去掉可能的 reference_model.（SFT 没有，但保险）
    sd = {k: v for k, v in sd.items() if not k.startswith("reference_model.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # 关键健壮性检查：LoRA / modules_to_save 的 key 应当全部匹配上
    lora_missing = [k for k in missing if "lora_" in k or "modules_to_save" in k]
    if lora_missing:
        print(f"  ⚠️  {len(lora_missing)} 个 LoRA/modules_to_save key 没匹配上！merge 会不完整。")
        for k in lora_missing[:5]:
            print(f"      {k}")
    print(f"  missing={len(missing)}  unexpected={len(unexpected)}  "
          f"(base 权重 missing 是正常的——ckpt 只存了训练的部分)")

    print(f"[3/4] merge_and_unload（LoRA 融进 base）...")
    if args.device != "cpu":
        model = model.to(args.device)
    merged_vlm = model.autovla.vlm.merge_and_unload()
    model.autovla.vlm = merged_vlm      # 现在 vlm 是普通 Qwen2_5_VL，无 peft 结构

    print(f"[4/4] 存成 AutoVLAAgent 兼容格式: {args.out}")
    # 只留 autovla.* （去掉 lightning 的其它 key），且此时结构已是普通 qwen
    out_sd = {k: v.to(torch.float32).cpu() for k, v in model.state_dict().items()
              if k.startswith("autovla.")}
    # 校验：merge 后不该再有 lora_ / base_model 残留
    bad = [k for k in out_sd if "lora_" in k or "base_model" in k or "modules_to_save" in k]
    assert not bad, f"merge 后仍有 peft 残留 key: {bad[:3]}"
    # 校验：关键结构都在
    has_vit = any("vlm.visual" in k for k in out_sd)
    has_llm = any("vlm.model.layers" in k for k in out_sd)
    has_emb = any("embed_tokens" in k for k in out_sd)
    print(f"  key 数: {len(out_sd)}   ViT={has_vit} LLM={has_llm} embed={has_emb}")
    assert has_vit and has_llm and has_emb, "merge 后结构不完整！"

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": out_sd}, args.out)
    sz = Path(args.out).stat().st_size / 1e9
    print(f"\n✅ 完成。{args.out}  ({sz:.1f} GB)")
    print(f"   现在可以直接用 eval_sft_ckpts.sh 评测这个 merged ckpt。")


if __name__ == "__main__":
    main()
