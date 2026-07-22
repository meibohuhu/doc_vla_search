"""Does the ViT actually receive gradients when train_vision_backbone=True?

requires_grad=True + being in an optimizer param group is necessary but NOT
sufficient -- gradient checkpointing on a module whose inputs don't require grad
can silently yield None grads. This runs one real fwd+bwd and reports grad norms.
"""
import sys, os, yaml, torch
from pathlib import Path
ROOT = Path("/home/mh2803/vla/doc_vla_search")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "navsim"))
os.chdir(ROOT)
from transformers import AutoProcessor
from dataset_utils.sft_dataset import SFTDataset, DataCollator
from models.autovla import SFTAutoVLA

cfg = yaml.safe_load(open("config/training/qwen2.5-vl-3B-nuplan-nocot-sft-vit-unfreeze.yaml"))
proc = AutoProcessor.from_pretrained(cfg['model']['pretrained_model_path'], use_fast=True)
ds = SFTDataset(cfg['data']['val'], cfg['model'], proc, using_cot=cfg['model']['use_cot'])
coll = DataCollator(processor=proc,
                    ignore_index=cfg['model']['tokens']['ignore_index'],
                    assistant_id=cfg['model']['tokens']['assistant_id'])

model = SFTAutoVLA(cfg)
# replicate run_sft.py exactly
model.autovla.vlm.model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False})
if cfg['model'].get('train_vision_backbone', False):
    model.autovla.vlm.visual.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    print("[mem] gradient checkpointing enabled on vision tower")

model.configure_optimizers()          # this is what sets requires_grad
model = model.cuda().train()

batch = coll([ds[0]])
batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}

loss = model.training_step(batch)
print(f"\nloss = {float(loss):.4f}")
loss.backward()

vis_ids = {id(p) for p in model.autovla.vlm.visual.parameters()}
def stats(params):
    params = [p for p in params if p.requires_grad]
    with_grad = [p for p in params if p.grad is not None]
    nonzero = [p for p in with_grad if float(p.grad.abs().sum()) > 0]
    gn = torch.sqrt(sum((p.grad.float()**2).sum() for p in with_grad)) if with_grad else torch.tensor(0.)
    return len(params), len(with_grad), len(nonzero), float(gn)

vis = [p for p in model.autovla.vlm.parameters() if id(p) in vis_ids]
oth = [p for p in model.autovla.vlm.parameters() if id(p) not in vis_ids]

print("\n" + "="*78)
print(f"{'module':<16}{'trainable':>10}{'grad!=None':>12}{'grad!=0':>10}{'grad_norm':>14}")
print("="*78)
for name, ps in [("ViT (visual)", vis), ("rest (LLM)", oth)]:
    n, g, nz, gn = stats(ps)
    print(f"{name:<16}{n:>10}{g:>12}{nz:>10}{gn:>14.6f}")
print("="*78)

n, g, nz, gn = stats(vis)
if n and g == n and nz > 0 and gn > 0:
    print("\n=> VERDICT: ViT IS receiving gradients.")
else:
    print(f"\n=> VERDICT: ViT is NOT training properly "
          f"(trainable={n}, with_grad={g}, nonzero={nz}, norm={gn}).")
