import sys, os
from pathlib import Path
ROOT = Path("/home/mh2803/vla/doc_vla_search")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"navsim"))
os.chdir(ROOT)
import yaml, torch
from transformers import AutoProcessor
from dataset_utils.sft_dataset import SFTDataset, DataCollator

# optional: pass a config path, e.g. config/training/qwen3-vl-2B-nuplan-nocot-sft.yaml
CFG = sys.argv[1] if len(sys.argv) > 1 else "config/training/qwen2.5-vl-3B-nuplan-nocot-sft.yaml"
print(f"config: {CFG}\n")
cfg = yaml.safe_load(open(CFG))
proc = AutoProcessor.from_pretrained(cfg['model']['pretrained_model_path'], use_fast=True)
ds = SFTDataset(cfg['data']['val'], cfg['model'], proc, using_cot=cfg['model']['use_cot'])
coll = DataCollator(processor=proc,
                    ignore_index=cfg['model']['tokens']['ignore_index'],
                    assistant_id=cfg['model']['tokens']['assistant_id'])

s = ds[0]
bar = "="*90

# ---------- 1. PROMPT ----------
print(bar); print("1) PROMPT  (text fed to apply_chat_template -> tokenizer)"); print(bar)
print(s['text'])

# ---------- 2. RAW VISION INPUT (what process_vision_info returns) ----------
print("\n"+bar); print("2) VISION INPUT  (video_inputs -> decoded frames that go into the ViT)"); print(bar)
vids = s['video_inputs']
print(f"num videos (cameras): {len(vids)}   (image_inputs: {s['image_inputs']})")
import numpy as np
for i, v in enumerate(vids):
    if torch.is_tensor(v):
        print(f"  video[{i}]: tensor shape={tuple(v.shape)} dtype={v.dtype} "
              f"min={float(v.min()):.1f} max={float(v.max()):.1f}  (T,C,H,W)")
    elif isinstance(v, np.ndarray):
        print(f"  video[{i}]: ndarray shape={v.shape} dtype={v.dtype}")
    elif isinstance(v, (list, tuple)):
        e = v[0]
        sz = getattr(e, 'size', None); mode = getattr(e, 'mode', None)
        print(f"  video[{i}]: {len(v)} frames, frame type={type(e).__name__}, "
              f"PIL size(W,H)={sz}, mode={mode}")
    else:
        print(f"  video[{i}]: type={type(v).__name__} "
              f"size={getattr(v,'size',None)} mode={getattr(v,'mode',None)}")

# ---------- 3. COLLATED TENSORS (actual model inputs) ----------
print("\n"+bar); print("3) COLLATED / PROCESSOR OUTPUT  (actual tensors into the model)"); print(bar)
b = coll([s])
for k, v in b.items():
    if torch.is_tensor(v):
        extra = ""
        if k == "input_ids": extra = f"  seq_len={v.shape[1]}"
        print(f"  {k:24s} shape={tuple(v.shape)} dtype={v.dtype}{extra}")
    else:
        print(f"  {k:24s} = {v}")

# video_grid_thw -> ViT patch grid & vision-token count
if "video_grid_thw" in b:
    thw = b["video_grid_thw"]
    merge = proc.image_processor.merge_size
    print(f"\n  video_grid_thw (per camera, in patch units T,H,W):\n  {thw.tolist()}")
    tot_patches = int(thw.prod(dim=1).sum())
    tot_tokens = tot_patches // (merge*merge)
    print(f"  merge_size={merge}  -> total ViT patches={tot_patches}, "
          f"vision tokens after merge={tot_tokens}")

# ---------- 4. TOKEN BREAKDOWN ----------
print("\n"+bar); print("4) TOKEN BREAKDOWN"); print(bar)
ids = b["input_ids"][0]
tok = proc.tokenizer
vid_pad = tok.convert_tokens_to_ids("<|video_pad|>")
img_pad = tok.convert_tokens_to_ids("<|image_pad|>")
n_vid = int((ids == vid_pad).sum()); n_img = int((ids == img_pad).sum())
action_start = cfg['model']['tokens']['action_start_id']
n_action = int((ids >= action_start).sum())
labels = b["labels"][0]
n_supervised = int((labels != cfg['model']['tokens']['ignore_index']).sum())
print(f"  total tokens        : {ids.numel()}")
print(f"  <|video_pad|> tokens: {n_vid}   <|image_pad|>: {n_img}")
print(f"  action tokens (>= {action_start}): {n_action}")
print(f"  supervised tokens (labels != -100): {n_supervised}")
print(f"\n  action token ids: {ids[ids>=action_start].tolist()}")
print(f"  decoded action span: {tok.decode(ids[ids>=action_start])}")
