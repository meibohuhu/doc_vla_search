import sys, os
from pathlib import Path
ROOT = Path("/home/mh2803/vla/doc_vla_search")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"navsim"))
os.chdir(ROOT)
import yaml, torch
from transformers import AutoProcessor
from dataset_utils.sft_dataset import SFTDataset, DataCollator
from models.autovla import SFTAutoVLA

cfg = yaml.safe_load(open("config/training/qwen2.5-vl-3B-nuplan-nocot-sft.yaml"))
proc = AutoProcessor.from_pretrained(cfg['model']['pretrained_model_path'], use_fast=True)
using_cot = cfg['model']['use_cot']
ds = SFTDataset(cfg['data']['val'], cfg['model'], proc, using_cot=using_cot)
coll = DataCollator(processor=proc, ignore_index=cfg['model']['tokens']['ignore_index'],
                    assistant_id=cfg['model']['tokens']['assistant_id'])
batch = coll([ds[0]])

model = SFTAutoVLA(cfg).cuda().eval()
emb = model.autovla.vlm.model.embed_tokens
print("embed num_embeddings =", emb.num_embeddings, flush=True)
print("tokenizer len =", len(proc.tokenizer), flush=True)
print("lm_head out =", model.autovla.vlm.lm_head.weight.shape, flush=True)

ii = batch['input_ids']
print("input_ids dtype/device:", ii.dtype, ii.device, "shape", tuple(ii.shape), flush=True)
print("input_ids min/max:", int(ii.min()), int(ii.max()), flush=True)
print("labels min/max:", int(batch['labels'].min()), int(batch['labels'].max()), flush=True)
print("action_start_id (model):", model.autovla.action_start_id, flush=True)

# move every tensor to cuda explicitly
for k in list(batch.keys()):
    v = batch[k]
    if torch.is_tensor(v):
        batch[k] = v.cuda()
print("running forward...", flush=True)
with torch.no_grad():
    out = model.autovla(batch)
print("FORWARD OK -> loss=", float(out.loss), "logits", tuple(out.logits.shape), flush=True)
