"""Derive the tokenizer-specific ids that config/training/*.yaml hardcodes.

Qwen2.5-VL and Qwen3-VL have different vocabs, so `action_start_id` and
`assistant_id` must be re-derived when switching backbones. Run:

    python logs/derive_token_ids.py <model_path>

Self-check: run it on ./Qwen2.5-VL-3B-Instruct and it must reproduce the known
good values action_start_id=151665, assistant_id=[151644, 77091].
"""
import sys, pickle
from pathlib import Path
ROOT = Path("/home/mh2803/vla/doc_vla_search")
sys.path.insert(0, str(ROOT))
from transformers import AutoProcessor

model_path = sys.argv[1] if len(sys.argv) > 1 else "./Qwen2.5-VL-3B-Instruct"

# how many action tokens does the codebook define?
with open(ROOT / "codebook_cache/agent_vocab.pkl", "rb") as f:
    n_actions = len(pickle.load(f)['token_all']['veh'])

proc = AutoProcessor.from_pretrained(model_path)
tok = proc.tokenizer
print(f"model_path      : {model_path}")
print(f"vocab before add: {len(tok)}")

# mirror ActionTokenizer.__init__
tok.add_tokens([f'<action_{i}>' for i in range(n_actions)], special_tokens=False)
print(f"action tokens   : {n_actions}")
print(f"vocab after add : {len(tok)}")

action_start_id = tok.convert_tokens_to_ids('<action_0>')
action_last_id  = tok.convert_tokens_to_ids(f'<action_{n_actions-1}>')

# assistant turn marker: what the DataCollator scans for to start label masking
im_start = tok.convert_tokens_to_ids('<|im_start|>')
assistant_ids = tok.encode("assistant", add_special_tokens=False)
assistant_id = [im_start] + assistant_ids

# verify against a real templated chat
probe = tok.apply_chat_template(
    [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
    tokenize=True, add_generation_prompt=False)
found = any(probe[i:i+len(assistant_id)] == assistant_id for i in range(len(probe)))

print(f"\naction_start_id : {action_start_id}   (<action_0>)")
print(f"action_last_id  : {action_last_id}   (<action_{n_actions-1}>)")
print(f"assistant_id    : {assistant_id}   decoded={tok.decode(assistant_id)!r}")
print(f"  marker found in templated chat: {found}")
if not found:
    print("  !! WARNING: assistant marker not found -- label masking would break.")

print("\n--- paste into config ---")
print(f"    action_start_id: {action_start_id}")
print(f"    ignore_index: -100")
print(f"    assistant_id: {assistant_id}")
