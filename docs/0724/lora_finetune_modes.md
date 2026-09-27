# LLM LoRA 微调支持（ViT × LLM 的四种组合）

> 给 SFT 加了第三种 LLM 调优方式：LoRA。配合 ViT 冻结/全参，共四种组合，靠 config 切换。
> 目的：全参 LLM 太慢/太占显存，用 LoRA 换取速度，看掉多少分。
> 日期：2026-07-25 · 相关：[bf16_master_weights_bug.md](bf16_master_weights_bug.md)

---

## 0. 四种组合一览

`train_lm_backbone` 从 bool 扩成三态（`True` / `False` / `"lora"`），
与 `train_vision_backbone`（true/false）正交组合：

| config 文件 | ViT | LLM | 可训参数 | 用途 |
|---|---|---|---|---|
| `...brev.yaml` | 冻结 | 全参 | 3.1B | **已验证基线 PDMS 80.06** |
| `...brev-vit.yaml` | 全参 | 全参 | 3.8B | 隔离"ViT 解冻"的作用 |
| `...brev-vit-lora.yaml` | 全参 | **LoRA** | **1.31B** | ViT 解冻 + LLM 省钱 |
| `...brev-frozenvit-lora.yaml` | 冻结 | **LoRA** | **~0.64B** | 最省，纯 LoRA |

（均在 `config/training/` 下，前缀 `qwen2.5-vl-3B-nuplan-nocot-sft-navtrain-`）

---

## 1. LoRA 模式的可训参数分布（实测，r=16 时）

```
ViT全参 + LLM LoRA：可训 1.306B / 4.40B (29.7%)
  ViT       : 669M   ← 全参解冻
  LoRA      : 7.4M   ← q/k/v/o_proj 的 adapter，省的就是这里（r=32 时约 15M）
  embed/head: 630M   ← 全参（★ action token 能学的关键，见 §3）
  LLM 主体  : 0M     ← 除 LoRA 外全冻
```

对比全参基线：LLM 主体的 ~2.5B 被 7.4M（r=16）/ ~15M（r=32）的 adapter 取代。
ViT 冻结版则再省掉 669M，可训参数降到约 640M（几乎只剩 embedding + LoRA）。

---

## 2. 代码改动（两处，都在 `models/autovla.py`）

### 2.1 `SFTAutoVLA.__init__`：应用 LoRA

**为什么在 `__init__` 而不是 `configure_optimizers`**：`run_sft.py` 的执行顺序是
`SFTAutoVLA(config)` → `model.float()`（fp32 master）→ 建 strategy → `trainer.fit`（内部才调
`configure_optimizers`）。LoRA 必须在 `model.float()` 和 strategy 包装**之前**就位，
所以放 `__init__`。

```python
self._is_lora = (str(self._train_llm_backbone).lower() == 'lora')
if self._is_lora:
    from peft import LoraConfig, get_peft_model, TaskType
    lc = config['model'].get('lora', {})
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=lc.get('target_modules', ['q_proj','v_proj','k_proj','o_proj']),
        modules_to_save=lc.get('modules_to_save', ['lm_head','embed_tokens']),  # ★ 见 §3
        r=lc.get('r', 16), lora_alpha=lc.get('alpha', 32),
        lora_dropout=lc.get('dropout', 0.05), bias='none',
    )
    self.autovla.vlm = get_peft_model(self.autovla.vlm, lora_config)
    # get_peft_model 冻结【所有】非 LoRA/非 modules_to_save 的参数——包括 ViT。
    # 想训 ViT 必须在这里手动重开，否则 config 写 true 也没用。
    if self._train_vision_backbone:
        for p in self.autovla.vlm.base_model.model.visual.parameters():
            p.requires_grad = True
    self.autovla.vlm.print_trainable_parameters()
```

### 2.2 `configure_optimizers`：LoRA 时跳过 flag 冻结

peft 包装后模块路径变了（`vlm.visual` → `vlm.base_model.model.visual`），
原来的 flag 冻结逻辑会 `AttributeError`；而 requires_grad 已在 `__init__` 里设好，直接跳过：

```python
if not getattr(self, '_is_lora', False):
    # 只有非 LoRA 才走原来的 flag 冻结
    if not self._train_vision_backbone: ...
    if not self._train_llm_backbone: ...
# LoRA 模式：requires_grad 已由 peft + ViT 重开设定
params_to_update = [p for p in self.autovla.vlm.parameters() if p.requires_grad]
```

---

## 3. ⚠️ 最容易踩的坑：`modules_to_save` 必须含 embedding

模型要学 **2048 个全新的 `<action_i>` token**，`resize_token_embeddings` 给它们的是
**随机初始化**。而 **LoRA 只作用于 `target_modules`（q/k/v/o_proj），碰不到 embedding 矩阵**。

→ 如果不把 `embed_tokens` / `lm_head` 放进 `modules_to_save`，那 2048 行 embedding
**永远是随机的**，模型根本学不会输出动作。

**症状**：PDMS 很低、action top-1 卡住——**酷似之前的 bf16 master bug**，极易误判成
"LoRA 容量不够"或"又中了精度坑"。

- Qwen2.5-VL-3B 是 `tie_word_embeddings=True`，embed 和 lm_head 共享矩阵；
  两个都列，由 peft 处理绑定关系。
- 冒烟测试务必确认 `embed/head` 那 ~630M 在**可训**参数里（见 §5）。

---

## 4. config 关键字段

```yaml
model:
  train_vision_backbone: true       # true=ViT全参 / false=ViT冻结
  train_lm_backbone: lora           # True=全参 / False=冻结 / lora=LoRA
  lora:
    r: 32
    alpha: 64                       # 常规 α=2r
    dropout: 0.05
    target_modules: [q_proj, v_proj, k_proj, o_proj]
    modules_to_save: [lm_head, embed_tokens]   # ★ 必须，见 §3

training:
  fp32_master: true                 # ★ LoRA 一样要开，见 §6
  learning_rate: 1.0e-4             # LoRA 比全参大（adapter 从~0 起，2e-5 太慢）
```

---

## 5. 冒烟测试（改完必跑，确认没踩 §3）

```python
m = SFTAutoVLA(cfg)                       # ViT全参 + LLM LoRA
# 断言可训参数分布正确：
#   LLM 主体可训 == 0        （LoRA 生效）
#   embed/head 可训 > 1e8    （action token 能学）
#   ViT 可训 > 1e8           （若 train_vision_backbone=true）
```

实测已通过（r=16）：可训 1.306B，LLM 主体 0M，embed 630M，ViT 669M，LoRA 7.4M。
forward + backward 也跑通，LoRA adapter 确实接收梯度（144 个张量有非零梯度）。

---

## 6. fp32 master 对 LoRA 同样必要

**别因为换了 LoRA 就关 `fp32_master`。** LoRA 的 adapter 参数量虽小，但它们的更新
**同样会被 bf16 舍入**（lr=1e-4 的更新相对 adapter 量级仍可能低于 bf16 分辨率）。
`run_sft.py` 的 `model.float()` 对 peft 模型一样适用——冻结的 base 也会被转 fp32，
略有显存浪费，但 Adam 状态只对可训参数存在，总显存仍远低于全参。

---

## 7. 评测加载 LoRA ckpt：必须先 merge（已解决）

LoRA 训练存的 ckpt 是 **peft 结构**（key 带 `base_model.model.` 前缀 + `lora_A/lora_B`），
而评测脚本 `eval_sft_ckpts.sh` 用的 `AutoVLAAgent` 是 **plain `load_state_dict(strict=False)`**。
**直接喂 LoRA ckpt → key 对不上 → 静默部分加载 → 跑出垃圾分数但不报错**（同 bf16 那类坑）。

**已实现方案 A**：[`scripts/0725/merge_lora_ckpt.py`](../../scripts/0725/merge_lora_ckpt.py)
用训练 config 重建 peft 模型 → 加载 ckpt → `merge_and_unload()` → 存成普通全参 ckpt，
key 回到 `autovla.vlm.visual/model/lm_head...`，评测脚本不用改。

```bash
python scripts/0725/merge_lora_ckpt.py \
    --config training/<lora config 名> \
    --ckpt   <训练存的 epoch=X.ckpt> \
    --out    <merged.ckpt>
# 然后照常评测：
CKPTS="<merged.ckpt>" bash scripts/0723/eval_sft_ckpts.sh
```

脚本内置三道断言防静默失败：① 加载时 LoRA/modules_to_save key 全匹配（missing=0）；
② merge 后无 `lora_/base_model` 残留；③ ViT+LLM+embed 结构完整。

> ⚠️ 已知代价：LoRA ckpt 目前会存 fp32 全量权重（~17.6GB，因为 `model.float()` 把 base 也转了
> fp32，Lightning 默认存整个 state_dict），而非只存 ~15M adapter。正式训练存 top-3 = ~53GB/次。
> 可在 `on_save_checkpoint` 里只保留可训参数来省，暂未做（`/data` 空间够）。

---

## 8. 实验设计提醒

`brev-vit-lora` **同时动了两个变量**（ViT 冻→全参、LLM 全参→LoRA）。
若结果比基线（80.06）差，分不清是 ViT 解冻帮倒忙还是 LoRA 容量不够。
干净的归因需要 `brev-vit`（只解冻 ViT）的基线做中间参照：

```
brev (80.06)  ──+ViT解冻──►  brev-vit (?)  ──LLM换LoRA──►  brev-vit-lora (?)
                                    └──────────────────────►  brev-frozenvit-lora (?)  ←只换LoRA
```

- `brev-frozenvit-lora` vs `brev`：**只差 LLM 全参→LoRA**，干净隔离 LoRA 的代价
- `brev-vit-lora` vs `brev-vit`：同上，但在 ViT 全参的基础上

---

## 9. 工具链就绪 & dry-run 验证

整套 LoRA 工具链已跑通并验证：

| 组件 | 路径 |
|---|---|
| ViT全参 + LLM LoRA config | `config/training/qwen2.5-vl-3B-nuplan-nocot-sft-navtrain-brev-vit-lora.yaml` |
| ViT冻 + LLM LoRA config | `config/training/qwen2.5-vl-3B-nuplan-nocot-sft-navtrain-brev-frozenvit-lora.yaml` |
| merge 脚本（评测前必须） | [`scripts/0725/merge_lora_ckpt.py`](../../scripts/0725/merge_lora_ckpt.py) |
| 本文档 | `docs/0724/lora_finetune_modes.md` |

**正式流程**：`run_sft.py`（用某个 lora config）→ `merge_lora_ckpt.py` → `eval_sft_ckpts.sh`。

### dry-run 全链路验证（500 样本 × 1 epoch，config `training/0725/lora_dryrun`）

| 环节 | 结果 |
|---|---|
| ① LoRA 训练 → 存 ckpt | ✅ trainable 1.31B(29.8%) · action_loss 19.4→12.6 在学 · `param dtype=float32`(fp32生效) · 无 OOM |
| ② merge → 普通 ckpt | ✅ missing=0 unexpected=0 · 825 key(=全参 ckpt 结构) · 无 peft 残留 |
| ③ 评测加载 merged ckpt | ✅ **PDMS 58.40**(n=20)，正常出分不静默失败 |

**结论**：`modules_to_save`（§3）和 fp32 master（§6）都对了——action token 在学（action_loss 下降），
merge 后评测能加载。58.40 是"只训 1 epoch × 500 样本"的欠训结果（DAC=100 稳、EP=32 慢的保守策略），
远高于恒速底分 19.3，证明路径正确、非垃圾。

## 10. 待办

- [x] 写 LoRA ckpt merge 脚本（§7 方案 A）
- [x] dry-run 验证全链路（训练→存→merge→评测）
- [ ] `brev-vit`（ViT全参+LLM全参）基线数字（在别的机器上跑）
- [ ] 两个 LoRA 配置的正式 PDMS（navtrain 101k 全量训练），与 80.06 对比
- [ ] （可选）`on_save_checkpoint` 只存可训参数，把 LoRA ckpt 从 17.6GB 降到几百 MB
