# Backbone 切换(Qwen3-VL)与 LLM LoRA 改造

> 配套文档:[AutoVLA_nocot_sft_runbook.md](AutoVLA_nocot_sft_runbook.md)(跑通手册)
> 日期:2026-07-21
> 机器:3× RTX PRO 6000 Blackwell 97G(见 runbook,已切 DDP + NCCL 双 flag)

本文分两部分:
- **§1 Qwen3-VL 切换** —— ✅ **已实现**,只差一条 pip 升级命令
- **§2 ViT 解冻 + LLM LoRA** —— ⚠️ **仅设计,未实现**,含 3 个会静默失败的坑

---

## 1. Qwen3-VL-2B / 4B 切换(已实现)

### 1.1 为什么值得换

对比之前评估过的 InternVL2.5-2B:

| | Qwen2.5-VL-3B(当前) | **Qwen3-VL-2B** | Qwen3-VL-4B | InternVL2.5-2B |
|---|---|---|---|---|
| 参数量 | 3.75B(实测) | **2.13B** | 4.44B | 2.21B |
| 视觉 token(12 张图) | **720**(实测) | 预期同量级 | 同 | **3072**(4.3×) |
| 原生 video / 时序合并 | ✅ | ✅ | ✅ | ❌ 无 |
| 数据管线改动 | — | **几乎为零** | 同 | **重写 ~200 行** |
| 工作量 | — | **半天** | 同 | 2–4 天 |

**结论:Qwen3-VL-2B 是唯一能真正压缩训练时间的选项**(更小的 LLM + 视觉 token 预算不变);InternVL 因为 tiling 没有时序合并,反而更慢。

> InternVL 的 448×448 tile 固定 256 token/图是地板价,无法像 Qwen 用 `min_pixels` 压到 120 token/帧。

### 1.2 兼容性验证结论(非破坏性验证)

验证方式:用 `python -m venv --system-site-packages` 建独立 venv 复用 base 环境的 torch,全程 `autovla_codeclean` 保持 transformers 4.49.0 未被修改。

| 检查项 | 结果 |
|---|---|
| Qwen3-VL 支持 | ✅ **transformers 4.57.6** 已注册 `qwen3_vl` / `qwen3_vl_text` / `qwen3_vl_moe` |
| Python 版本 | ✅ **无需升级**。4.57.6 兼容 Python 3.9(我们是 3.9.23) |
| 依赖冲突 | ✅ dry-run 干净:**不动 torch / peft / pytorch-lightning / numpy** |
| 实际会装 | transformers 4.57.6、tokenizers 0.22.2、huggingface_hub 0.36.2、qwen-vl-utils 0.0.14 |

> ⚠️ **排查提示**:`'qwen3_vl' in transformers.models.__dict__` 会给**假阴性**(惰性导入)。要用权威注册表:
> `from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES`

### 1.3 transformers 4.49 → 4.57 的 VLM 重构影响

新版把 VLM 结构改了(`vlm.model` 从「LLM 本体」变成「同时含 language_model 和 visual 的壳」):

| 路径 | 4.57.6 下 | 仓库影响 |
|---|---|---|
| `vlm.visual` | ✅ 仍可用 | 无 |
| `vlm.lm_head` | ✅ 仍可用 | 无 |
| `vlm.resize_token_embeddings` | ✅ 仍可用 | 无 |
| `vlm.model.embed_tokens` | ❌ **失效** → `vlm.model.language_model.embed_tokens` | 仓库代码未使用,仅调试脚本受影响 |
| `vlm.model.parameters()` | ⚠️ **语义变宽**(现在含 ViT) | [autovla.py](../../models/autovla.py) `configure_optimizers` 里 `train_lm_backbone: false` 分支会**连 ViT 一起冻**。我们用 `true`,当前无影响,但属**潜伏 bug** |
| `vlm.model.gradient_checkpointing_enable()` | ⚠️ 仍可用,但**现在会连 ViT 一起 checkpoint** | [run_sft.py:75](../../tools/run_sft.py) —— 省显存、略慢,**升级后速度需重测** |

### 1.4 已做的代码改动

**(a) 模型类自动选择** —— [models/autovla.py](../../models/autovla.py) 新增 `load_vlm()`:

```python
def load_vlm(model_path, device):
    """根据 checkpoint 的 model_type 选类,Qwen2.5-VL / Qwen3-VL 共用一套代码。"""
    model_type = AutoConfig.from_pretrained(model_path).model_type
    if model_type.startswith("qwen3_vl"):
        cls = Qwen3VLMoeForConditionalGeneration if "moe" in model_type \
              else Qwen3VLForConditionalGeneration
    else:
        cls = Qwen2_5_VLForConditionalGeneration
    return cls.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device)
```

transformers 太老时抛**可操作**的 ImportError,而不是 `AutoConfig` 那句含糊的 ValueError。

✅ 回归测试:原 Qwen2.5-VL 路径经 `load_vlm` 正常加载(3.75B),无回归。

**(b) 新增 config**
- [config/training/qwen3-vl-2B-nuplan-nocot-sft.yaml](../../config/training/qwen3-vl-2B-nuplan-nocot-sft.yaml)
- [config/training/qwen3-vl-4B-nuplan-nocot-sft.yaml](../../config/training/qwen3-vl-4B-nuplan-nocot-sft.yaml)

继承 DDP / batch=2 / accum=8(global batch 32),仅改模型路径与 token id。

**(c) 新增工具** —— [logs/derive_token_ids.py](../../logs/derive_token_ids.py)

自动推导 tokenizer 相关 id。**已自检**:在 Qwen2.5-VL 上精确复现已知正确值(151665 / `[151644, 77091]` / vocab 151665→153713)。

```bash
python logs/derive_token_ids.py Qwen/Qwen3-VL-2B-Instruct
```

### 1.5 ⚠️ 坑:`action_start_id` 只差 4,且静默失败

| | Qwen2.5-VL-3B | **Qwen3-VL-2B / 4B** |
|---|---|---|
| vocab(加 action token 前) | 151665 | **151669** |
| `action_start_id` | 151665 | **151669** |
| `assistant_id` | `[151644, 77091]` | **`[151644, 77091]`**(相同) |

**只差 4**。直接沿用旧 config **不会报任何错**,但:
- `action_mask = (labels >= action_start_id)`([autovla.py](../../models/autovla.py) `training_step`)会混进 4 个非 action token
- 动作解码整体错位 → **输出垃圾轨迹,训练照跑,loss 照降**

→ 换 backbone **必须**重跑 `derive_token_ids.py`。

### 1.6 启用步骤

```bash
# 1. 升级(dry-run 已验证安全)
pip install -U 'transformers>=4.57' 'qwen-vl-utils'

# 2. 先验证移植,不要直接盲跑训练
python logs/inspect_sample.py        # 改 config 路径后

# 3. 训练
bash logs/run_nocot_sft.sh           # 把 --config 指向 qwen3-vl-2B-...
```

**第 2 步的验收清单**(三个数一对就说明移植成功):

| 检查 | Qwen2.5 基准值 | 说明 |
|---|---|---|
| 视觉 token 数 | **720** | 变了说明视觉预算变了,时间估算要重算 |
| action token id 范围 | 起点 = `action_start_id` | 错位就是 §1.5 那个坑 |
| 被监督 token 数 | **32**(其中 10 个 action) | 变了说明 label masking 出问题 |

升级后还需**重测速度**(因 §1.3 的 gradient checkpointing 行为变化)。

---

## 1.7 附:ViT 解冻(不含 LoRA)—— ✅ 已实现

如果只想解冻 ViT、**LLM 仍然全参**(即在原配方上只动一个变量),现在只需两行 config:

```yaml
model:
  train_vision_backbone: true      # 解冻 ViT
training:
  vision_learning_rate: 2.0e-6     # ViT 单独 lr,比 LLM 小 10 倍
```

**已实现**:`SFTAutoVLA.configure_optimizers`([autovla.py](../../models/autovla.py))现在按 `vlm.visual` 拆分 param group。
`vision_learning_rate` **不设时默认回退到主 lr,行为与改动前完全一致**。

实测:

| 设置 | param groups |
|---|---|
| `train_vision_backbone: true` + `vision_learning_rate: 2e-6` | **2 组**:ViT **669M @ 2e-6**,其余 **3090M @ 2e-5** |
| 默认(不设) | **1 组**,行为不变 |

启动时打印 `[optim] vision tower trainable: 669M params @ lr=2e-06 | rest: 3090M @ lr=2e-05`。

> LambdaLR 的 warmup 会把两组**按同比例**缩放(step 0 时 ×0.05),10:1 的比例被正确保持。

**为什么这里必须分层,而 §2 的 LoRA 方案不用**:此处 LLM 和 ViT **都是全参**,共用 2e-5 对 669M 预训练 ViT 偏大(runbook §8(a));LoRA 方案靠 B 零初始化自动形成不对称,不需要手动分层(见 §2.3)。

⚠️ **显存**:[run_sft.py:75](../../tools/run_sft.py) 的 gradient checkpointing **只覆盖 LLM,不覆盖 ViT**。解冻后 batch=2 会留着 24 张图的 ViT 激活。当前 44G/97G 有余量,但这是主要风险点。

⚠️ **实验设计**:ViT 解冻是 reasoning 归因的混淆变量,建议当独立消融(同 §2.5)。

---

## 2. ViT 解冻 + LLM LoRA(设计,未实现)

即 **SimLingo 配方**,与 AutoVLA 原配方正好相反:

| | vision encoder | LLM |
|---|---|---|
| SimLingo | **全量训练** | **LoRA** r=16 |
| AutoVLA SFT(当前) | **冻结** | **全参** |
| 本节目标 | **解冻** | **LoRA** |

### 2.1 ⚠️ 致命坑:LoRA 学不到 action token(会静默毁掉训练)

- `resize_token_embeddings` 给 2048 个新 `<action_i>` token **随机初始化**(实测 embed 153713 行 vs base vocab 151665)
- LoRA 只打 `q/k/v/o_proj`,**碰不到 `embed_tokens`**
- Qwen2.5-VL-3B 的 `tie_word_embeddings: true`(已从 config.json 确认)→ **读写共用同一张表** → 输出端那 2048 行也是随机的
- **结果:模型永远学不会输出动作,且不报任何错**

**解法**:`modules_to_save=["embed_tokens"]`(因 tied,一次覆盖输入输出两端)

> **反证也要记住**:[run_rft.py:126-134](../../tools/run_rft.py) 的 LoRA **没有** `modules_to_save` 却能工作 —— 因为 RFT 从 SFT checkpoint 加载([run_rft.py:120](../../tools/run_rft.py)),embedding 早已学好。
> → **「从零 SFT 用 LoRA」和「在 SFT ckpt 上做 RFT」不是一回事,别照抄 RFT 的 LoRA 配置。**

**推论**:action token 约束的是「embedding 必须可训」,**不是**「LLM 必须全参」。全参的真实理由是 baseline 可比性 + 仓库没有 peft 路径(见 runbook §7.2),不是技术硬约束。

### 2.2 ⚠️ 坑:peft 会悄悄把 ViT 冻回去

`get_peft_model()` 把所有非 LoRA 参数设 `requires_grad=False`,**包括 ViT**。而 `configure_optimizers`([autovla.py](../../models/autovla.py))的逻辑**只会设 False,从不设 True**:

```python
if not self._train_vision_backbone:
    for param in self.autovla.vlm.visual.parameters():
        param.requires_grad = False      # 只有这一个方向
```

→ **`train_vision_backbone: true` 在 LoRA 之后完全无效,ViT 仍然冻着**(静默)。
→ 且 peft 包装后路径变成 `vlm.base_model.model.visual`,原来的 `vlm.visual` 可能直接 AttributeError。

**解法**:在 `get_peft_model()` **之后**显式 `requires_grad=True` 解冻 ViT,并修正模块路径。

### 2.3 学习率:单一 LR 可行(修正早期判断)

早期评估曾说「ViT 2e-6 / LoRA 1e-4~3e-4,差 100 倍必崩一边」—— **这个说法过头了**,已修正:

- LoRA 的 **B 矩阵零初始化**,起步等效更新量 ≈ 0 再缓慢增长
- 同一 LR 下,**LoRA 的实际权重改变量远小于全参**
- 于是「ViT 全参 + LLM LoRA」**自动形成不对称**,天然就是 ViT 主导,**不需要手动分层 LR**

SimLingo 即如此:`lr=3e-5`,ViT 全参 + LLM LoRA r=16。**选 LR 时按 ViT 的需求选**(1e-5 ~ 3e-5 是 ViT 微调正常区间)。

> **原警告的正确适用范围**是 runbook §8(a):**ViT 解冻 + LLM 全参**共用 2e-5 —— 那时两边都是全幅度更新,才真的危险。本节是 LoRA 方案,不适用。

### 2.4 改动清单

| # | 位置 | 改什么 | 风险 |
|---|---|---|---|
| 1 | [run_sft.py](../../tools/run_sft.py) | 加 peft 路径(可抄 [run_rft.py:123-136](../../tools/run_rft.py)) | 低 |
| 2 | LoraConfig | **加 `modules_to_save=["embed_tokens"]`** | ★ **不加必挂,且静默** |
| 3 | peft 包装后 | 显式解冻 ViT + 修正 `visual` 模块路径 | ★ 静默 |
| 4 | [run_sft.py:75](../../tools/run_sft.py) | gradient checkpointing 覆盖 ViT(解冻后 12 张图激活全留着) | 显存 |
| 5 | LR | **不需要分层**(见 §2.3),按 ViT 选单一 LR | — |
| 6 | FSDP wrap policy | runbook §8(b) 说要加 `Qwen2_5_VLVisionBlock` —— **我们已切 DDP,此条不适用** ✅ | 无 |

**显存**:DDP + batch=2 当前只占 44G/97G,解冻 ViT 有余量。
**DDP**:已设 `find_unused_parameters=True`,能容纳冻结/未用参数,无需额外改动。

### 2.5 实验设计提醒

主线目标是「用 reasoning 提升 performance」,**baseline 必须同配方才能归因**。

同时改「ViT 冻结→解冻」**和**「LLM 全参→LoRA」= 一次动两个变量,后续 CoT 涨点将无法归因。

→ **建议**:先出 no-CoT 全参 baseline,再把「ViT 解冻 / LoRA」作为**独立的后置消融**。

---

## 3. 速查

| 想做 | 状态 | 入口 |
|---|---|---|
| 换 Qwen3-VL-2B/4B | ✅ 已实现 | `pip install -U 'transformers>=4.57'` + `config/training/qwen3-vl-*B-nuplan-nocot-sft.yaml` |
| 验证移植是否成功 | ✅ 已实现 | `python logs/inspect_sample.py`(查 720 / action id / 32) |
| 推导新 backbone 的 token id | ✅ 已实现 | `python logs/derive_token_ids.py <model_path>` |
| **ViT 解冻(LLM 仍全参)** | ✅ **已实现** | `train_vision_backbone: true` + `vision_learning_rate: 2.0e-6`(见 §1.7) |
| ViT 解冻 + LLM LoRA | ⚠️ 未实现 | 按 §2.4 清单改,**务必先看 §2.1** |
