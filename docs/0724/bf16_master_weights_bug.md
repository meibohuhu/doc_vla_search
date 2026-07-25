# SFT 训练不收敛的根因：bf16 master weights

> 一句话：模型以 `torch_dtype=bfloat16` 加载，Lightning 的 `32-true` / `bf16-mixed` 都**不会**把它转回 fp32，
> 于是 AdamW 直接在 bf16 参数上更新；`lr=2e-5` 的单步更新量低于 bf16 的分辨率，被反复舍入成 0。
> **实测 200 步后 74% 的参数一次都没变过。**
> 日期：2026-07-24 · 相关：[data_pipeline.md](../721/data_pipeline.md)

---

## 0. 症状与最终结论

| | |
|---|---|
| **表面症状** | no-CoT SFT 训完 PDMS 只有 **60-62**，预期（论文 action-only@100k）**71** |
| **误导性信号** | `val_loss = 1.05` 且在下降 → 看起来"在正常收敛" |
| **真实情况** | action token 的 loss 卡在 **3.32**（困惑度 27），**训练集** top-1 仅 **17.3%** |
| **根因** | bf16 master weights，参数实际没在更新 |
| **验证** | 同数据同 epoch 的 A/B：fp32 → val_loss **0.005**，bf16 → **1.56**（差 ~10 倍） |

---

## 1. 排查过程（先排除了什么）

按怀疑度依次排查，**全部排除**后才找到真因。记录下来是因为这些检查本身可复用：

> **top1** = action token 的 top-1 精确命中率：模型每步从 2048 个 action token 里选一个，
> 共 10 步，`top1` 是 `argmax(预测)==GT` 的比例（随机猜 = 1/2048 ≈ 0.05%）。
> 本文两种口径都出现过：**teacher-forcing**（喂 GT 前缀，单步能力，如 16.3%）与
> **自回归**（模型自己逐步生成，有误差累积，如 7.7%）。用它而非 loss，是因为它只统计
> action token、不被模板稀释；局限是 codebook 相邻编号是相似运动基元，"猜到隔壁"和
> "猜错十万八千里"在 top1 上没区别——所以本文同时用 ADE 交叉印证（2.65m vs 发布 0.23m）。

| # | 检查项 | 方法 | 结果 |
|---|---|---|---|
| 1 | checkpoint 是否正确加载 | 检查 state_dict key 结构、embed 尺寸 | ✅ `autovla.vlm.*`→`vlm.*`，embed 153713 = 151665+2048 |
| 2 | 模型输出格式 | dump 原始生成文本 | ✅ `<answer>` + 恰好 10 个 action token，无 `<think>` |
| 3 | eval config 的 `use_cot` | 与训练 config 比对 | ✅ 均为 False |
| 4 | 图像预处理是否一致 | 对真实 1920×1080 跑 `smart_resize` | ✅ 训练/评测都是 **420×224 / 120 token** |
| 5 | **训练标签是否正确** | GT轨迹 → action token → 还原，比对误差 | ✅ 平均 **0.048 m**，误差>1m 的 0/200 |
| 6 | 标签 token 分布 | 统计均值/分位 | ✅ 均值 601，与发布模型输出（622）吻合 |
| 7 | 输入特征 train vs test | 速度/加速度/位移/指令分布 | ✅ 基本一致（4.62 vs 5.15 m/s） |
| 8 | 有效 batch / lr | 计算 `bs × accum × ngpu` | ✅ **32**，与论文一致；lr 2e-5 为作者建议值 |
| 9 | batch=2 的 padding 污染 | 实测 batch 内序列长度与 pad token 占比 | ✅ 序列长 970/971，pad 仅占 loss 的 1.5% |
| 10 | HF 的 loss 计算 | HF 内部 loss vs 手算逐位对拍 | ✅ 差 **0.000000** |
| 11 | 视觉是否真被使用 | 换成另一场景的图，看预测是否改变 | ✅ 6 个样本里 5 个改变 |
| 12 | 欠拟合 vs 过拟合 | 在**训练集**上测 top-1 | ❌ 17.3% ≈ 验证集 16.3% → **纯欠拟合** |

第 12 项是转折点：**连训练数据都拟合不了**，排除了"数据不够/太难"，指向优化本身。

---

## 2. 关键发现一：`val_loss` 是假象

监督 token 只有 32 个/样本，其中 **10 个 action + 22 个固定模板**：

```
仅 action token 的 loss  : 3.3193   ← 真正要学的
仅模板  token 的 loss    : 0.0002   ← 早就学完了
全部监督 token 平均      : 1.0374   ← 这就是你看到的 val_loss

验算: (10 × 3.3193 + 22 × 0.0002) / 32 = 1.0374  ✅
```

**`val_loss` 收敛到 1.0 ⟺ action loss 收敛到 3.3**，这是同一件事。
模板（`<answer>The final output action is:` 等）占 69%，几步就学到 ~0，把平均值拽了下来。

> ⚠️ 直接后果：**不能用 `val_loss` 判断收敛，也不能用它选 checkpoint。**

三个 ckpt 的扫描证实已经平台化：

| ckpt | all_loss | **action_loss** | ppl | **top1** |
|---|---|---|---|---|
| epoch=2 | 1.0737 | 3.4358 | 31.1 | 16.2% |
| epoch=3 | 1.0452 | 3.3446 | 28.3 | 15.7% |
| epoch=4 | 1.0374 | 3.3193 | 27.6 | **16.3%** |

top-1 三个 epoch 纹丝不动（16.2 → 15.7 → 16.3，是噪声）。

---

## 3. 关键发现二：根因是 bf16 master weights

### 机制

```python
# models/autovla.py:480
Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16)
#                                                             ↑ 参数是 bf16
```

Lightning 2.2.1 各 precision 的 `convert_module` 行为（**实测**）：

| precision | convert_module | 结果 |
|---|---|---|
| `32-true` | **no-op** | 模型保持 bf16 |
| `bf16-mixed` | **no-op** | 模型保持 bf16 |
| `bf16-true` | 转 bf16 | 模型保持 bf16 |

**三种都不会把 bf16 模型转回 fp32。** 于是 AdamW 的更新对象是 bf16 张量。

bf16 只有 8 位尾数 → 相对精度 ≈ **0.39%**。而 `lr=2e-5` 时单步更新相对参数大小约 **0.1%** —— **低于分辨率，舍入成 0。**

### 数值实验

取真实 ckpt 里 `layers.10.mlp.down_proj.weight`（|w| 均值 0.019）做 200 步 AdamW：

| | 参数发生变化的比例 | 200 步累积位移 |
|---|---|---|
| **bf16 master** | **25.9%** | 3.47e-05 |
| **fp32 master** | **100.0%** | 2.32e-04（**6.7×**） |

**74% 的参数在 200 步里一次都没动过。** 不是"训得慢"，是大部分网络被冻结。

### 这解释了所有观测

| 观测 | 解释 |
|---|---|
| 模板 loss → 0.0002 | 早期大梯度阶段几步就学完 |
| action loss 卡在 3.32 三个 epoch | 需要精细更新，全被舍入 |
| train_loss ≈ val_loss，不过拟合 | 参数不动，自然不会过拟合 |
| 训练集 top-1 也只有 17.3% | 连记忆都做不到 |
| ADE 2.65 m（发布 ckpt 0.23 m） | 从未真正收敛 |
| 预测塌缩到少数高频 token（5、534 反复） | 只学到边缘分布 |

---

## 4. Smoke test：500 样本过拟合 A/B

### 设计

用**同一批 500 个样本、同样 40 epochs、train=val**（测的就是记忆能力），
**唯一变量是精度**。lr、ViT 冻结、batch_size 全部保持与主训练一致，避免引入混淆变量。

配置：[`config/training/0723/overfit500.yaml`](../../config/training/0723/overfit500.yaml)（fp32）
· [`overfit500_bf16ctrl.yaml`](../../config/training/0723/overfit500_bf16ctrl.yaml)（bf16 对照）

### 结果

```
epoch:     0     5     10    15    20    39
fp32 :   2.62  1.62  1.16  0.67  0.16  0.005   ← 完全记住
bf16 :   3.41  1.90  1.85  1.71  1.56    —     ← 蠕行
```

**Epoch 20 时相差 ~10 倍（0.161 vs 1.564）。**

bf16 的平台特征很典型：`ep3=1.945 → ep9=1.867`，6 个 epoch 只降 0.078；之后以约 −0.03/epoch 蠕行
（fp32 同期是 −0.10/epoch 并持续加速）。换个角度：**bf16 跑 20 个 epoch 才到达 fp32 第 5-6 个 epoch 的水平**。

> 注意 bf16 并非完全冻结（仍有 ~25% 参数在动），所以 loss 仍在缓慢下降——
> 这与数值实验的 25.9% 吻合。它会停在半路，而不是彻底不动。

### 为什么必须做这个对照

最初的比较是 **101k 数据/5 epoch/bf16** vs **500 数据/40 epoch/fp32** —— 三个变量同时变，
不能把改善归因给精度。补上控制变量后归因才成立。

---

## 5. 作者为什么"没有"这个问题

**他也有。** 实测确认（`world_size=2`，`FULL_SHARD`，复刻原始 FSDP 路径）：

```
flat param dtype = torch.bfloat16    ← optimizer 更新对象
200 步后: 参数变化比例 24.3%          ← 与 DDP+bf16 的 25.9% 基本一致
```

`MixedPrecision(param_dtype=bf16)` 的语义是"master 保持**原** dtype、计算时转 param_dtype"，
而原 dtype 就是 bf16。**FSDP 和 DDP 殊途同归。**

那他怎么拿到 80.54 的？三种可能，按可能性排序：

1. **带着残疾硬训出来的**（我倾向这个）——bf16 仍有 ~25% 参数在动，作者数据是 1.6 倍（166k vs 101k）、
   总步数 1.37 倍，靠数据量和步数把有效容量压榨出来。与对照实验一致：bf16 组也在降，只是慢得多。
2. 发布代码与实际实验版本不一致（论文代码库常见）。
3. 我漏了某个环节（如实际脚本里另有 `.float()`，或用了语义不同的 Lightning 版本——我只验了 2.2.1）。

**推论**：若 (1) 成立，修复后在同样数据量下可能**超过**论文数字。但这只是推论，需实测。

### ⚠️ 两处曾被我当作证据、后被推翻的推断

| 曾经的说法 | 为什么错 |
|---|---|
| "发布 ckpt 是 fp32 ⇒ 作者用了 fp32 master" | 那个 fp32 是 README 所述 **LoRA merge 后处理**的产物，不能反推训练时的精度 |
| "模板 token 稀释了 action 的梯度" | 模板 loss ≈ 0 ⇒ 梯度也 ≈ 0，**不抢梯度**；真实影响只是分母从 10 变成 32（等效 lr × 0.31） |

记下来是因为：**用工件（artifact）反推过程是不可靠的，要用直接实验。**

---

## 6. 修复

### 代码改动

**[`tools/run_sft.py`](../../tools/run_sft.py)**

```python
# 建 strategy 之前，显式转 fp32
if config['training'].get('fp32_master', True):
    model = model.float()
    print("[precision] master weights -> fp32 (计算仍为 bf16)")

_pd = next(model.parameters()).dtype
print(f"[precision] strategy={strategy_name}  实际 param dtype={_pd}")
if _pd != torch.float32:
    print("[precision] ⚠️  master 不是 fp32，小幅更新会被舍入，训练可能静默失效！")
```

DDP 分支的 precision 与该开关联动：

```python
# fp32_master=True  -> "bf16-mixed"：fp32 master + bf16 autocast
# fp32_master=False -> "bf16-true" ：还原修复前行为，仅用于精度对照实验
trainer_precision = "bf16-mixed" if config['training'].get('fp32_master', True) else "bf16-true"
```

FSDP 分支**无需改代码** —— 模型转 fp32 后，`MixedPrecision(param_dtype=bf16)` 自动变成
"fp32 master + bf16 计算"。已实测：flat param dtype = fp32，200 步 100% 参数在更新。

**[`models/autovla.py`](../../models/autovla.py)** —— 把真正要看的指标记录出来：

```python
self.log("train_action_loss", ...)   # 只算 10 个 action token
self.log("train_action_top1",  ...)  # 精确命中率，最直观
self.log("val_action_loss",   ...)
self.log("val_action_top1",   ...)
```

`prog_bar=True`，进度条直接可见，同时进 `metrics.csv` / wandb。

### 代价

单卡显存 **43 GB → 74.8 GB**（fp32 参数 +12 GB、AdamW 状态 +24 GB）。
80 GB 卡可行但余量只剩 5 GB；如遇 OOM：`batch_size` 降到 1、`accumulate_grad_batches` 提到 8，
保持有效 batch 仍为 32。

（显存为什么翻倍的详细拆解见 §6.5。）

---

## 6.5 为什么 fp32 显存会翻倍——按每参数字节数拆开

设可训练参数 N（本模型 LLM 全参 ≈ 3.1B）。训练一个参数要占的显存分四块：

| 占用项 | bf16-true（修复前） | fp32 master（修复后） |
|---|---|---|
| **模型权重**（前向/反向用） | 2N | 2N（bf16 计算副本，没变） |
| **梯度** | 2N | 2N（bf16，没变） |
| **权重 master 副本** | —（就用上面那份 bf16） | **+4N**（fp32 主副本，新增） |
| **AdamW 状态 m**（一阶动量） | 2N | **4N**（fp32，翻倍） |
| **AdamW 状态 v**（二阶动量） | 2N | **4N**（fp32，翻倍） |
| **合计** | **8N** | **16N** |

**恰好翻倍：8N → 16N。** 对 N=3.1B：`8×3.1 = 25GB` → `16×3.1 = 50GB`（纯参数相关），
再加激活/KV/分配器缓存，实测 43GB → 75GB。

### 三点直觉

1. **占大头的是优化器状态，不是权重。** AdamW 每个参数要存两个动量 buffer（m、v），
   它们从 bf16 变 fp32，是 `2N+2N → 4N+4N` 这 4N 的增量，比"多存一份 fp32 权重"(4N) 还大。
   **换句话说：一半的新增显存是 Adam 的动量,不是权重本身。**
2. **前向/反向的显存没变。** 计算仍走 bf16 autocast，激活、KV 都还是 bf16——
   这也是为什么速度几乎没退化。变重的只是"优化器要精确记账"这部分。
3. **为什么必须 fp32 存这些**：舍入发生在 `param -= lr·m̂/(√v̂+ε)` 这一步。
   要让 2e-5 量级的小更新累积得住，master 权重和动量都得有 fp32 的分辨率——
   任何一个是 bf16，小更新就在那一步被吞掉（§3 的数值实验）。

### FSDP 能把这笔账摊薄——这才是作者用 8 卡的原因

DDP 参数**各卡各存一份**，所以 fp32 master 在每卡都是完整的 16N。
FSDP FULL_SHARD 把这 16N **分片到所有卡**：8 卡 → 每卡 16N/8 = 2N。
所以作者 L40S 48GB 能装下 fp32 master——**不是因为他用了 bf16 省显存，而是因为分片摊薄了**。

| 配置 | fp32 master 每卡显存(参数相关) |
|---|---|
| DDP（我们，不分片） | 16N ≈ 50GB |
| FSDP FULL_SHARD / 4 卡 | 16N/4 ≈ 12.5GB |
| FSDP FULL_SHARD / 8 卡 | 16N/8 ≈ 6GB |

**如果你觉得 GPU 占用太多，正解不是退回 bf16，而是改用 FSDP（把显存分片），或增加数据并行卡数。**
config 里 `strategy: fsdp` 即可切换——注意 FSDP 通信开销更大（每层前向要 all-gather 参数），
在没有 NVLink 的机器上会更慢，这是"省显存 vs 更快"的取舍。

---

## 6.6 修复后的实测：60 → 80 分

### 全量 navtest（12,126 场景，seed=42，唯一差精度）

| checkpoint | PDMS | NC | DAC | EP | TTC | C | DDC |
|---|---|---|---|---|---|---|---|
| **bf16 epoch4**（坏跑） | **60.54** | 91.90 | 74.75 | 54.89 | 83.45 | 100 | 89.12 |
| **fp32 epoch4**（修复） | **80.06** | 96.79 | 89.34 | 75.30 | 91.52 | 99.98 | 96.54 |
| 发布 RFT ckpt | 89.48 | 99.41 | 95.89 | 82.69 | 97.53 | 99.88 | 98.14 |

**修复值 +19.52 分**（60.54 → 80.06），全量、同场景、唯一变量是精度。涨分全在两项：

```
EP :  54.89 → 75.30   (+20.4)   ← 不再保守慢开
DAC:  74.75 → 89.34   (+14.6)   ← 冲出可行驶区域大幅减少
```

### 1000 抽样（seed=42）——训练动态与全量的一致性

四个 ckpt 放进逐字节相同的 1000 场景，用来看训练动态（哪个 epoch 最优）：

| checkpoint | PDMS | DAC | EP |
|---|---|---|---|
| bf16 epoch4（坏跑） | 58.97 | 73.07 | 54.02 |
| fp32 epoch2 | 78.46 | 87.19 | 73.47 |
| fp32 epoch3 | 79.39 | 88.59 | 74.92 |
| **fp32 epoch4** | **79.76** | 88.29 | 75.53 |

**1000 抽样与全量高度吻合**（fp32: 79.76→80.06；bf16: 58.97→60.54，差 0.3–1.6 分），
说明 seed=42 的 1000 抽样是很好的快速代理，日常迭代用抽样即可，出正式数字才跑全量。

三个衍生结论：

1. **超过论文 action-only@100k(71) 约 8.8 分，逼近论文 SFT-only(80.54，用 166k+CoT)只差 0.78。**
   我们用**少 39% 的数据、纯 action-only** 做到——强烈支持 §5 的假说"作者也带着 bf16 残疾，
   修了精度就能用更少数据反超"。
2. **PDMS 单调上升，epoch4 最优（79.76）——但它的 val_action_loss 反而比 epoch3 回升了。**
   `val_action_loss: 2.33 → 2.18(ep3) → 2.23(ep4)`，而 PDMS `78.46 → 79.39 → 79.76`。
   **过拟合期 loss 与实际性能背离**（模型更自信：对的更笃定、错的也重罚，argmax 却更准）。
   → **选 checkpoint 要看 PDMS/top-1，不能看 val_loss**（这是本项目第二次独立验证该结论，
   第一次是 §2 的模板稀释）。
3. **离发布 ckpt(89.06)还差 9.3**，短板仍是 DAC(88→96)和 EP(75→83)——
   但发布版有 CoT 训练 + 166k 数据 + RFT 三重加成，不是同一起跑线。

---

## 7. 教训

1. **不要用被稀释的平均 loss 判断收敛。** 监督信号里 69% 是早已学会的模板，
   `val_loss=1.05` 与 `action_loss=3.32` 是同一件事的两种说法，前者会骗人。
2. **精度要显式打印。** 这个 bug 藏了整整一轮训练（20+ 小时），
   仅仅因为**没有任何地方显示过参数是什么 dtype**。现已永久加上启动打印 + 告警。
3. **A/B 必须控制变量。** 最初的"修复后变好了"结论建立在三变量同变的比较上，
   不成立；补对照组后才算数。
4. **不要用工件反推过程。** "发布 ckpt 是 fp32" 不能推出"训练用了 fp32"——
   中间隔着一次 LoRA merge。
5. **`load_state_dict(strict=False)` 会静默吞掉不匹配。** 评测前应主动核对
   key 结构和 embedding 尺寸，否则可能拿随机权重跑出"很差但不报错"的分数。

---

## 8. 复现命令

```bash
# fp32 组
CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/run_sft.py --config training/0723/overfit500

# bf16 对照组（同数据同 epoch，只切精度）
CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/run_sft.py --config training/0723/overfit500_bf16ctrl

# 离线测任意 ckpt 的 action loss / top-1（不依赖训练日志）
python /tmp/action_loss_sweep.py          # 改里面的 ckpt 路径

# 对比两次的 val_loss 曲线
python -c "
import pandas as pd, glob
for d in sorted(glob.glob('/data/autovla_data/checkpoints/sft/2026-07-24_*')):
    f = glob.glob(d+'/lightning_logs/version_*/metrics.csv')[0]
    v = pd.read_csv(f); v = v[v.val_loss.notna()][['epoch','val_loss']].drop_duplicates('epoch')
    print(d.split('/')[-1], ' '.join(f'ep{int(r.epoch)}={r.val_loss:.3f}' for _,r in v.iterrows()))
"
```

---

## 9. 下一步

1. **重训主实验**（navtrain 101k，`fp32_master: true`）—— 训练中盯 `val_action_top1`，
   爬到 50%+ 说明真的在学；若又卡在 20% 以下，说明精度之外还有问题，当场就能看见。
2. **`action_loss` 暂不加。** 它不是 bug 而是超参（等效 action 头 lr × 4）；
   过拟合测试证明不加也能学到 val_loss 0.005。一次只动一个变量——
   若第一步后 action loss 仍居高，再作为第二个实验加入。
3. 若修复后仍显著低于 71，下一个怀疑对象是**训练数据口径**：
   你用 navtrain 精选 103k（纯 nuPlan），论文用 trainval `fi=4` 采样 ~100k + nuScenes 混合。
