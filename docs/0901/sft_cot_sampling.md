# SFT 的 CoT 采样与 loss 改造

`dataset_utils/sft_dataset.py` + `models/autovla.py`
2026-09-02 · 备份 `sft_dataset.py.bak_0902` / `autovla.py.bak_0902` · 基线 = git HEAD

配套数据：`/data/autovla_data/nuplan/navtrain_cot`（101,288）、`navtrain_cot_val`（2,000）、
`navtest_cot`（12,146），生成流程见 `docs/0806/vcrd_flow_autovla.md` 与
`logs/0901/navsim_reasoning_bytype*.md`。

跑起来：`bash logs/0902/run_cot_sft_0902.sh`
（config `training/qwen2.5-vl-3B-nuplan-cot-sft-navtrain-brev`）

**验收标准（用户 2026-09-02 定的）：action 指标不降即可，不要求涨。**
理由:这一步的目的是把 reasoning 这条通路建起来给后面的 RL 用，
SFT 阶段 reasoning 带不带来轨迹收益不是这一步要证明的事。

---

## 1. 为什么要改

原版决定走不走 `<think>` 的依据是**这条样本有没有 CoT 数据**：

```python
has_cot = False
if isinstance(gt_cot, str):
    "<think>\nThis is a complex scenario requiring additional reasoning.\n{gt_cot}\n</think>\n<answer>…"
    has_cot = True
else:
    "<think>\nThis is a straightforward scenario, and a direct decision can be made.\n</think>\n<answer>…"
```

三个问题，前两个是致命的：

**① prompt 侧完全不区分。** 两种样本的 user prompt 一模一样，模型没有任何输入信号能判断
该走哪条分支，于是必然塌向 token 数最少、loss 最容易降的那条。
后果已经记录在 `models/autovla.py:820`：发布的 RFT ckpt 在 navtest 上 **100% 走 fast thinking**，
`<think>` 里恒为 "This is a straightforward scenario…"，一次真实推理都不做。

**② 用"有没有数据"当条件，等于把数据缺失编码成了语义。**
我们现在 98.9% 的 token 都有 reasoning（缺的 1.1% 是 log 首尾凑不出 14 帧窗口），
按原逻辑几乎全部走 think 分支，相当于 `cot_ratio = 0.989`——没有 direct 档做对照，
也没有任何一档在教"给定 reasoning → 产出轨迹"。

**③ 两句固定的 "This is a complex/straightforward scenario…"** 是纯噪声前缀，
且它正好是 fast-thinking 塌陷时被复读的那句。一并删掉。

参照 simlingo 的做法（`simlingo_training/dataloader/dataset_driving.py:312` 起）：
**每条样本随机抽模式，且 prompt 侧同步区分**。

---

## 2. 改成什么样

三档，比例对齐 simlingo 的 `commentary_ratio: 0.75`（其中 in-prompt 占 18%）：

| mode | simlingo | **本项目实际采用** | user prompt 尾 | assistant target |
|---|---|---|---|---|
| `reason_then_act` | 61.5% | **75.0%** | `… next five seconds. Explain your reasoning first.` | `<think>{cot}</think>` + `<answer>…</answer>` |
| `reasoning_given` | 13.5% | **0%（关掉）** | `… next five seconds. Here is the situation: {cot} Answer directly.` | 只有 `<answer>…</answer>` |
| `act_directly` | 25.0% | **25.0%** | `… next five seconds. Answer directly.` | 只有 `<answer>…</answer>` |

即 `cot_ratio: 0.75` + `cot_in_prompt_ratio: 0.0`。代码三档都在，改一个数就能开。

### 为什么关掉 `reasoning_given`

我一开始主张必须留，给了两条理由，**逐条查证后都不成立**：

- ❌ *"waypoint head 会绕开 reasoning"* —— 那是 simlingo 的架构问题（它有独立的
  waypoint head）。AutoVLA **没有任何 head**：动作就是词表 token
  （`labels >= action_start_id`），和 `<think>` 在同一条自回归流里，loss 也是 token CE。
  旁路在架构层面不存在。
- ❌ *"§7 的因果验证需要它才在分布内"* —— 用 prefill 就行：把改写后的 reasoning 填进
  assistant 的 `<think>` 再看动作变不变，这本来就是 `reason_then_act` 的训练分布
  （`models/autovla.py:820` 那段注释就是这么干的）。

留下的唯一实打实的代价是:那 13.5% 的样本不训练"生成 reasoning"这个技能，
而这恰恰是 DriveDiRL 要用 RL 去提升的技能。所以关掉。

### 为什么 `cot_ratio` 取 0.75 而不是 0.615 / 0.67

删掉 in_prompt 之后它的份额要么并进 reasoning 侧、要么并进 direct 侧，两个锚：

- **0.75** —— 保住 `direct = 25%`，和 simlingo 的 `only_trajectory` 一字不差。
  in_prompt 是个 *reasoning* 条件，份额留在 reasoning 侧。**采用这个。**
- **0.615** —— 保住 `reason = 61.5%`。理由是 in_prompt 的 *target* 和 act_directly
  完全一样（都只有 `<answer>`）。

`0.67` 两个锚都不占，论文里那句脚注没法写。而且这个区间里差别多半测不出来——
reasoning 里 **36.5% 只有一句决策**（`I should speed up and bear left: <PLAN>…`），
挪动的那 13.5% 有三分之一是这种"把标签念一遍"。

```
submode 分布 (n=100222)          带场景描述
  none         67789   67.6%      46.1%
  stay_behind  29525   29.5%     100.0%
  其余 4 类     2908    2.9%     100.0%
全体带场景描述 63.5%   纯决策一句话 36.5%
```

⚠️ 别为了提高这个比例去给那 36.5% 编理由。之前量过静态障碍物（锥桶/护栏）在 14.2%
的帧里存在却只能解释 10% 的无理由减速，硬套会在 90% 的情况下造出假因果。

---

## 3. 具体改了哪几处

### (a) `__init__`：新增两个参数

```python
self.cot_ratio = float(model_config.get('cot_ratio', 0.75))
self.cot_in_prompt_ratio = float(model_config.get('cot_in_prompt_ratio', 0.18))
```

config 的 `model:` 段里配：

```yaml
model:
  use_cot: true
  cot_ratio: 0.75              # 有 reasoning 参与的比例（其余为 act_directly）
  cot_in_prompt_ratio: 0.0     # 上者之中，reasoning 给在 prompt 里的比例（本项目关掉，见 §2）
```

消融：`cot_ratio: 1.0, cot_in_prompt_ratio: 0.0` → 全部 `reason_then_act`；
`cot_ratio: 0.0` → 全部 direct，但 prompt 仍带 "Answer directly." 后缀
（和 `use_cot: false` 不同，后者连后缀都没有）。

### (b) nuplan/waymo 分支：抽模式

```python
_avail = bool(isinstance(gt_cot, str) and gt_cot)
_use = _avail and random.random() < self.cot_ratio
cot_mode = ("reasoning_given"
            if _use and random.random() < self.cot_in_prompt_ratio
            else "reason_then_act" if _use else "act_directly")
has_cot = (cot_mode == "reason_then_act")
```

非 `reason_then_act` 的两档 **不写空的 `<think>`**——写了等于教模型"想一下但什么都不说"。

### (c) user prompt 尾部按模式加后缀

```python
f"… plan the action trajectory for the autonomous vehicle over the next five seconds. "
# ⚠️ 只有开了 CoT 训练才加后缀。use_cot=false 的纯 action 训练必须与改动前逐字节一致。
+ ("" if not self.using_cot else
   ("Explain your reasoning first."          if cot_mode == "reason_then_act" else
    f"Here is the situation: {gt_cot} Answer directly." if cot_mode == "reasoning_given" else
    "Answer directly."))
```

### (d) CoT 分支的 system prompt 重写

原文有三处和本项目的设定冲突：

- 让模型"考虑 traffic lights、lane markings" —— 红绿灯已整条砍掉（判别不可靠，见
  `logs/0901` 的排查记录），车道线我们从来不说。**承诺了 target 里不存在的内容。**
- `"If necessary, use step-by-step reasoning … Otherwise, you may directly predict"`
  —— 把要不要推理交给模型自己判断，而我们已在 user prompt 里**逐样本指定**模式，两者打架；
  "让模型自己决定"正是 fast-thinking 塌陷的成因。
- `"step-by-step reasoning"` —— 我们的 reasoning 是一句话（p50 = 32 token），
  不是分步推理，措辞会把预期设错。

改为：

```
You are an autonomous driving system.
You receive camera observations from the ego vehicle and its current dynamic state.
Your task is to predict the driving action for the next five seconds.

When asked to explain, say briefly what constrains you right now and what you will do.
Then give the final action.
```

### (e) 样本打印开关

```bash
PRINT_SFT_DEBUG=2 bash logs/0902/run_cot_sft_0902.sh
```

每个 dataloader worker 打印前 N 条（总量 = N x num_workers x GPU 数），只打印不改行为。
输出 mode / has_cot / token / SYSTEM / USER / TARGET，用来肉眼确认三档采样和 prompt 后缀。

---

## 3.5 🔴 后续:文案抽成共用模块(2026-09-02 同日)

上面 (c)(d) 改的是 `sft_dataset.py` 里的那一份。**RL 侧还有独立的另一份**
(`models/autovla.py::get_prompt`),已经漂到旧版 —— 连撇号都是 `’` 而不是 `'`。
prompt 一漂,RL rollout 就跑在训练分布外,而这种偏差不报错、只让指标莫名其妙。

现在两边都从 **`dataset_utils/prompt_spec.py`** 取(`system_text` / `user_text`),
那是唯一出处。RL 一律传 `cot_mode="reason_then_act"` —— 方法要改写的就是那句 `<PLAN>`,
模型必须先把它说出来。

**顺带修掉一个 (c) 引入的字节差异**:重构前 `base` 写成 `"…next five seconds. "`
(带尾空格)再拼后缀,于是 `use_cot: false` 的 prompt 也多了一个尾空格 ——
和改动前**不是**逐字节一致。现在 `base` 以 `"…next five seconds."` 结束(无尾空格),
后缀自带前导空格,三档 CoT prompt 一字未变,而纯 action 路径回到与原文完全相同。
已用 `.bak_0902` 逐字节比对确认。

---

## 4. 不受影响的部分

**`sft_dataset.py` 里 `use_cot: false` 的路径与改动前逐字节一致**（用户明确要求：
"我需要你不要影响之前只 train action 的方式，通过设置参数的方式调节 train 的方式"）。
与 `.bak_0902` 逐字节比对确认:`using_cot=False` 时 system prompt 与 user prompt
都和改动前**完全相同**，assistant target 一个字符没变。
（🔴 中途曾因为 base 多了一个尾空格而不成立，见 §3.5 —— 已修。）

🔴 **但 §7 的 loss 改动是全局的，`use_cot: false` 也会走到**（`loss` 多了 `action_loss`
一项）。已跑完的 no-CoT 基线用现在的代码复现不出来。这一点在 §7 末尾展开。

nuscenes 分支未动（只补了一行 `cot_mode = "act_directly"` 防 `NameError`）。

simlingo 的 `use_cfg` / `cfg_dropout_prob`（随机 drop 掉 `Command:` 选项）**没有移植**，
`instruction` 始终原样进 prompt。

---

## 5. 验证

`navtrain_cot` 随机 5000 条实跑采样（当时 `cot_in_prompt_ratio` 还是 0.18）：

```
reason_then_act   61.5%   (目标 61.5%)
reasoning_given   13.0%   (目标 13.5%)
act_directly      25.5%   (目标 25.0%)
```

关掉 in_prompt 后（0.75 / 0.0），启动脚本会从 config 现算比例打在 banner 上：

```
采样 : reason_then_act 75.0% / reasoning_given 0.0% / act_directly 25.0%
```

⚠️ 这行以前是**写死**的 `61.5/13.5/25`，改了 config 也不变，读 log 的人会被骗。
已改成从 yaml 现算。

---

## 6. `<PLAN>` 标签泄漏多少？——量过，很少

reasoning 结尾带着 `<PLAN>STOP,STRAIGHT</PLAN>`。我一开始担心这在 `reasoning_given`
那档等于把答案告诉模型，**量完发现不成立**：

```
first action token 种类 791   H(action) = 8.59 bit
只看 <PLAN> 猜 action:  top1 = 5.0%   (多数类基线 3.2%)
H(action|PLAN) = 6.89 bit   PLAN 只提供 1.70 bit (20%)
  ('STOP','STRAIGHT')   n=1070  格内最常见 action 仅占 3.6%
  ('KEEP','STRAIGHT')   n=1048  格内最常见 action 仅占 1.1%
```

PLAN 是 4x3 的粗标签，action codebook 有近 800 类，泄漏很有限。

反过来说也重要:**它对轨迹的约束力也就 1.7 bit，而这些信息图像里本来就有**。
所以"in-prompt 档能强迫轨迹条件于 reasoning"这个作用比想象的弱得多——
这也是 §2 里关掉那一档的旁证。

---

## 7. loss 改造（`models/autovla.py`）

### 原来长什么样

```python
# # add more penalty for CoT reasoning data
if hascot[0] == True:
    loss = loss * 40           # = 论文里的 lambda_cot
    loss = loss + action_loss  # = 论文里的 lambda_a * L_action
```

**没有 else 分支。** 这段是 AutoVLA 上游原样发布的（commit `60f64b5`），我们没改过。

作者在 [issue #23](https://github.com/ucla-mobility/AutoVLA/issues/23) 里解释了两项的用意：

> "When CoT is present, the action tokens are **heavily diluted by the much longer
> reasoning text** … we explicitly introduce the auxiliary L_action."

> "Because CoT samples are **much scarcer** in the dataset compared to action-only
> samples, we apply a larger weight (lambda_cot) to boost the reasoning loss."

### 为什么必须改

**稀释是真的**（实测 target 长度）：

```
act_directly     target 53 tok，action 10 → 占 19%
reason_then_act  target 90 tok（reasoning 均 37）→ 占 11%    差 1.69x
```

**但"CoT 样本稀少"这个前提在我们这里是反的**——`cot_ratio=0.75`，CoT 是多数档。
照搬 x40 的后果：

- 它**不是全局缩放**，只作用在 75% 的样本上。`accum=8` 累加后，
  那 25% 的 `act_directly` 等于按 1/40 权重参与，基本被丢掉。
- `run_sft.py:253` 用的是 `gradient_clip_algorithm='value'`, `clip_val=1.0`,
  **按值裁剪不是尺度不变的**，放大 40 倍会让大量梯度元素撞截断、方向失真。
- issue 里两位使用者报告的正是这个后果：SFT 完模型只吐 CoT 文本、
  一个 action token 都不生成。

顺带纠正一个想当然:x40 **并没有**在单条样本内部饿死 action。按真实 CE 量级算
（action CE 起点 21.6、text ~2.5），x40 下 action 仍占 74%。伤害只在上面两条。

### 改成什么

取作者给的通式 `L = w_i * L_LM + lambda_a * L_action`，令 `w_i = lambda_a = 1`，两档同等对待：

```python
if action_loss.numel() > 0:
    loss = loss + action_loss
```

论文公式本来就对**所有**样本加 `L_action`，上游代码漏了 no-CoT 分支
（issue 提问者一开始问的就是这个）。

### 各档的有效梯度权重

```
                                  文本tok    +L_action   action tok 总权重
改前  use_cot:false (已跑的基线)    0.0189      0.0000          0.0189
改前  reason_then_act (x40)        0.4444      0.1000          0.5444
改后  use_cot:false                0.0189      0.1000          0.1189
改后  reason_then_act              0.0111      0.1000          0.1111

改前:CoT 档的文本 token 权重 / 基线 action token 权重 = 23.6x   ← 没法比
改后:CoT 档 action 权重 / no-CoT 档 action 权重       = 0.935   ← 两臂对齐
```

### 一个曾经想错的地方

**没有 `L_action` 时 action token 并非没被训练。** `output.loss` 是对所有 target token
的标准 CE，那 10 个 `<action_i>` 一直在里面；`L_action` 只是额外加权。已跑完的基线
（完全没有 `L_action`）就是证据：`train_action_loss` 21.60 → 0.832，`top1` 0 → 0.750。

而且 `1/53` 是**均匀缩放**，AdamW 的 `m/sqrt(v)` 会把它约掉；模板 token 几步就收敛到
loss~0、不再贡献梯度。所以 no-CoT 那条臂上加不加 `L_action` 影响很小——
唯一的机制是 `clip_grad_value` 不是尺度不变的。**因此基线不必重跑。**
（真要写进论文的严格对照，再用同一个 loss 重跑一次基线更干净。）

### reasoning 会不会被饿死

按真实 CE 量级估：

```
                     训练初期 action/reasoning     收敛后
  无 L_action          70.0% / 30.0%            52.9% / 47.1%
  加 L_action          95.9% /  4.1%            91.8% /  8.2%   ← 当前
```

reasoning 只剩 4~8%。**实跑结果是没被饿死**（见 §9）：一个 epoch 后
`val_text_loss = 0.04`，reasoning 基本解完了——因为它高度模板化。所以 `lambda_a = 1` 不用调。

为了能看见这件事，加了两条日志（`train_text_loss` / `val_text_loss`），
统计的是**非 action 的 target token**。

---

## 8. 踩过的三个运维坑

### (a) 按 rank 分支的 `sync_dist` = DDP 死锁

`train_text_loss` 一开始写成这样，**训练在 batch 63 卡死，4 卡 GPU 100% 但一步不走**：

```python
if bool(hascot[0]) and _tm.any():          # ❌
    self.log("train_text_loss", ..., sync_dist=True)
```

`sync_dist=True` 会触发跨 rank 的 all-reduce，而 `has_cot` 是逐样本随机的（75/25），
4 个 rank 全一致的概率只有 `0.75^4 + 0.25^4 ≈ 32%`——有的 rank 进集合通信、有的不进。

改成**无条件记录**，所有 rank 走同一条路径：

```python
_tm = (labels_flat != -100) & (~action_mask)
if _tm.any():                              # ✅ 每个 rank 都成立
    self.log("train_text_loss", ce_loss_all[_tm].mean(), sync_dist=True, ...)
```

代价：`act_directly` 档这里只剩模板 token，会把曲线拉低一点；但 75% 的 step 是 CoT、
模板 loss 很快趋近 0，曲线仍主要反映 reasoning。

**教训：DDP 下任何带 `sync_dist=True` 的 `self.log`，其调用条件必须对所有 rank 相同。**

### (b) 怎么判断训练是不是卡住了

**GPU 利用率不管用**——NCCL 集合通信空转也是 100%，CPU 也照样忙。
终端日志同样不管用：进度条用 `\r` 刷新不带 `\n`，管道里的 stderr 是行缓冲，
画面会停在某一帧看起来像卡住（脚本里已加 `PYTHONUNBUFFERED=1` 缓解）。

**可靠的信号**（wandb 在 Python 进程内捕获 console，不过管道）：

```bash
ls -l runs/sft/<ts>/wandb/run-*/files/output.log      # mtime 不动 = 真卡了
```

再加一条:wandb 的 history 记录数为 0 说明**一个指标点都没上传过**。
`strings run-*.wandb | grep -c train_loss` 会骗人——那是 console 捕获，不是指标。

### (c) metrics.csv 的落盘节奏

| | 条件（Lightning 2.2.1 默认） | = batch | ≈ 时间 |
|---|---|---|---|
| 写进内存 | `log_every_n_steps` = 50 optimizer step | 400 | 4.5 min |
| 落盘 | `flush_logs_every_n_steps` = 100 | 800 | 9 min |

`optimizer step = 8 batch`（accum=8）。刚起跑时目录里只有 `hparams.yaml` 是正常的。

---

## 9. 第一个 epoch 的结果（2026-09-02）

和 no-CoT 基线在**完全相同的 step** 上比。两边 global batch 32、lr 2e-5、ViT 冻结、
同一份 2000 条 val token，每 epoch 3165 个 optimizer step：

| step 3165（epoch 0 末） | no-CoT `2026-07-27_18-05-16` | CoT `2026-09-02_05-47-39` |
|---|---|---|
| `val_action_top1` | 0.21525 | **0.21870** |
| `val_action_loss` | 2.7149 | **2.6766** |
| `val_loss` | 0.8484 | 0.5485 ← **不可比** |

⚠️ `val_loss` 跨两条臂**不能比**：CoT 的 target 多了 37 个 reasoning token，
而 reasoning 学得极好，把 mean-CE 拉低了。只能看 `val_action_*`。

⚠️ 别用 `runs/sft/2026-07-23_04-41-59_bf16` 当基线，它的 CSV 只有 `train_loss/val_loss`
两列（早于 0724 加的 action 指标）。

训练本身健康：

```
train_action_loss  10.28 -> 2.44
train_action_top1   0.00 -> 0.225
train_text_loss     0.66 -> 0.018        val_text_loss 0.0398
```

**结论：action 指标与基线持平（top1 +0.35pp）——按验收标准（不降即可）是通过的。**

两点留给后面看：

1. `val_text_loss = 0.04`，reasoning 一个 epoch 就解完了。§7 担心的"reasoning 被饿死"
   没有发生，`lambda_a = 1` 不用调。
2. 但反过来:reasoning 学到几乎无损，action 却和不给 reasoning 时一模一样。
   考虑到 36.5% 的 reasoning 只是把自己的动作决策念一遍，模型很可能只是学会了**复述**。
   这与 §6 的测量一致——`<PLAN>` 只提供 1.70 bit，而这些信息图像里本来就有。
   **对 SFT 这一步不是问题**（验收标准是不降），但 RL 阶段要靠 reasoning 拿收益时，
   这是第一个要回来查的地方。

基线的后续曲线，用于对照：

```
no-CoT  epoch 1  top1 0.24685
no-CoT  epoch 2  top1 0.28465
```

---

## 10. epoch-0 ckpt 的真实生成

`runs/sft/2026-09-02_05-47-39/epoch=0-loss=0.5485.ckpt`，val 集，**贪心解码**
（`do_sample=False`），prompt 严格照 §3 的两档构造。

复现脚本的关键点（`predict()` 现在还不能直接用，见文末）：

```python
sd = torch.load(CKPT)['state_dict']
V  = sd['autovla.vlm.lm_head.weight'].shape[0]      # 153713
NB = V - len(proc.tokenizer)                         # 2048 个 <action_i>
proc.tokenizer.add_tokens([f'<action_{i}>' for i in range(NB)])
model.resize_token_embeddings(len(proc.tokenizer))   # 先注册 action token 再 resize
model.load_state_dict({k[len('autovla.vlm.'):]: v for k, v in sd.items()
                       if k.startswith('autovla.vlm.')}, strict=False)
```

⚠️ action token 是 `ActionTokenizer` 加进去的（`models/action_tokenizer.py:48`），
**不在 processor 里**。不先 `add_tokens` 就 resize，会撞 153713 vs 151665 的 shape mismatch。

### 例 1 — token `fbc963dad5c956c0`

GT reasoning：
`The vehicle 9 meters ahead in my lane is stopped. Because I need to stay behind it, I should remain stopped: <PLAN>STOP,STRAIGHT</PLAN>`

**`reason_then_act`**（prompt 尾 `Explain your reasoning first.`）

```
<think>
I see a pedestrian 8 meters ahead of me. The vehicle 10 meters ahead in my lane is
moving slowly. Because I need to stay behind it, I should remain stopped:
<PLAN>STOP,STRAIGHT</PLAN>
</think>
<answer>
The final output action is: <action_51><action_114><action_16><action_8><action_87>
<action_167><action_125><action_12><action_12><action_12>
</answer>
```

**`act_directly`**（prompt 尾 `Answer directly.`）

```
<answer>
The final output action is: <action_51><action_114><action_16><action_8><action_87>
<action_167><action_125><action_12><action_19><action_12>
</answer>
```

### 例 2 — token `ffdb3409d3035213`

GT reasoning：
`The vehicle 21 meters ahead in my lane is moving at 7 m/s. I should hold my current speed and bear left: <PLAN>KEEP,LEFT</PLAN>`

**`reason_then_act`**

```
<think>
A pedestrian 32 meters in front of me will move into my path. I should hold my
current speed and bear left: <PLAN>KEEP,LEFT</PLAN>
</think>
<answer>
The final output action is: <action_7><action_1172><action_1538><action_258><action_1074>
<action_1074><action_1780><action_1814><action_1659><action_198>
</answer>
```

**`act_directly`**

```
<answer>
The final output action is: <action_7><action_1172><action_1981><action_258><action_1074>
<action_1074><action_1780><action_869><action_666><action_1943>
</answer>
```

### 读出来的三件事

**① 格式完全正确。** 两档都按 prompt 分流了：`act_directly` 不吐 `<think>`，
也没有空的 `<think>` 壳子；`<PLAN>` 标签、句式、距离表述都学到了。
**没有**出现 issue #23 里那种"只吐 CoT 文本、一个 action token 都不生成"的情况。

**② reasoning 会编。** 例 1 把"已停住的车"说成 `moving slowly`，还凭空加了
`I see a pedestrian 8 meters ahead of me.`；例 2 把"前方 21 米的车"说成
`A pedestrian 32 meters in front of me`。`<PLAN>` 大多对（3 条抽样里 2 条与 GT 一致），
但描述句是编的。才 1 个 epoch，属正常欠拟合，**epoch 3 再抽一次看有没有收敛**。

**③ 两档的 action token 前几位几乎相同。** 例 1 前 8 个完全一致，只有末尾分叉；
例 2 也是同样的模式。即**加不加 reasoning，轨迹基本不变**——
和 §9 里 `val_action_top1` 持平是同一件事的两种看法。
按验收标准（不降即可）这没问题，但 **RL 阶段要靠 reasoning 拿收益时，这是第一个要回来查的地方**。

---

## 11. `predict()` 的 prompt —— 已随 §3.5 的重构一并修好

**背景（发现时的状态）**：`models/autovla.py::get_prompt` 原本自带一套独立文案——
nuScenes 风格的五步式（`1. Scene Analysis / 2. Identification of Critical Objects / …`
＋ 那份 lateral/longitudinal 动作词表），而且 user prompt 不带模式后缀。
跑评测时模型会拿到训练中从没见过的 prompt。§10 的输出之所以正常，
是因为当时我手工照 SFT 的格式拼了 prompt。

**现状**：§3.5 的重构已经把两边统一到 `dataset_utils/prompt_spec.py`，
`get_prompt` 现在是：

```python
"text": prompt_spec.user_text(velocity, acceleration, instruction, self.use_cot,
                              cot_mode="reason_then_act"),
...
{"role": "system", "content": [{"type": "text", "text": prompt_spec.system_text(self.use_cot)}]},
```

旧文案已从 `autovla.py` 消失（`grep "Scene Analysis"` 无命中）。
评测/RL 一律走 `reason_then_act`——方法要改写的就是那句 `<PLAN>`，模型必须先说出来。

⚠️ 这次改动发生在训练**进行中**（2026-09-02 14:36）。不影响正在跑的 job：
Linux 默认 fork 启动 dataloader worker，继承父进程已导入的模块，不会重新 import；
且 §3.5 确认三档 CoT prompt 一字未变，只有 `use_cot: false` 那条路径回退了一个尾空格。

**评测前仍要确认的一件事**：`predict()` 走 `reason_then_act` 时是否需要
按档切换（例如想量"不给 reasoning 时轨迹变不变"，就得能切到 `act_directly`）。
`prompt_spec.user_text` 已经接受 `cot_mode` 参数，`get_prompt` 里写死成
`reason_then_act`，要做对照实验时从这里开一个环境变量即可。
