# 反事实指令 SFT：训练代码改动

2026-10-27。计划和数据：[cf_sft_autovla_plan.md](cf_sft_autovla_plan.md)（§2 样本格式、§3 数据、§4 代码）。对应 simlingo 那边的 `doc_drive_search/docs/1019/cf_sft_v1_code_changes.md`。

## 做了什么

训练时以 `model.cf.ratio`（默认 0.25）的概率把一条样本换成**反事实（cf）样本**：

- 从 D6 导出的 cf 索引（`cf_navtrain_v1_{train,val}.npz`）里按类别抽一条；
- 按 token 读 `navtrain_cot` 里的原 JSON，把 `gt_trajectory` 换成 cf 轨迹（`compute_targets` 会现场编码成 action token）；
- prompt 末尾加 `Scene report: … Suggested action: ….`，后面不接任何指令；
- 答案只有 `<PLAN>速度,方向</PLAN>` 和 10 个 action token，没有 `<think>`、没有理由。

其余 75% 的普通样本和现在的 CoT SFT 逐字一致。

```
# cf 样本（照做）
user:   … over the next five seconds. Scene report: A cyclist is on the road ahead. Suggested action: slow down.
target: <PLAN>DECELERATE,LEFT</PLAN>
        <answer>
        The final output action is: <action_…>×10
        </answer>

# 解耦样本（cf 的 10%）：suggestion 换成别的动作，report 仍和答案相符；只有 10 个 action token 算 loss
user:   … Scene report: A cyclist is on the road ahead. Suggested action: change lanes to the right.
target: <PLAN>STOP,STRAIGHT</PLAN>
        <answer> … </answer>
```

## 改了哪些文件

| 文件 | 改动 |
|---|---|
| `dataset_utils/prompt_spec.py` | 第四种模式 `counterfactual`；report、suggestion、`<PLAN>` 词表 |
| `dataset_utils/sft_dataset.py` | cf 抽样、换轨迹、cf 答案；collator 屏蔽解耦样本的文字 label；像素读 config；`__main__` 冒烟测试 |
| `models/autovla.py` | `train_text_loss` / `val_text_loss` 改成每个 rank 都 log（防 DDP 死锁） |
| `tools/run_sft.py` | 给 collator 传 `action_start_id`；可选的调试参数 |
| `config/training/1026/qwen2.5-vl-3B-nuplan-cf-sft-navtrain.yaml` | 正式配置（数据路径暂时是本机的） |
| `config/training/1026/qwen2.5-vl-3B-nuplan-cf-sft-debug-2gpu.yaml` | 本机两卡调试配置 |
| `scripts/1027/run_cf_sft_debug_2gpu.sh` | 本机两卡调试启动脚本 |

### `dataset_utils/prompt_spec.py`

- `user_text(...)` 加两个可选参数 `report=""`、`suggestion=""`，新模式 `cot_mode="counterfactual"` 返回 base + ` Scene report: {report} Suggested action: {suggestion}.`。
- 原来的三种模式（`reason_then_act` / `reasoning_given` / `act_directly`）和 `use_cot: false` 一行没动。
- 新增常量：
  - `CF_SUGGESTIONS`：5 个 suggestion。KEEP 没有 suggestion。

    | 类别 | suggestion |
    |---|---|
    | STOP | come to a stop |
    | DECELERATE | slow down |
    | ACCELERATE | speed up |
    | LANE_LEFT / LANE_RIGHT | change lanes to the left / right |

  - `CF_REPORTS`：固定 7 句 report。
  - `CF_ACTION_REPORTS`：答案的动作 → 可以配的 report。

    | 动作 | 可配的 report |
    |---|---|
    | STOP、DECELERATE | pedestrian / cyclist / animal / stopped vehicle / obstacle / cut-in |
    | ACCELERATE、KEEP | no hazard |
    | 换道 | stopped vehicle / obstacle |

  - `PLAN_SPEED_WORDS`、`PLAN_PATH_WORDS`：方向词在 LEFT / STRAIGHT / RIGHT 之外加了 `LANE_LEFT` / `LANE_RIGHT`。`models/utils/rl_rewards.py` 的 `PATH_WORDS` 没改（那份只读），评测解析换道要用这里的。

### `dataset_utils/sft_dataset.py`

**`__init__`**

- 像素：`self.min_pixels` / `self.max_pixels` 读 `model.video`。原来写死 28·28·128，而推理（`AutoVLA.get_prompt`）读的是 config 的 140，两边对不上。没写 `model.video` 时仍是 128。
- cf：配了 `data.<split>.cf_index` 且 `model.cf.ratio > 0` 时：
  - 读索引；
  - 按类别建池子；
  - 检查 `type_probs` 里没有 KEEP，`decouple_type_probs` 里的类别在索引里都有；
  - 检查索引里每个 token 在 JSON 目录里都找得到；
  - 打印一行 `[cf] …` 汇总。
- `use_cot: false` 时配了 cf 会直接报错，因为 cf 样本用的是 CoT 的 system prompt。

**`_sample_cf()`**

1. 以 `decouple_ratio` 的概率定为解耦样本；
2. 照做样本按 `type_probs` 选类别，解耦样本按 `decouple_type_probs` 选；
3. 从这个类别的池子里随机取一条；
4. report 从 `CF_ACTION_REPORTS[类别]` 里随机挑一句。每次取样本都现抽，不写进索引；
5. suggestion：照做样本是本类的 suggestion，解耦样本是另外 4 个里随机一个。

**`__getitem__`**

- **cf 样本**：
  - 不用 `idx` 这一帧，按 cf 的 token 读 JSON，再换掉 `gt_trajectory`；
  - 答案是 `<PLAN>{plan_speed},{plan_path}</PLAN>\n<answer>…</answer>`，`has_cot=False`；
  - 返回值多了 `skip_text_loss`（解耦样本为 True）和 `cf`（抽样信息，调试用）。
- **普通样本**：路径不变。没配 cf 时一次 `random()` 都不多调，随机数序列和原来一样。
- **`PRINT_SFT_DEBUG`**：打印里多了 `cf=类别 decouple=…`。

**`DataCollator`**

- 新字段 `action_start_id`（默认 151665），由 `run_sft.py` 从 config 传入。
- `skip_text_loss` 的样本把 `< action_start_id` 的 label 全设成 -100，只剩 10 个 action token。
- 这个标记不放进 batch：`AutoVLA.forward` 只 pop 掉 `gt_trajectory` / `gt_action` / `has_cot`，其余的键会原样传给 VLM。

**`__main__`（冒烟测试）**

```bash
python -m dataset_utils.sft_dataset --config training/1026/qwen2.5-vl-3B-nuplan-cf-sft-debug-2gpu --n 200
```

逐条检查 cf 样本：
- prompt 结尾只有 report + suggestion；
- report 和答案的动作相符；
- 照做 ⇔ suggestion 等于本类；
- KEEP 只当解耦样本；
- `skip_text_loss` ⇔ 解耦；
- 答案没有 `<think>`；
- `<PLAN>` 在词表里且和类别一致；
- gt 轨迹就是 cf 轨迹；
- 过一遍 collator 后，解耦样本只剩 10 个 action label。

最后打印各类别和各句 report 的占比，有错误时 exit 1。

### `models/autovla.py`

- 原来 `train_text_loss` 只在 `_tm.any()`（有文字 token）时 log，并且带 `sync_dist=True`。
- 解耦样本的文字 label 全是 -100。batch_size=1 时，这个 rank 就会跳过这次 log，而别的 rank 进了 all-reduce，结果是 DDP 死锁。09-02 因为同类问题挂过一次，见同一处的注释。
- 现在改成每个 rank 都 log，没有文字 token 时记 0。解耦样本约占全部样本的 2.5%，对曲线影响很小。`val_text_loss` 同样处理。
- loss 本身没改，还是 `output.loss + action_loss`。

### `tools/run_sft.py`

- `DataCollator(..., action_start_id=config['model']['tokens']['action_start_id'])`。
- `pl.Trainer` 加了三个可选参数，config 不写时和原来一样：
  - `max_steps=training.max_steps`（默认 -1）
  - `val_check_interval=training.val_check_interval`（默认 1.0）
  - `limit_val_batches=training.limit_val_batches`（默认 1.0）

## 配置

```yaml
model:
  use_cot: true
  cot_ratio: 0.75                  # 普通样本的 reason_then_act / act_directly，和 103k CoT 基线一样
  cot_in_prompt_ratio: 0.0
  cf:
    ratio: 0.25                    # cf 样本占全部样本的比例
    decouple_ratio: 0.1            # cf 样本里解耦样本的比例
    type_probs: {STOP: 0.25, DECELERATE: 0.25, ACCELERATE: 0.25, LANE_LEFT: 0.125, LANE_RIGHT: 0.125}
    decouple_type_probs: {STOP: 0.2, DECELERATE: 0.2, KEEP: 0.2, ACCELERATE: 0.2, LANE_LEFT: 0.1, LANE_RIGHT: 0.1}
  video: {min_pixels: 109760, max_pixels: 109760}   # 28·28·140，训练现在也读这里
data:
  train:
    json_dataset_path: …/navtrain_cot
    sensor_data_path: …/sensor_blobs                 # navtrain_cot 的相机路径是相对路径，不能是 null
    cf_index: …/cf_navtrain_v1/cf_navtrain_v1_train.npz
  val:   （同上，_val）
training:
  # 调试用，可选
  max_steps: 40
  val_check_interval: 20
  limit_val_batches: 8
```

- 类别比例是 10-27 定的默认值。
- 换道的池子只有约 2.9k 条，每个 epoch 每条大约重复 2 次。
- 去掉 `cf_index`（或把 `ratio` 设成 0）就退回普通的 CoT SFT。

## 验证

| 检查 | 结果 |
|---|---|
| 普通样本 prompt 和改之前（git HEAD 的 `prompt_spec.py`）逐字比对 | 3 万组随机输入 × 3 种模式 × 开 / 关 CoT，全部一致 |
| 冒烟测试（200 条、120 条两次） | 0 错误。cf 占 22.5% / 25%，解耦约 10%；解耦样本只剩 10 个 action label，照做样本 43–45 个 |
| 本机两卡调试（GPU 1,2，DDP，40 个 optimizer step，accum 2，每 20 个 batch 验证 8 条） | 没有报错，也没有死锁；`PRINT_SFT_DEBUG` 里能看到 cf 样本 |
| 显存 | 每张卡峰值约 79 GB（batch_size=1；fp32 主权重 + AdamW + 140 像素） |
| 可复现 | 两次调试的最终 val_loss 是 3.8593 / 3.8579 |

调试时的验证 loss：

| step | val_action_loss | val_text_loss | val_loss |
|---|---|---|---|
| 9 | 17.6 | 3.33 | 6.44 |
| 19 | 14.7 | 2.73 | 5.53 |
| 29 | 13.1 | 1.93 | 4.32 |
| 39 | 12.2 | 1.33 | 3.86 |

速度：调试时约 0.37 batch/s/卡，每隔一步就有一步约 5 s。原因是 accum=2 时每两个 batch 就要 all-reduce 一次 fp32 梯度，而本机 NCCL 只能走 loopback。正式配置 accum=16，会快不少，但没有实测。

## 怎么跑

```bash
# 冒烟测试（CPU）
conda activate autovla_codeclean
export PYTHONPATH=$PWD/navsim:$PWD NUPLAN_MAPS_ROOT=/scratch/mh2803/vla/nuplan/maps NUPLAN_MAP_VERSION=nuplan-maps-v1.0
python -m dataset_utils.sft_dataset --config training/1026/qwen2.5-vl-3B-nuplan-cf-sft-debug-2gpu --n 200

# 本机两卡调试（默认 GPU 1,2）
bash scripts/1027/run_cf_sft_debug_2gpu.sh

# 本机跑正式配置
CONFIG=training/1026/qwen2.5-vl-3B-nuplan-cf-sft-navtrain bash scripts/1027/run_cf_sft_debug_2gpu.sh
```

本机注意：
- 两卡必须同时设 `NCCL_P2P_DISABLE=1` 和 `NCCL_SHM_DISABLE=1`，脚本里已经设了。
- 基座权重在 `/scratch/mh2803/vla/checkpoints/Qwen2.5-VL-3B-Instruct`，仓库里的 `./Qwen2.5-VL-3B-Instruct` 是软链接，已在 `.gitignore` 里。

## 上 cluster 之前

- **路径**：改 `data.train/val` 三个路径（`json_dataset_path`、`sensor_data_path`、`cf_index`）。`sensor_data_path` 设成 cluster 上的 `sensor_blobs`。
- **基座权重**：在 cluster 的仓库里放好 `./Qwen2.5-VL-3B-Instruct`，它不在 git 里。
- **要搬过去的数据**：
  - `navtrain_cot` 和 `navtrain_cot_val`（JSON）；
  - `cf_navtrain_v1_{train,val}.npz`（23 MB / 0.5 MB）；
  - navtrain 的 `sensor_blobs`。
- **batch**：`accumulate_grad_batches` 按卡数改，保持 global batch = 1 × accum × 卡数 = 32。
- **显存**：每张卡约 79 GB。80 GB 的卡会很紧，放不下就换 8-bit AdamW 或降像素。

## 还没做 / 已知问题

- **评测侧还没改**：
  - 开环评测解析 `<PLAN>` 要用 `prompt_spec.PLAN_PATH_WORDS`，否则换道会被当成解析失败；
  - 推理的 `get_prompt` 还不能带 report / suggestion；
  - NavSafe 的 oracle 实验原来预填 `<think>…<PLAN>…</think>`，接新 checkpoint 要改成只预填 `<PLAN>…</PLAN>`。
- **report 很可能被忽略**：照做样本里 suggestion 就等于答案。这一版和 simlingo 一致，不加对策，开环时做"固定 suggestion、只换 report"的检查并记录下来（计划 §5、§6）。
- **cf 格式不走 RL 的格式检查**：`rl_rewards` 的格式检查要求 `reason_then_act` 有 `<think>`。这一版不做 RL，先记下。
