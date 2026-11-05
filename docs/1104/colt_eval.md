# CoLT-Drive 测评：VLM + AutoVLA

2026-11-04 ~ 05。设计见 [experiment_design.md](../1102/experiment_design.md) §4、§5。

代码：

- VLM 分类：[tools/navsafe_vlm/colt_probe.py](../../tools/navsafe_vlm/colt_probe.py)
- AutoVLA 测评：[tools/colt/eval_autovla.py](../../tools/colt/eval_autovla.py)
- 双卡启动：[scripts/1104/run_colt_autovla.sh](../../scripts/1104/run_colt_autovla.sh)

## 1. 数据和打分

数据在 `/scratch/mh2803/vla/CoLT-Drive/data/{vfull,vclean}`，每个 split 1768 个样本。每个样本有：

| 文件 | 内容 |
|---|---|
| `front_camera.jpg` | 前视图，1376×768。没有历史帧，也没有左右相机 |
| `meta.json` | 障碍物类别、类型、位置，导航指令，过去 2 s 的车速和加速度 |
| `gt.json` | `acceptable_pairs`：所有可接受的（纵向, 横向）组合 |

类别分布：

| 类别 | 样本数 |
|---|---|
| 生物 | 348 |
| 非生物物体 | 347 |
| 路面危险 | 348 |
| 整条路堵死 | 290 |
| 误报 | 435 |

障碍物位置有居中、偏左、偏右、悬空四种。

动作空间：

- 纵向：`keep_speed`、`slow_down`、`stop`、`yield`、`creep`、`speed_up`
- 横向：`keep_lane`、`nudge_left`、`nudge_right`

**打分**：把预测的（纵向, 横向）拿去 `acceptable_pairs` 里查，在里面就算对。官方测评用 LLM judge 把自由文本归到标准动作。我们的输出本来就是固定词表，所以直接查表，不用 judge。

下文的"组"按类别和位置划分：生物、整条路堵死各算一组，其余按"类别/位置"划分，如 `object/center`。

## 2. VLM：一次调用，输出 7 个标签之一

- **模型**：Qwen3-VL-32B-Instruct，输入一张前视图（缩到宽 1280）。
- **prompt**：[v28_single_f0.txt](../../tools/navsafe_vlm/prompts/v28_single_f0.txt)，`{speed}` 填 `history[-1].ego_speed`。
- **解析**：取输出里最后一个 `<TOKEN>`。

```
This photo was taken by a camera on the windshield of a car, looking forward. The car with the camera is not in the photo. The car is driving at {speed} m/s.
Look at the car's lane ahead: the stretch of road the car will drive over in the next few seconds, starting right in front of the car.
Is there anything on the road surface in the car's lane, other than vehicles driving or waiting in traffic? If you are not sure, assume there is. People on the sidewalk do not count.
Answer with exactly one token.
If it is a person or an animal:
<PEDESTRIAN> if it is a person, whatever they are doing
<CYCLIST> if it is a person riding a bicycle or scooter
<ANIMAL> if it is an animal
For anything else, look where it is within the car's lane, as seen in the photo, and steer away from it:
<AROUND_LEFT> if it is on the right part of the lane, or in the middle with more free road on its left
<AROUND_RIGHT> if it is on the left part of the lane, or in the middle with more free road on its right
<BLOCKED> only if it covers the whole width of the road, from edge to edge, with no free road on either side
<NONE> only if the car's lane ahead is empty road, with nothing on it except painted markings
Output only the token.
```

prompt 里只出现人、骑车的人、动物三个大类，不列具体物体。

### 2.1 VLM + 固定映射

标签按 §4.1 的表直接换成 CoLT 动作。这相当于一个完全照指令做的执行器，是 AutoVLA 要对比的基准线。

| 标签 | CoLT 动作 |
|---|---|
| PEDESTRIAN / ANIMAL / BLOCKED | stop + keep_lane |
| CYCLIST | slow_down + keep_lane |
| AROUND_LEFT / AROUND_RIGHT | slow_down + nudge_left / nudge_right |
| NONE | keep_speed + keep_lane |

vfull 全量结果：**73.9%**。

| 组 | n | 分数 |
|---|---|---|
| 生物 | 348 | 86.5 |
| 整条路堵死 | 290 | 74.8 |
| 物体，居中 | 289 | 63.7 |
| 物体，偏左 | 29 | 13.8 |
| 物体，偏右 | 29 | 58.6 |
| 路面危险，居中 | 290 | 44.5 |
| 路面危险，偏左 | 29 | 3.4 |
| 路面危险，偏右 | 29 | 86.2 |
| 误报（各位置） | 435 | 97.9–100 |
| **总计** | 1768 | **73.9** |

标签分布：

| 标签 | 次数 |
|---|---|
| AROUND_LEFT | 950 |
| BLOCKED | 317 |
| NONE | 196 |
| ANIMAL | 146 |
| PEDESTRIAN | 89 |
| CYCLIST | 57 |
| AROUND_RIGHT | 13 |

32B 很少输出 AROUND_RIGHT，所以偏左的障碍物（应该往右绕）几乎全错。

### 2.2 prompt 迭代（vfull 全量）

| 设置 | 分数 |
|---|---|
| 两步，8B + 8B（v22 + v23） | 64.3 |
| 两步，8B + 32B（v22 + v26） | 76.2 |
| 一步，32B v25 | 67.1 |
| **一步，32B v28（选定）** | **73.9** |
| 论文：Qwen3-VL-8B 直接答题 | 65.1 |
| 论文：Qwen3-VL-32B 直接答题 | 56.05 |

起作用的改动：

- **先问"车道里有没有东西"，拿不准就当作有**。旧问法下模型把大部分障碍物都答成 NONE。误报类答绕行基本都算对，所以宁可多答绕行。
- **按障碍物在车道里的位置决定往哪边绕**，比"比较两边空路"更准。

目标是一步调用、约 70%，不追求更高，给后面的 OPD 留空间。

## 3. AutoVLA：输出 meta-action

### 3.1 模型

| 项 | 值 |
|---|---|
| ckpt | `/data/mh2803/navsafe/navsafe_data/model_zoo/autovla_cf018/epoch=4-loss=0.6051.ckpt`（cf SFT，带 CoT） |
| 底座 | `/data/mh2803/navsafe/navsafe_data/model_zoo/_base/qwen2.5-vl-3b-instruct` |
| config | `config/eval/1001/autovla-navtest-eval-cot-tigris.yaml`，只把 `pretrained_model_path` 换成上面的底座 |
| 解码 | 和 NAVSIM 测评相同，temperature 0.01 |

载入后 missing 和 unexpected keys 都是 0。

### 3.2 输入

模型训练时的输入是 3 路相机 × 4 帧，CoLT 只有一张前视图。这里用 §3 的备选做法：直接复制前视图。这属于分布外输入，但 11-05 定了就用它，不重训。

| 输入 | 填法 |
|---|---|
| 前视，4 帧 | `front_camera.jpg` 复制 4 次 |
| 左前 / 右前，各 4 帧 | 同一张前视图（`--sides front`，默认），或全黑图（`--sides black`） |
| 车速 | `history[-1].ego_speed` |
| 加速度 | `abs(history[-1].ego_accel)`。训练 JSON 里存的是 [ax, ay] 的模长，非负 |
| 导航 | GO STRAIGHT / SLIGHT LEFT → `keep forward`，TURN LEFT → `turn left` |

prompt 由 `AutoVLA.get_prompt` 原样拼出，文案来自 `prompt_spec`，和训练一致。图像缩到 224×420，每路 240 个视觉 token。

### 3.3 三种指令来源

| 来源 | prompt | 对应 §5.1 的行 |
|---|---|---|
| `none` | 普通 `reason_then_act`，不给指令 | AutoVLA（不接 VLM） |
| `vlm` | 2 节的 v28 标签按下表变成 `Scene report: … Suggested action: ….`；标签是 NONE 时同 `none` | VLM + AutoVLA（cf SFT） |
| `oracle` | 由 GT 选标签，再同样变成指令 | 底座 + GT 指令 |

| 标签 | Scene report | Suggested action |
|---|---|---|
| PEDESTRIAN | A pedestrian is on the road ahead. | come to a stop |
| CYCLIST | A cyclist is on the road ahead. | slow down |
| ANIMAL | An animal is on the road ahead. | come to a stop |
| AROUND_LEFT | An obstacle is ahead in the ego lane. | change lanes to the left |
| AROUND_RIGHT | An obstacle is ahead in the ego lane. | change lanes to the right |
| BLOCKED | An obstacle is ahead in the ego lane. | come to a stop |

`oracle` 怎么选标签：

- **生物**：按类型选。狗、鹿、鸭/鹅群 → ANIMAL；骑车、骑电动滑板车 → CYCLIST；其余 → PEDESTRIAN。
- **其他类别**：按优先级取第一个"映射后的动作可接受"的标签。

  | 类别 / 位置 | 优先级 |
  |---|---|
  | 误报 | NONE |
  | 整条路堵死 | BLOCKED |
  | 偏左 | AROUND_RIGHT |
  | 其余 | AROUND_LEFT → AROUND_RIGHT → BLOCKED → NONE |

同一个（样本, 标签）只生成一次，三种来源共用缓存。

### 3.4 读数：`<PLAN>` 到 CoLT 动作

模型输出的 `<PLAN>SPEED,PATH</PLAN>` 就是它的 meta-action：

| PLAN | CoLT |
|---|---|
| STOP / DECELERATE / KEEP / ACCELERATE | stop / slow_down / keep_speed / speed_up |
| LANE_LEFT / LANE_RIGHT | nudge_left / nudge_right |
| STRAIGHT / LEFT / RIGHT（顺着路走、转弯） | keep_lane |

- 解析不出 `<PLAN>` 算错。冒烟测试里 100% 能解析。
- **AutoVLA 换道整体读成"减速 + 绕"**（`--lane-readout avoid`，默认，2026-11-05 定）：`<PLAN>` 的方向是 LANE_LEFT / LANE_RIGHT 时，不管速度词是什么，都读成 slow_down + nudge_left / nudge_right。
  - **和映射表同一口径**：VLM 映射表里 AROUND 就是"减速 + 绕"。AutoVLA 的词表里没有 nudge，换道只能对应 CoLT 的避让动作。
  - **按 AutoVLA 自己的输出，不按 VLM 的指令**：VLA 不换道、换反方向或改成停车时，照它的输出算。所以 VLM+映射 和 VLM+AutoVLA 的差别只来自 VLA 偏离指令，正好对应 OPD 要纠正的错误（绕行方向、AROUND 和 BLOCKED 混淆、不必要的停车）。
  - **CoLT 的 GT 支持这样读**（vfull）：
    - 1264 个"绕行可接受"的样本里，"减速 + 绕"100% 可接受。
    - "保持速度 + 绕"只在误报类（377 个）和 29 个非生物物体样本里可接受。
    - 任何速度配绕行可接受时，"减速 + 同方向绕"也一定可接受（vfull 981 种情况、vclean 983 种，0 例外）。

    所以这条规则只会把错的读成对的，不会把对的读成错的。
  - **代价**：换道时速度对不对就不考了。cf 训练数据里换道样本约 30% 配 KEEP（见第 6 节），用 `--lane-readout plan` 照原样读时，这部分会被判错。
  - **不要全部替换**（`all`）：CoLT 本身就接受"减速 + 保持车道"，一直答它就有 66.7%。全部替换会把不接 VLM 的基准线从 36% 抬到 56%，失去对比意义。
- **轨迹读数**（辅助）：速度用 `plan_from_track`；横向看 3 s 处的 y（向左为正），≥ 1 m 算 nudge。这只用来核对 PLAN 和轨迹是否一致。转弯样本的横向会被误读成 nudge，所以以 PLAN 读数为准。

### 3.5 输出的列

| 列 | 含义 |
|---|---|
| VLM+映射 | 该来源的标签按 2.1 的表直接换成动作的分数 |
| VLA PLAN | AutoVLA 的 `<PLAN>` 读数分数（主指标） |
| VLA 轨迹 | 轨迹读数分数 |
| 照做率 | 给了指令的样本里，`<PLAN>` 对应维度和 suggestion 相同的比例。速度类看速度维，换道类看方向维 |

另外还会输出每个标签实际得到的 PLAN 分布。

## 4. 结果（冒烟测试：vfull 每 50 个取 1 个，36 个样本）

读数用 `--lane-readout avoid`。

| 设置 | 分数 |
|---|---|
| AutoVLA 单独跑（不接 VLM） | 36.1 |
| **v28 VLM + AutoVLA** | **72.2** |
| v28 VLM + 固定映射 | 75.0 |
| GT 指令 + AutoVLA | 94.4 |

- **不接 VLM**：78% 的样本输出 keep_speed + keep_lane，模型基本不对障碍物做反应，只在误报类上得分。
- **接 VLM**：照做率 96.9%。BLOCKED → stop、AROUND_RIGHT → 右绕、ANIMAL → stop 全部照做。比固定映射低的 3 分来自 1 个 AROUND_LEFT 样本，模型没照做，输出 speed_up + keep_lane。
- **换道照 PLAN 原样读时**（`--lane-readout plan`）：VLM + AutoVLA 是 63.9，GT 指令 + AutoVLA 是 86.1。
  - 接 VLM 时，AROUND_LEFT 有 13% 输出 keep_speed + nudge_left。
  - 给 GT 指令时，这个比例是 30%。

36 个样本只看趋势。正式分数要跑全量 1768 个样本，两张卡（GPU0、GPU2，和别人共用）约 3 小时。冒烟测试里每次生成约 6 s。

## 5. 怎么跑

```bash
cd /home/mh2803/vla/doc_vla_search
PY=/home/mh2803/miniconda3/envs/autovla_codeclean/bin/python

# VLM 标签（已跑完、有缓存）
CUDA_VISIBLE_DEVICES=0 /scratch/mh2803/vla/simlingo/conda_envs/qwen3vl/bin/python tools/navsafe_vlm/colt_probe.py \
    --model qwen3vl-32b --single tools/navsafe_vlm/prompts/v28_single_f0.txt

# AutoVLA
$PY tools/colt/eval_autovla.py run --dry-run                 # 不占 GPU：打印 prompt 和视觉输入尺寸
CUDA_VISIBLE_DEVICES=0 $PY tools/colt/eval_autovla.py run --every 50       # 冒烟
GPUS="0 2" bash scripts/1104/run_colt_autovla.sh             # 全量，每张卡一片，跑完自动 agg
$PY tools/colt/eval_autovla.py agg                           # 随时出表（跑到一半也行）
$PY tools/colt/eval_autovla.py agg --lane-readout plan       # 换读数规则，不用重跑
```

常用参数：

| 参数 | 作用 |
|---|---|
| `--split vclean` | 测 vclean。vlm 来源需要先在 vclean 上跑 v28 |
| `--sides black` | 左前 / 右前喂全黑图 |
| `--sources none vlm oracle` | 选择指令来源 |
| `--ckpt` | 换模型。结果按 ckpt 所在目录名分开存 |

run 和 agg 的 `--split / --sides / --tag / --every` 要一致。中断后重跑会接着做。

## 6. 注意

- **输入是分布外的**：单张前视图复制成 3 路 × 4 帧，模型看到的是"静止画面 + 三路相同"。2026-11-05 定：不为 CoLT 做 §3 的混合训练，就用这种输入。冒烟测试里 `<PLAN>` 100% 可解析，照做率 97%。不接 VLM 时模型基本不对障碍物做反应（36%），这里分不清有多少是输入造成的。
- **换道读成"减速 + 绕"只是打分时的读数规则**（3.4），模型和训练数据都没改。cf 训练数据里换道样本的速度词由合成轨迹决定，并不都是减速：LANE_LEFT 1973 条中，DECELERATE 60%、KEEP 30%、ACCELERATE 10%；LANE_RIGHT 963 条中分别是 49% / 31% / 20%。所以模型照"change lanes"做时，有一部分输出 KEEP，轨迹也确实没减速。轨迹读数不套这条规则，看的是车实际怎么开。
- **SLIGHT LEFT 当作 keep forward**（vfull 中 122 个）。训练数据只有 keep forward / turn left / turn right 三种指令。
- **AutoVLA 不会输出 yield / creep**，这两种动作只能靠 slow_down / stop 对上。
- **VLM 漏报（NONE）时没有指令**，只能靠 AutoVLA 自己看见障碍物，而它单独跑时基本看不见。

## 7. 示例（冒烟测试里挑的 9 个）

图在 [colt_examples/](colt_examples/)，由 [tools/colt/render_examples.py](../../tools/colt/render_examples.py) 读已跑完的结果画出，不重跑模型。

- **左边**：前视图。上面写着 GT 可接受的动作，以及不接 VLM 时模型的 reasoning。
- **右边**：ego 系俯视轨迹，向上为前，向右为右，虚线是本车道边界（±1.85 m）。

| 线 | 指令 |
|---|---|
| 灰 | 不接 VLM |
| 红 | 接 VLM（v28） |
| 绿虚线 | GT 指令 |

VLM 答 NONE 时红线和灰线重合。图下方写着每种指令、模型的 `<PLAN>`、读成的 CoLT 动作和对错。

**VLM 指令起作用（不接 VLM 错，接了对）**

| 图 | 说明 |
|---|---|
| [00 狗，偏左](colt_examples/00_R6_O5_left_biased_v_full.png) | 不接 VLM：保持速度。VLM 答 ANIMAL → "come to a stop" → `STOP,STRAIGHT` ✓ |
| [01 大石头，整条路堵死](colt_examples/01_NP11_O41_full_lane_blockage_v_full.png) | 不接 VLM：保持速度。VLM 答 BLOCKED → `STOP` ✓ |
| [02 锥桶，居中](colt_examples/02_NP14_O8_center_v_full.png) | 不接 VLM：保持速度。VLM 答 AROUND_LEFT → `DECELERATE,LANE_LEFT` ✓ |
| [03 木托盘，居中](colt_examples/03_NP12_O30_center_v_full.png) | VLM 答 AROUND_LEFT → `KEEP,LANE_LEFT`。按 `--lane-readout avoid` 读成 slow_down + nudge_left ✓，用 `plan` 读则 ✗ |
| [04 树影，误报](colt_examples/04_NP04_O1_center_v_full.png) | VLM 答 NONE，模型保持速度 ✓，没有误报 |

**VLM 错了，AutoVLA 照着错（OPD 要纠正的）**

| 图 | 说明 |
|---|---|
| [05 坑洞，偏左](colt_examples/05_NP10_O10_left_biased_v_full.png) | 坑在车道左侧，应往右绕。VLM 答 AROUND_LEFT，模型照做，往左越过双黄线 ✗。GT 指令 AROUND_RIGHT → `DECELERATE,LANE_RIGHT` ✓ |
| [07 推婴儿车的人，居中](colt_examples/07_R5_O18_center_v_full.png) | 不接 VLM 时模型自己减速 ✓。VLM 答 AROUND_LEFT，模型反而 14 m/s 往左换道 ✗。GT 指令 PEDESTRIAN → `STOP` ✓ |

**VLM 漏报（没有指令可纠正）**

| 图 | 说明 |
|---|---|
| [06 消防车横停，整条路堵死](colt_examples/06_NP20_O42_full_lane_blockage_v_full.png) | VLM 答 NONE，模型保持速度 ✗。GT 指令 BLOCKED → `STOP` ✓ |

**AutoVLA 没照做**

| 图 | 说明 |
|---|---|
| [08 掉落的车轮，左转路口](colt_examples/08_NP17_O26_center_v_full.png) | 导航 TURN LEFT，车速 3.2 m/s。三种指令下都输出 `ACCELERATE,LEFT`（顺着路口左转加速），忽略了 "change lanes to the left"。这是 36 个样本里唯一不照做的 |

## 8. 结果文件

| 内容 | 路径 |
|---|---|
| VLM 标签 | `/scratch/mh2803/vla/colt/vlm_probe/results/vfull/qwen3vl-32b/final_single_v28_single_f0_01aabc_w1280.jsonl` |
| AutoVLA 逐条输出 | `/scratch/mh2803/vla/colt/autovla/vfull/autovla_cf018_sides-front/gen_s*.jsonl`（`text` / `plan` / `traj`） |
| AutoVLA 汇总 | 同目录下的 `summary.json` |
| 日志 | `/scratch/mh2803/vla/colt/autovla/logs/` |
