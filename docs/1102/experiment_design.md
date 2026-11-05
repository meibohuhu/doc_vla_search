# 实验设计：两条线、四个 benchmark（2026-11-03）

## 1. 总体结构

同一个方法（VLM 判断 → 指令 → 用 cf 训练过、能听指令的驾驶模型）在两个底座模型、两个数据域上各实现一遍。

| | CARLA 线 | nuPlan 线 |
|---|---|---|
| 底座模型 | SimLingo | AutoVLA |
| 训练数据 | CARLA（SimLingo 数据 + cf） | nuPlan navtrain（+ cf） |
| 长尾 / 危险测试 | Fail2Drive（ID + OOD），闭环 | NavSafe（闭环）+ CoLT（开环） |
| 通用驾驶能力 | Bench2Drive | NAVSIM |
| 共同的方法 | VLM 判断 → 指令 → cf 训练的驾驶模型 | 同左 |

覆盖面：

| 维度 | 覆盖 |
|---|---|
| 底座 / codebase | SimLingo、AutoVLA |
| 数据域 | CARLA 仿真、nuPlan 真实数据 |
| 闭环 | CARLA（Fail2Drive ID/OOD、Bench2Drive）、NavSafe |
| 开环 | CoLT（50 种长尾障碍物）、NAVSIM（通用能力，半闭环） |

这套组合能说明三点：

- 方法不依赖某一个底座模型；
- 在仿真和真实数据上都有效；
- 闭环行为和开环决策都有改进。

## 2. nuPlan 线要补 NAVSIM

CARLA 线用 Bench2Drive 说明"加了 cf 训练后通用驾驶没变差"，nuPlan 线也需要对应的测试，用的是 NAVSIM。否则只能说明在危险场景有效，说明不了通用能力没有损失。

NavSafe 的普通场景也能证明这一点。NavSafe 的 280 个场景分两类，正好对应 Fail2Drive 的 Base / Gen：

| | 分布内 | 分布外 |
|---|---|---|
| CARLA 线 | Fail2Drive Base 路线 | Fail2Drive Gen 路线 |
| nuPlan 线 | NavSafe 普通场景（218）：重放 nuPlan 日志，事件是日志里本来就有的（行人、锥桶、施工区、前车急刹等） | NavSafe 特殊场景（62）：插入危险参与者（动物、微出行、事故车、对向车等），nuPlan 里很少或没有 |

- 两类场景的日志都来自 navtest（个别是 nuPlan test 划分），和 navtrain 没有重合。
- NavSafe 结果按普通 / 特殊分开报，要做到分布外提升、分布内不掉。

NAVSIM 和 NavSafe 普通场景互补：

- **NAVSIM**：标准 benchmark，和别人的数直接可比。
- **NavSafe 普通场景**：闭环，而且接着 VLM 跑，还能看出 VLM 误报会不会让车无故减速或停车。

## 3. NavSafe 和 CoLT 用同一个 checkpoint

- CoLT 只有一张前视图（1376×768），没有历史帧，也没有左右相机。
- 现在的 AutoVLA 输入是 3 路相机 × 4 帧。

做法是**混合训练**：nuPlan 数据里一部分是前视单帧，一部分是 3 路 × 4 帧（和原 AutoVLA 一样），cf 样本按同样比例拆分。

- 前视单帧仍按 video 喂（num_frames=1），forward 不变。
- 比例待定，初步设想是 70% 多帧多视角、30% 前视单帧。
- 训好的新 checkpoint 要同时用于 **NAVSIM、NavSafe、CoLT**，NavSafe 要用新 checkpoint 重新跑。

备选方案是不重训，在 CoLT 上把前视图复制成 3 路 × 4 帧硬喂。这属于分布外输入，只当临时检查用。

**2026-11-05 定：不做混合训练，CoLT 就用复制前视图的输入**。冒烟测试里 cf SFT ckpt 照做率 97%，`<PLAN>` 全部可解析，测评见 [docs/1104/colt_eval.md](../1104/colt_eval.md)。NAVSIM、NavSafe 仍用 3 路 × 4 帧。

## 4. VLM 部分统一（NavSafe 和 CoLT 共用；Fail2Drive 不变）

### 4.1 标签 = scene report + suggested action

VLM 只做分类，标签按固定的表变成 cf 训练用的 report 和 suggestion（`prompt_spec.CF_REPORTS` / `CF_SUGGESTIONS`），再交给 VLA。

| 组 | VLM 标签 | Scene report | Suggested action | CoLT 得到的动作 |
|---|---|---|---|---|
| 生物 | PEDESTRIAN | A pedestrian is on the road ahead. | come to a stop | 停 + keep lane |
| | CYCLIST | A cyclist is on the road ahead. | slow down | 减速 + keep lane |
| | ANIMAL | An animal is on the road ahead. | come to a stop | 停 + keep lane |
| 非生物 | AROUND_LEFT | An obstacle is ahead in the ego lane. | change lanes to the left | 减速 + nudge left |
| | AROUND_RIGHT | An obstacle is ahead in the ego lane. | change lanes to the right | 减速 + nudge right |
| | BLOCKED | An obstacle is ahead in the ego lane. | come to a stop | 停 + keep lane |
| — | NONE | 不给指令 | | 保持速度 + keep lane |

- **按反应方式分，不按物体身份分**：生物会动、方向难料，减速或停车让它先走；静止的东西有空路就绕，没有就停。
- **生物**：report 说是哪一类，suggestion 由类别决定。**非生物**：report 都是 `obstacle`，suggestion 由哪边有空路决定。NavSafe 的 I-3 也报 `obstacle`，测试时不点名。
- prompt 里只出现人、骑车的人、动物三个大类。非生物只说"路上有车不该压过去的东西"，不列具体物体。
- 这套分类早于 CoLT：NavSafe 先定了 PEDESTRIAN / CYCLIST / ANIMAL / BLOCKED，cf 训练的 report 是 10 月为 SimLingo 定的。
- 标签全对时 CoLT 的上限是 93.0%（vfull）；BLOCKED 如果映射成减速，上限是 95.3%。为了和 NavSafe 闭环一致，BLOCKED 映射成停车。

### 4.2 两步调用

| 步骤 | 判什么 | NavSafe 输入 | CoLT 输入 |
|---|---|---|---|
| 1 | PEDESTRIAN / CYCLIST / ANIMAL / NONE | 三张图（前、左前、右前） | 前视图 |
| 2（第 1 步答 NONE 时） | AROUND_LEFT / AROUND_RIGHT / BLOCKED / NONE | 前视图 | 前视图 |

- 规则和标签两边相同，输入跟着各自的传感器走。第 1 步的 prompt 只有开头介绍相机的那句话不同。
- 第 2 步两边完全一样。原来 v7 里"两车斜停、像小事故"只贴合 NavSafe 一个画面，要换成通用说法：路径上有没有不该压过去的东西、哪边有空路。
- NavSafe 的 I-3 按定义（`navsafe/benchmark/event_types/I-3.yaml`）一定有同向车道可以绕，"停下不动"算失败，所以只有 AROUND_LEFT / AROUND_RIGHT。BLOCKED 只出现在 CoLT 的整条路堵死这一类。
- 第 2 步用 8B 还是 32B 待测。旧问法下 8B 判不出 BLOCKED，新问法要重新测。

### 4.3 CoLT 的 GT（vfull，各动作的可接受率 %）

| 类别 / 位置 | 样本 | 保持 + keep lane | 减速 + keep lane | 停 + keep lane | 减速 + 左绕 | 减速 + 右绕 |
|---|---|---|---|---|---|---|
| 生物，所有位置 | 348 | ≈0 | 100 | 100 | — | — |
| 非生物物体，居中 | 289 | 0 | 10 | 24 | 65 | 68 |
| 路面危险，居中 | 290 | 10 | 42 | 39 | 59 | 57 |
| 物体 / 危险，偏左或偏右 | 116 | 部分可以 | 低 | 低 | 往空的一侧绕：约 65–88 | |
| 整条路堵死 | 290 | 12 | 100 | 86 | 33 | 10 |
| 误报类 | 435 | 100 | 80–100 | **0** | 80–100 | 80–100 |

误报类答停车一定错，答绕行大多算对。所以 prompt 要让 VLM 拿不准时倾向于答绕行，而不是堵死。

### 4.4 还要改的

- **换道时的速度**：cf 数据里换道样本的速度词由合成轨迹决定，约 30% 是 KEEP（LANE_LEFT：DECELERATE 60%、KEEP 30%、ACCELERATE 10%），而 CoLT 要"减速 + 绕"。要么 cf 数据让换道都配减速，要么新增 suggestion（如 `slow down and change lanes to the left`）。和题库一起定。CoLT 测评里 AutoVLA 的 `<PLAN>` 按固定表映射到 CoLT 动作，LANE_LEFT / LANE_RIGHT → slow_down + nudge_left / nudge_right，和 VLM 映射表的 AROUND 一致（[docs/1104/colt_eval.md](../1104/colt_eval.md) §3.4）。
- **NavSafe agent**：`HazardCommander` 要加 AROUND → LANE_LEFT / LANE_RIGHT、BLOCKED → STOP。
- **NavSafe 测试数据**：62 个特殊场景加 I-1、I-2 的 20 个普通场景，存三张图（`--vis-cameras CAM_L0,CAM_R0`）重跑，用来测第 1 步三张图对前视图的差别，以及第 2 步的绕行方向。

## 5. CoLT 上各部分的贡献

CoLT 只考"纵向动作 + 横向动作"文字，不要轨迹，也不是闭环。纯 VLM 就能直接答题，论文表里大部分条目都是这样。

在我们的流水线里，最终答案主要由 VLM 那一步决定：

| VLM 输出 | 指令 | 最终答案 |
|---|---|---|
| LIVING | slow / stop | slow down / stop + keep lane |
| AROUND_LEFT | lane left | slow down + nudge left |
| AROUND_RIGHT | lane right | slow down + nudge right |
| BLOCKED | stop | stop + keep lane |
| NONE | 无 | keep speed + keep lane |

GT 分析结果：类别和绕行方向都判对时，用固定映射就能拿到 95.7%（vfull）。

**"VLM + 固定映射"等于一个完全照做的执行器**，是比较的基准线：

- **只做 cf SFT**：AutoVLA 基本只会照做（CARLA 线 cf SFT 实测 override 率 0.0%），所以 VLM + AutoVLA ≤ VLM + 映射表。
- **做了 OPD**：VLA 能纠正错误指令，**最终结果必须高于 VLM + 映射表**，这正是 OPD 要证明的东西。按 `doc_drive_search/docs/1022/ablation_discussion.md` §3 的公式，最终可接受率 = a·(1−e) + (1−a)·f，只有 (1−a)·f > a·e 时才有提升。

### 5.1 CoLT 表（附录；nuPlan 线也做 OPD 时）

主消融在 Fail2Drive 上做，CoLT 只放附录，四行：

| 设置 | 预期 |
|---|---|
| AutoVLA（不接 VLM） | 底座模型自己的长尾能力 |
| Qwen3-VL-8B 直接答题（官方 `COMPLEX_PROMPT`） | 论文 65.5% |
| 8B + AutoVLA（cf SFT，只照做） | 约等于 VLM 指令本身的准确率 |
| **8B + AutoVLA（OPD，完整系统）** | **高于上面三行** |

最后一行高于 AutoVLA 说明接 VLM 有用，高于 cf SFT 说明 OPD 有用，高于 8B 直接答题说明比直接用 VLM 答题好。

只报 CoLT 一列。论文 Table 1 的 In-domain Acc. 是作者在私有语料上测的（10,000 个 clip，抽 3,613 个测试样本），我们测不了。它的作用是证明常规驾驶能力没退化，这在我们这边由 NAVSIM（见第 2 节）和 NavSafe 的普通场景承担。

### 5.2 注意

- **前提是 nuPlan 线也要做 OPD**。OPD 目前只在 SimLingo（CARLA）上实现，题库的规则和屏蔽依赖 CARLA 的特权信息（`lane_change_type`、`classify_v2`）。nuPlan 上需要用地图 API（相邻车道）和 `anns`（障碍物、车辆）重写。不做的话，nuPlan 线只能说明"指令有用"，不能支持 OPD 的结论。
- **prompt 调得越准，OPD 的空间越小**。盈亏点是 a < f/(f+e)，例如 f=0.7、e=0.05 时约 93%。8B 的指令可接受率在 75–90% 时最好说明问题。CoLT 上用和 NavSafe 统一的 prompt，不为 CoLT 单独追求最高分，这也符合"一个系统"的原则。
- **CoLT 上 OPD 能纠正的错误**：绕行方向错、AROUND 和 BLOCKED 混淆、不必要的减速或停车。**VLM 漏报（NONE）时没有指令可纠正**，只能靠 VLA 自己看到。

## 6. 让结果有说服力的要求

1. **每个 benchmark 都做消融**：底座单独、底座 + VLM、底座 + GT 指令。
2. **通用能力不能掉**：Bench2Drive 和 NAVSIM 上至少与底座模型持平。
3. **VLM 和分类原则一致**：同一个 VLM、按反应方式分类、不说障碍物名字。
4. **闭环跑多个 seed**：CARLA 闭环方差大。
5. **两条线用同一套 cf 训练配方**：AutoVLA 的 cf 是从 SimLingo 移植来的。差异要列出来，例如 report 句子、suggestion 词表。
6. **CoLT 的背景帧可能来自 navtrain**，要说明这一点。CoLT 考的是插入的障碍物，影响有限。

## 7. 待处理

- cf 数据里的变道样本约 30% 配 KEEP 速度，而 CoLT 要的是 slow down + nudge。需要决定是否让变道都配减速（CoLT 测评按映射表处理，见 4.4）。

## 8. 执行顺序

1. **选 VLM、调 CoLT prompt**，用固定映射出分。这一步不依赖训练，可以马上做。
2. ~~确定前视单帧 / 多帧的混合比例，训新的 checkpoint。~~ 11-05 定不做，见第 3 节。
3. nuPlan 线的 OPD：用 nuPlan 地图和 `anns` 重写题库规则和屏蔽，teacher 打分，从新 checkpoint 出发训练。
4. 用 cf SFT 和 OPD 两个 checkpoint 跑 NAVSIM、NavSafe、CoLT，加上 5.1 的四行对比。
