# AutoVLA 反事实指令 SFT：数据准备和代码改动计划

日期：2026-10-26。仓库：`doc_vla_search`。

参考：
- simlingo 侧的同类改动：`doc_drive_search/docs/1019/cf_sft_v1_plan.md`、`cf_sft_v1_code_changes.md`、`change_lane_data.md`
- 本仓库：[1022 可行性](../1022/navsafe_feasibility.md) §4.2、[1025 目标场景](../1025/navsafe_target_scenarios.md)、[0901 CoT 采样](../0901/sft_cot_sampling.md)

## 0. 已定

- 只做 SFT，不动 teacher / RL（`models/utils/teacher.py`、`rl_rewards.py` 只读不改）。
- 指令用 `<PLAN>` 词表的说法，和 CoT 决策句用同一套动词，不引入自然语言改写。
- 从 Qwen2.5-VL-3B 基座重新训。代码 push 到 cluster 上训练。
- **先准备数据，再改训练代码。** 数据在本机（gcis）生成，这里有 `navtrain_metric_cache`、nuPlan 地图和 72 核。
- ~~CoT 数据（`navtrain_cot` / `navtrain_cot_val`）由用户下载，本机不重新生成。~~ 2026-10-27 改：103k CoT 和生成器都只在 brev 上，丢了，已在本机重新生成（§3 D0）。
- （2026-10-26 补）反事实分两部分：**速度**（stop / slow down / speed up）和**换道**（左 / 右），对应 simlingo 第一版的 5 个动作。换道必须有，NavSafe 的 I-3 这类场景要靠它。
- （2026-10-26 补）方向词表新增 `LANE_LEFT` / `LANE_RIGHT` 表示换道；原来的 `LEFT` / `RIGHT` 不动，继续表示跟着路转弯或过弯。
- （2026-10-26 补）**先做 D3 合成换道。**

## 1. 和 simlingo cf SFT 的对应关系

| simlingo（CARLA） | AutoVLA（nuPlan） |
|---|---|
| `Scene report` + `Suggested action` | 一样，两段都放（10-27 用户定加 report，见 §2）。report 按动作随机配、不看画面，和 simlingo 第一版相同；词表扩成 7 句 |
| 动作句子：Stop and wait / Slow down / Accelerate / Change lane left·right / Wait | suggestion：come to a stop / slow down / speed up / change lanes to the left·right。答案的 `<PLAN>`：速度 {STOP, DECELERATE, KEEP, ACCELERATE} × 方向 {STRAIGHT, LEFT, RIGHT, LANE_LEFT, LANE_RIGHT} |
| 答案 `X. Waypoints:`，轨迹由 waypoint head 回归 | 答案是 `<think>I should X: <PLAN>S,D</PLAN></think><answer>…</answer>`，轨迹是同一条序列里的 10 个 action token |
| dreamer stop / slower / faster | 沿专家路径重新配速（`models/utils/feasible.py` 的 `synth`），再用 PDM 检查 |
| 换道 v2（重新生成） | 新写：用 nuPlan 地图取相邻同向车道，从 t=0 开始平滑横移 |
| Wait | 不单独设词。等空当就是 `STOP` / `DECELERATE,STRAIGHT` |
| 第 4 类：专家真实绕障 | ❌ 不做（2026-10-27 定，见 §3 D4）。原计划是 navtrain 里专家自己的换道（约占 navtrain 的 1%） |
| 解耦样本 10%，只训轨迹 | 一样。只训 action token，文字 token 的 label 设成 -100 |
| 推翻样本（给 OPD 热启动） | 这一版不做 |
| `dreamer_ratio` 0.25 | `cf_ratio` 0.25 |

## 2. 样本格式

普通样本（占 75%）和现在的 CoT SFT 完全一样：`reason_then_act` 75%、`act_directly` 25%（`cot_ratio: 0.75`，`cot_in_prompt_ratio: 0.0`）。

cf 样本（占 25%）在 user prompt 的 base 后面加 scene report 和 suggestion，**后面不再接 `Explain your reasoning first.`，答案没有 `<think>`、没有决策句，只有 `<PLAN>` 和动作**（2026-10-27 用户定）：

```
# 照做（suggestion 和答案一致）
user:   … plan the action trajectory for the autonomous vehicle over the next five seconds. Scene report: A pedestrian is on the road ahead. Suggested action: slow down.
target: <PLAN>DECELERATE,STRAIGHT</PLAN>\n<answer>\nThe final output action is: <action_…>×10\n</answer>

# 速度反事实：suggestion 只给速度，方向从轨迹算（路口前可能是 LEFT/RIGHT）
user:   … Scene report: A stopped vehicle is ahead in the ego lane. Suggested action: slow down.
target: <PLAN>DECELERATE,LEFT</PLAN>\n<answer>…</answer>

# 换道：suggestion 只给换道，速度从轨迹算
user:   … Scene report: An obstacle is ahead in the ego lane. Suggested action: change lanes to the right.
target: <PLAN>KEEP,LANE_RIGHT</PLAN>\n<answer>…</answer>

# 解耦（suggestion 被换成别的动作，report 仍和答案相符；只训 action token）
user:   … Scene report: An animal is on the road ahead. Suggested action: speed up.
target: <PLAN>STOP,STRAIGHT</PLAN>\n<answer>[停车轨迹]</answer>
```

- 普通样本有三种模式（`reason_then_act` 带 `Explain your reasoning first.`、`act_directly` 带 `Answer directly.`）；cf 样本是第四种，靠 prompt 里有没有 `Scene report:` / `Suggested action:` 区分，结尾不加任何指令。
- 影响到别处的：`rl_rewards` 的格式检查要求 `reason_then_act` 有非空 `<think>`，cf 格式不走它（这一版不做 RL，先记下）。`parse_plan` 是在整段输出里找 `<PLAN>`，不受影响。测试时（NavSafe 接 VLM）带 report + suggestion 的 prompt 也用这个格式，1025 §6 oracle 实验原来预填的是 `<think>…<PLAN>…</think>`，接新 checkpoint 时要改成预填 `<PLAN>…</PLAN>`。
- D6 索引里的 `decision`（"I should …"整句）不再用，训练时只取 `plan_speed`、`plan_path`。

**Scene report（2026-10-27 用户定）**：
- 只给 cf 样本加，普通样本（75%）不加，和现有 CoT SFT 逐字一致（和 simlingo 一样）。
- **按答案的动作随机配一句相符的，不看画面**（simlingo 第一版的做法）。report 当作真的信息；测试时由外部 VLM 输出类别，换成同样的句子。
- 固定 7 句，不做同义改写：

  | report | 可以配的动作 | NavSafe 对应 |
  |---|---|---|
  | `A pedestrian is on the road ahead.` | STOP / DECELERATE | R-1、R-3 |
  | `A cyclist is on the road ahead.` | STOP / DECELERATE | R-2 |
  | `An animal is on the road ahead.` | STOP / DECELERATE | R-4 |
  | `A stopped vehicle is ahead in the ego lane.` | STOP / DECELERATE / 换道 | 前方静止车辆、I-3 |
  | `An obstacle is ahead in the ego lane.` | STOP / DECELERATE / 换道 | I-1、I-2 |
  | `A vehicle is cutting into the ego lane.` | STOP / DECELERATE | V-2 |
  | `No hazard is in the ego lane.` | ACCELERATE / KEEP | 起步、危险离开后 |

- 换道样本的 report 只配"停着的车 / 障碍物"，速度词由轨迹决定（多是 DECELERATE / KEEP）。
- 解耦样本里 report 跟着答案配（和答案相符），只有 suggestion 是干扰项。
- 每条样本在 `__getitem__` 里现抽 report（每个 epoch 可能不同），不写进 D6 索引。
- 已知的代价：report 多数和画面对不上（cf 帧里有和动作相符的真实依据的只有约 20–30%），模型要学的是"相信 report"。普通样本没有 report，"prompt 里有 report"就等于"这是 cf 样本"，开环检查时要看一下有没有走这条捷径（§5）。

- 动词和 `teacher.py` 的 `SPEED_VERB` / `speed_verb()` 保持一致：
  - STOP：v0 ≥ 2 m/s 时是 `come to a stop`，否则是 `remain stopped`
  - DECELERATE：`slow down`
  - KEEP：`hold my current speed`
  - ACCELERATE：`speed up`
  - 方向是 LEFT/RIGHT 时后面加 ` and bear left/right`（和现有 CoT 一致），是 LANE_LEFT/LANE_RIGHT 时加 ` and change lanes to the left/right`
- suggestion 只有 5 个：`come to a stop`、`slow down`、`speed up`、`change lanes to the left`、`change lanes to the right`。KEEP 不当 suggestion（simlingo 也没有），只出现在答案的速度位上。
- 判断照做还是推翻只看 `<PLAN>` 里对应的那一维：速度类 suggestion 看速度，换道类看方向。
- 上面"动词"那条现在只用于普通 CoT 样本；cf 样本的答案里没有动词了。
- **KEEP 类样本只用在解耦样本里**（10-27 用户定）：KEEP 不当 suggestion，所以 KEEP 的 cf 样本 suggestion 换成别的动作、report 配 `No hazard`、只训 action token。换道样本答案里的 KEEP（如 `KEEP,LANE_RIGHT`）照常训。

## 3. 数据准备（先做）

### 当前状态（2026-10-27，数据准备已完成）

| | train | val | 位置 | 说明 |
|---|---|---|---|---|
| CoT（D0）| 101,288（有 CoT 100,222）| 2,000（有 CoT 1,977）| `/scratch/mh2803/vla/nuplan/navtrain_cot`、`navtrain_cot_val` | 重新生成的，见 [cot_regen_navtrain.md](cot_regen_navtrain.md)。相机路径是相对路径，config 要设 `sensor_data_path` |
| 速度反事实（D2）| 224,897 | 4,444 | `cf_navtrain_v1/speed/` | 见 [cf_speed_proto_examples/](cf_speed_proto_examples/README.md) |
| 合成换道（D3）| 4,897 | 97 | `cf_navtrain_v1/lane_change/` | 见 [cf_lc_proto_examples/](cf_lc_proto_examples/README.md) |
| **cf 索引（D6，训练直接用这个）**| **209,139**（98,332 帧）| **4,143**（1,944 帧）| `cf_navtrain_v1/cf_navtrain_v1_{train,val}.npz` | D2 + D3 合并；去掉 PDM TTC = 0 的 19,481 条、专家自己也换进目标车道的换道 645 条、比专家更贴着别的车的换道 514 条、token 往返误差 > 1 m 的 15 条 |

- D1、D4 不做；D5（val）跟着 D0、D2、D3 一起生成了。
- 示例图三部分都有：[cot_regen_examples/](cot_regen_examples/README.md)、[cf_speed_proto_examples/](cf_speed_proto_examples/README.md)、[cf_lc_proto_examples/](cf_lc_proto_examples/README.md)。
- 下一步是 §4 的训练代码（C1–C6）。

输出放在 `/scratch/mh2803/vla/nuplan/cf_navtrain_v1/`。和 simlingo 的 `cf_lane_change_v2.npz` 一样，**只存一个小的索引文件**：每个候选存 token、类型、`<PLAN>`、10×3 轨迹和 PDM 子分。训练时按 token 读原 JSON，再把 `gt_trajectory` 换掉（`TrajectoryTargetBuilder.compute_targets` 会现场把 `gt_trajectory` 编码成 token，见 `navsim/navsim/agents/autovla_agent.py:309`）。这样不用复制 10 万个 JSON，也不用处理图片路径。

### D0. 合并 CoT 数据

- **2026-10-27 已做，改成重新生成**：103k 的 CoT 和生成器都只在 brev 上，丢了。照 HF 上 166k CoT 的输出反推规则，在 Impromptu-VLA 里重写了 `build_navsim_cot.py` 和 `merge_cot_into_samples.py`。反推不出来的三类（切入、停车标志、前方放空）按用户选的方案 B 重新定义。输出 `/scratch/mh2803/vla/nuplan/navtrain_cot`（101,288 条，有 CoT 100,222 条，和老版条数完全一样）和 `navtrain_cot_val`（2,000 条）。相机路径已是相对路径。详见 [cot_regen_navtrain.md](cot_regen_navtrain.md)。
- 原计划：按 token 把下载来的 `cot_output` 合并进本机的 `navtrain_nocot` / `navtrain_nocot_val`（文件名就是 `<token>.json`），输出 `navtrain_cot` / `navtrain_cot_val`。
- **相机路径改成相对 `sensor_blobs/` 的相对路径**，各机器在 config 里设 `sensor_data_path`（`sft_dataset.py:142` 已经支持）。现在 JSON 里是本机的绝对路径，`os.path.join` 遇到绝对路径会丢掉前缀，放到 cluster 上会找不到图。
- 检查：
  - token 一一对应；
  - 两边的 `gt_trajectory` 逐元素相等，防止两边数据版本不同；
  - 覆盖率约 98.9%（0901 文档的数字）。

### D1. 修正 CoT 里专家换道的方向标签（搁置：做法依赖 D4 的地图检测器，D4 不做）

- 现有 CoT 的 `<PLAN>` 方向是 `plan_from_track` 从轨迹几何算的：末端朝向 ≥ 30° 或横纵比 ≥ 0.2 才算 LEFT/RIGHT。现有的 LEFT/RIGHT 约 92% 落在 turn left / turn right 上，就是路口转弯。
- 车速稍高时，换道的横纵比 < 0.2，会被标成 STRAIGHT。示例图和分组统计见 [lc_label_examples/](lc_label_examples/README.md)：
  - 🔴 第一版写的"313 条像换道的里 311 条被标成 STRAIGHT"**是错的**。那个筛法（自车坐标系末端横移 ≥ 2.5 m、朝向 < 10°）抓到的大部分是弯道，标签没问题。
  - 真换道确实被标成 STRAIGHT（例如 `eb6b0ad19067509d`：锥桶封道，往右换一条道），但量级只有几百到一千帧，约占 navtrain 的 1%。
  - 用自车坐标系的横移判断，连方向都可能判反（`9f6599ad5f5158cd`）；只看相对 PDM centerline 的偏移，在路口、车道分叉、centerline 起点不在自车车道时会误检。
- 如果 cf 换道样本标 `bear right / RIGHT`，而专家真实换道仍标 STRAIGHT，两种标签互相矛盾，只是规模不大。
- 做法：用 D4 的地图检测器找出真换道，把这些帧的决策句改写成 `… and change lanes to the left/right: <PLAN>S,LANE_LEFT/LANE_RIGHT</PLAN>`。写法照 `teacher.splice` 的思路，散文和 tag 一起改。弯道（LEFT/RIGHT/STRAIGHT 的其余帧）不动。

### D2. 速度反事实（对应 dreamer 的 stop / slower / faster）

- 对每个 token 用 `FeasibleSet.synth` 生成 4 档：STOP −2.5、DECELERATE −1.5、KEEP 0、ACCELERATE +1.5 m/s²。路径用专家路径，不够长时沿 centerline 续接。然后跑 `pdm_score`。
- **标签用 `plan_from_track` 在合成轨迹上重新算**，不能直接用档位名。档位名和标签规则对不上，例如：
  - DECELERATE 档在 v0 < 6.5 m/s 时，3 s 后车速已经低于 2 m/s，会被标成 STOP；
  - STOP 档在 v0 ≥ 9.5 m/s 时会被标成 DECELERATE。

  重新算出的速度词和档位不一致时，按算出来的词用，或者直接丢掉（待定，倾向丢掉，让每个词对应的轨迹更干净）。
- **可行性过滤**（simlingo 第一版不过滤；我们有 PDM，直接过滤）：
  - NC = 1、DAC = 1、DDC = 1。
  - **另外查红灯**：🔴（10-27 更正）不是"NC 跳过红灯"那么简单：navsim v1 建 metric cache 时根本没把红绿灯传进 PDMObservation（`metric_cache_processor.py` 只调了 `update_detections_tracks`），observation 里一个红灯都没有，PDM 打分从来不管红灯。红绿灯要从 `navsim_logs` 每帧的 `traffic_lights` 取。
  - ACCELERATE 再加一条：末速不超过地图限速（`lane.speed_limit_mps`）。
- **从静止起步**：1025 的 A 档需要 STOP → ACCELERATE 能重新起步。v0 < 0.5 m/s 而 ACCELERATE 可行的帧单独记一个子类，训练时保证有一定比例。
- 规模估计：10 万 token × 4 档 = 40 万次 PDM，按约 0.1 s/次、48 进程算，约 15 分钟。
- **原型**（2026-10-27，已做，等看过再跑全量）：[tools/cf_data/speed.py](../../tools/cf_data/speed.py)，和换道同一批 300 帧 → 1200 个候选 → **655 条通过**（STOP 260、DECELERATE 153、KEEP 148、ACCELERATE 94；全量估计约 22 万条）。示例图和统计见 [cf_speed_proto_examples/](cf_speed_proto_examples/README.md)。定下 / 改过的：
  - 每类按顺序试几档加速度，取第一档标签对得上的：STOP −2.5；DECELERATE −1.0、−0.5；KEEP 0；ACCELERATE +1.0（加到限速为止）。都对不上就丢。结果是 STOP 只有 v0 ≲ 10 m/s 的，DECELERATE 只有 v0 ≳ 3.5 m/s 的，KEEP 只有 v0 ≥ 2 m/s 的；
  - ACCELERATE 用 +1.0 不用 +1.5（9 月的 step0 就记过 +1.5 持续 5 s 太猛）；
  - 横向加速度 ≤ 3.0 m/s²：快的档会开过专家路径终点、沿路线高速拐弯，PDM 不拦，tokenizer 也表示不了；
  - 红灯从日志取，只看路线上的 connector，往红灯 connector 里开得比 t = 0 时更深就算闯。专家自己被判闯的 1%；
  - 限速只取 lane 上的，不取路口 connector 上的转弯限速；拉斯维加斯 lane 上本身有不少 4.5 / 2.2 m/s 的限速（酒店接送区），照地图办；
  - 解码后的标签也要对得上。
  - 高速 STOP（加一档 −3.5）：10-27 用户定先不加，v0 > ~10 m/s 没有 STOP 样本；
  - "有车紧跟还急刹"的 STOP 不查（日志车辆不反应，PDM 不算自车责任）；
  - 速度约 22 万条、换道约 5000 条，差很多：配比在改训练代码（§4 C2 / C5）时定。
- **全量**（2026-10-27，用户看过原型后同意，已跑）：68 进程 10 分钟，输出 `/scratch/mh2803/vla/nuplan/cf_navtrain_v1/speed/{navtrain,navtrain_val}/`。navtrain **224,897 条**（STOP 89,825、DECELERATE 53,053、KEEP 49,083、ACCELERATE 32,936，其中从静止起步 7,443），navtrain_val **4,444 条**。和专家速度词不同的 140,225 条。第一次全量发现专家停着时 GT 抖动导致合成停车轨迹朝向翻 180°，加了 `forward_only` 后重跑。详见 [cf_speed_proto_examples/](cf_speed_proto_examples/README.md#全量结果)。

### D3. 合成换道（对应 simlingo 第 1 类 / 换道 v2）

**先做这一步。** 思路和速度反事实对称：速度是"路径不变、重新配速"，换道是"车速用专家的、路径换到相邻车道"。画面是这一帧真实的，只换未来 5 s 的轨迹；其他车按日志回放，不会反应。

1. **找车道**：用 nuPlan 地图（`/scratch/mh2803/vla/nuplan/maps`；`navsim_logs` 里有 `map_location`）找自车所在的 lane，取它的 `adjacent_edges`。nuPlan 里同一个 lane group 的车道都是同向的（`adjacent_edges` 按 `lane_group_fid` 和 `lane_index` 找邻居），所以取到的天然只有同向车道。借对向车道（I-2）留到之后。目标车道的 baseline 接上后继车道，接够长度。
2. **横向**：从 t = 0 开始，用 smoothstep 从当前车道中心线过渡到目标车道中心线。不从 t = 0 开始的话，测试时每帧都重复给指令，模型会一直把横移往后推（simlingo 1014 闭环里车几乎不动）。
3. **纵向**：用专家每 0.5 s 走的距离。simlingo v2 第二轮修正就是因为不用专家车速，换道轨迹多走、冲过停止线。
4. **过渡时间 T**：先量 navtrain 里专家真实换道的分布再定，不照搬 CARLA 的 0.8–1.5 s。同时要满足 NavSafe I-3 的需要：大约 3 s 内横移 ≥ 3.8 m（1025 §4.1）。
5. **跟踪**：用 PDM 自带的 LQR + 自行车模型（`PDMSimulator.simulate_proposals`，`pdm_score` 内部也是先跟踪再打分）跟踪一遍，跟踪后的轨迹和设计的偏差 > 0.5 m 就丢，保证物理上开得出来。
6. **过滤**：
   - PDM 的 NC / DAC / DDC 全为 1（PDM 只查前 4 s）；
   - t = 0 时目标车道里紧挨着自车没有车；并入时和目标车道前后车的间距 ≥ max(2 m, 0.5 s × 车速)。用 `observation` 里日志车辆的未来位置算。PDM 的 NC 只看框有没有真的重叠，"擦着旁边车切进去"也算不撞（simlingo §3.1）；
   - 5 s 内不进路口（不进 lane connector）；
   - 自车在动（总位移 ≥ 5 m）。不加这一条会出现"收到换道指令，车速却是 0"的样本（simlingo 那边 42% 是这种）；
   - 目标车道宽度 ≥ 2.5 m。
7. **tokenizer 误差**：编码成 10 个 action token 再解码，统计和原轨迹的横向误差（1022 §4.2 要求先查这个）。先看分布再定阈值。
8. **标签**：方向按构造给 LANE_LEFT / LANE_RIGHT，速度用 `plan_from_track` 重算。
9. **原型**（2026-10-26，10-27 加路线检查）：[tools/cf_data/lane_change.py](../../tools/cf_data/lane_change.py)，navtrain 随机 300 帧 → 600 个候选 → **14 条通过（约 4.7%，全量估计 ~4700 条）**。示例图和统计见 [cf_lc_proto_examples/](cf_lc_proto_examples/README.md)。原型中定下 / 改过的：
   - 过渡时间 T ∈ U[3.0, 4.5] s。navtrain 专家换道（5 s 内换完的 505 个）横移 10→90% 用时中位 2.76 s，换算成 smoothstep 总时长中位 4.5 s；下限 3 s 照顾 I-3；
   - 跟踪误差只量横向，阈值 1.0 m（PDM 的 LQR 纵向本来就跟不紧，专家 GT 横向偏差 p95 0.91 m）；
   - 横向斜率上限 tan 20°（挡低速时横着挪）。原型先用 25°，5 m/s 减速中并入、斜率正好 25° 的样本看图后不要，10-27 收到 20°；
   - 路口：keep forward 且两条车道 5 s 内都直（朝向变化 < 20°）时允许直行穿过；
   - 目标车道 5 s 内必须一直平行（挡右转专用道）；
   - 路线（10-27 加）：目标车道 100 m 内的第一个出口至少有一条在 metric cache 的 `route_lane_ids` 里。挡"换进转弯专用道"和"路线要转弯却往反方向换"，PDM 不罚这两种。用路线不用导航指令：指令只看较近的一段，路线 51 m 后右转时指令还是 keep forward；
   - nuPlan 地图分不出实线和虚线（车道之间的 `boundary_type_fid` 基本都是 0），压实线换道挡不住，路口前的那部分靠上一条路线检查挡；
   - tokenizer 往返误差 p50 0.07 / max 0.43 m，不需要额外过滤。
   - （10-27 加）自车在中心线起点之前时往回接前驱再投影（否则设计轨迹第一步就跳出去）；每 0.5 s 走的距离要和专家对得上（`geometry`）；横向加速度 ≤ 3.0 m/s²；红灯（从日志取）。T 改成按 token 单独抽，结果和进程数无关。
10. **全量**（2026-10-27，已跑）：navtrain **4,897 条**（4,697 帧，左 3,095 / 右 1,802），navtrain_val **97 条**。输出 `/scratch/mh2803/vla/nuplan/cf_navtrain_v1/lane_change/{navtrain,navtrain_val}/ok.jsonl`。统计见 [cf_lc_proto_examples/](cf_lc_proto_examples/README.md)。

局限：其他车不反应；第 5 s 不查；大部分帧画面里没有换道的理由，理由只来自 suggestion（和 simlingo 第 1 类一样。有理由的换道原计划靠 D4 补，D4 不做，所以这一版没有）。

### D4. 专家真实换道（对应 simlingo 第 4 类）—— ❌ 不做

**2026-10-27 用户定：不做。** 下面的设计只留作记录。D1 的做法用的是这里的检测器，所以 D1 一起搁置。D3 的过渡时间 T 用的专家换道时长分布已经量过（§3 D3 第 9 条），不受影响。

- 用 nuPlan 地图的 lane 拓扑判断：t = 0 和末端都在 lane 里（不在 lane connector 或路口里），而且末端所在的 lane 是起点 lane（或它的后继）的左右邻居（`adjacent_edges`）。
  - 不用自车坐标系的横移，弯道上会判错，方向也可能判反（simlingo §5.4 也踩过）。
  - 也不能只看相对 PDM centerline 的偏移：全量 keep forward 里这样算出来横移 ≥ 2.5 m 的有 2,060 帧，看图后估计只有一半左右是真换道（[lc_label_examples/](lc_label_examples/README.md) 的 D、E 组）。
- 这类样本的画面里有真实的换道原因。suggestion 和答案都是专家的方向，轨迹是专家的 GT。
- 估计量：几百到一千帧（约占 navtrain 的 1%），以地图检测器的全量结果为准。比 simlingo 第 4 类少，训练时要按比例重复抽。
- 同时输出专家换道的横移时长分布，给 D3 用。

### D5. val 集

- 对 `navtrain_cot_val`（2000 条）同样跑 D2–D3，用于开环检查（§5）。
- NavSafe 的场景来自 nuPlan test split，不进训练也不进 val（1022 §4.3）。

### D6. 导出和抽查

- 导出 `cf_navtrain_v1_{train,val}.npz`，附一份统计：每类数量、每个 `<PLAN>` 组合的数量、被过滤的原因各占多少。
- 每类存 10 张示例图：左边前视图，右边 BEV（地图、日志车辆、专家轨迹、cf 轨迹、tokenizer 解码后的轨迹）。人工看过再接进训练。
- **已做（2026-10-27）**：[tools/cf_data/export_index.py](../../tools/cf_data/export_index.py)，35 秒。输出 `/scratch/mh2803/vla/nuplan/cf_navtrain_v1/cf_navtrain_v1_{train,val}.npz`（23 MB / 0.5 MB，`np.load` 不需要 `allow_pickle`）和 `stats.json`。
  - 字段：`token`、`cls`（STOP / DECELERATE / KEEP / ACCELERATE / LANE_LEFT / LANE_RIGHT）、`suggestion`（§2 的 5 个动词，KEEP 为空）、`decision`（答案决策句，含 `<PLAN>`）、`plan_speed`、`plan_path`、`v0`、`traj`（N×10×3，和 JSON 的 `gt_trajectory` 同格式）、`pdm`（N×6：score nc dac ep ttc ddc）、`token_err`、`start_from_stop`、`differs_from_expert`、`expert_speed`。
  - 决策句用 `plan_from_track`（按 float32 轨迹重算）+ `teacher.speed_verb`，和 CoT 同一套动词。换道是 `… and change lanes to the left/right: <PLAN>S,LANE_LEFT/LANE_RIGHT</PLAN>`。STOP 的动词按**当前车速**选（v0 ≥ 2 是 come to a stop），不按轨迹前 0.5 s 的平均速度：−2.5 刹车会把后者拉到 2 以下（4,789 条受影响，已修）。
  - 训练样本示例（每种类别 × 方向 2 张）：[cf_export_examples/](cf_export_examples/README.md)。
  - train 各类：STOP 87,693、DECELERATE 48,527、KEEP 38,811、ACCELERATE 31,172（其中从静止起步 7,261）、LANE_LEFT 1,973、LANE_RIGHT 963。和专家速度词不同的（真反事实）：STOP 68,374、DECELERATE 33,589、KEEP 12,431、ACCELERATE 10,006；换道都算。
  - val 各类：STOP 1,750、DECELERATE 970、KEEP 758、ACCELERATE 601、LANE_LEFT 46、LANE_RIGHT 18。
  - 导出时的过滤（10-27 看图后加的两条都在 `export_index.py` 里做，`lane_change.py` / `speed.py` 没重跑）：
    - token 往返误差 > 1 m：15 条。
    - 🔴 **PDM TTC = 0：速度 18,679 条、换道 802 条**。合成时只要求 NC / DAC / DDC，TTC = 0 是按当前车速 1 s 内会撞上前方的车。用户看出 `lane_right_0`（`f80f84c1127d5a59`）不像能往右换：并进右车道时离那里的车最近 1.0 m，之后一直贴着 1.5 m。各类原来的 TTC = 0 比例：KEEP 21%（专家在前车后面减速，KEEP 越贴越近）、换道 18%、DECELERATE 8.5%、ACCELERATE 5.3%、STOP 2.4%。
    - 🔴 **换道末端离专家 < 半个车道宽：645 条**（先过 TTC 再算；只算这一条是 702 条，14%）。t=0 专家还在本车道中间附近（过了 `not_in_lane`），但车头已经偏向目标车道、5 s 内自己换过去了，合成的"换道"就是专家原本的轨迹，不是反事实（例：`f2f3fc476dc45ea5`，往右换道的 cf 末端只离专家 0.12 m，车道宽 3.48 m）。`LC_MIN_SEP = 0.5`。
    - 🔴 **换道比专家更贴着别的车：514 条**（用户问 `c18eb96197aa5b1c` 是不是撞了以后查的；那条没撞，cf 最近 2.2 m、专家 2.7 m，是图把 t=2.5 s 的他车和 t=5 s 的 cf 叠在一起了）。量 5 s 内每 0.5 s 自车框到最近的**车 / 自行车**框的距离（PDM 只查 4 s），**cf < 2.5 m 且比专家同一帧的最小值还近**就丢（`LC_GAP_ABS = 2.5`、`LC_GAP_REL = 0`，`lane_change.clearance`）。
      - 调过三次：先是 < 2.0 m 且近 1.0 m 以上（391 条）；用户看了 `c18eb` 要求也丢，收到 2.5 / 0（646 条）；用户又看出 `lc_tight` 里有的"不用去掉"——查下来离得近的是路边的杆子（`a87edb35ddf85c5f`，0.09 m）和酒店门口的花盆（`2332cf379a5f5bf9`），这些静止物体（GENERIC_OBJECT、锥桶、施工牌、护栏）占了 30%，离它们近由 DAC 管，于是只算车和自行车（514 条）。
      - 留下的例子：`a47a84c2cb66548d` 往左并道时贴上左车道从后面 11 m/s 开上来的车（0.07 m）；`83fcaecf1f145da3` 斜着插进左车道排队的车之间只隔 1.1 m（专家 4.0 m）。
      - 不用纯绝对阈值：相邻车道并排开时车框本来就只隔 1 m 左右，专家自己的 p5 是 0.82 m，< 2.5 m 一刀切会去掉 41%。按 (token, 方向) 判，同一帧另一边的换道不受影响。
  - 示例图按部分各自出过（见上面"当前状态"），没有再单独出 D6 的图。

## 4. 训练代码改动（数据定下来以后再做）

| # | 文件 | 改动 |
|---|---|---|
| C1 | `dataset_utils/prompt_spec.py` | `user_text` 加可选参数 `suggestion=""`，不为空时在 base 后面加 ` Suggested action: {suggestion}.`。新增 `decision_text(speed, path, v0)`，返回决策句和 `<PLAN>`，规则和 `teacher.speed_verb` 一致。**`use_cot: false` 和现有三档的 prompt 必须逐字节不变**（0901 文档的要求），改完用 `.bak` 比对确认。 |
| C2 | `dataset_utils/sft_dataset.py` | 读 `model.cf`（`path`、`ratio`、`type_probs`、`decouple_ratio`）。`__getitem__` 先按 `ratio` 决定这一条是不是 cf 样本，再按 `type_probs` 选类别，从这一类的池子里抽 token，读原 JSON 并换掉 `gt_trajectory`。解耦样本把 suggestion 换成另一个动词，并设 `skip_text_loss=True`。`cf.path` 为空时行为和现在完全一样。 |
| C3 | `dataset_utils/sft_dataset.py` 的 `DataCollator` | `skip_text_loss` 的样本，把 assistant 段里所有 `< action_start_id` 的 label 设成 -100，只留 10 个 action token。`training_step` 里的 `train_text_loss` 会自动排除这些样本，loss 本身不用改。 |
| C4 | `dataset_utils/sft_dataset.py` 的像素 | `min_pixels` / `max_pixels` 现在写死成 28·28·128（第 286 行），推理用的是 140（hparams 和 `get_prompt`）。改成读 `model.video`。这次是重新训练，可以一并统一。 |
| C5 | `config/training/1026/` | 新的 yaml：CoT 0.75 + cf 0.25。数据路径和 `sensor_data_path` 写成 cluster 上的路径。global batch 保持 32，accum 按 cluster 的 GPU 数设。 |
| C6 | `dataset_utils/sft_dataset.py` 的 `__main__` | smoke test：抽 2000 条，检查 suggestion 和答案都在词表里；照做样本两者相等；解耦样本当且仅当 `skip_text_loss`；cf 样本的 `gt_action` 等于索引里的轨迹编码出来的 token。打印各类占比。 |

建议的类别比例（占 cf 样本）：速度 60%（按速度词均匀抽，其中从静止起步不少于 5%）、合成换道 25%（左右各一半）、专家换道 15%。解耦样本 10%。D4 不做，专家换道这 15% 没有了，比例在改训练代码时重定。

### 实际改动（2026-10-27，已实现，和上表的差别）

上表是 10-26 写的计划，§2 后来改过格式，实际按下面做：

| # | 文件 | 实际做的 |
|---|---|---|
| C1 | `dataset_utils/prompt_spec.py` | 加第四种模式 `counterfactual`：`user_text(..., report=, suggestion=)` 返回 base + ` Scene report: {report} Suggested action: {suggestion}.`，后面不接任何指令。加 `CF_SUGGESTIONS`（5 个）、`CF_REPORTS`（7 句）、`CF_ACTION_REPORTS`（动作 → 可配的 report）、`PLAN_PATH_WORDS`（加 `LANE_LEFT/RIGHT`，`rl_rewards.PATH_WORDS` 不动）。没有 `decision_text`（cf 答案里没有决策句了）。和 git HEAD 的旧版逐字比对：3 万组随机输入 × 3 种模式 × 开关 CoT 全部一致 |
| C2 | `dataset_utils/sft_dataset.py` | cf 索引按 split 放在 `data.<split>.cf_index`，比例在 `model.cf`（`ratio`、`decouple_ratio`、`type_probs`、`decouple_type_probs`）。`_sample_cf` 先定是不是解耦，再按比例选类别、池子里随机抽一条、按答案的动作配 report。cf 样本不用 `idx` 这一帧，按 token 读 `navtrain_cot` 的 JSON 再换 `gt_trajectory`。答案 `<PLAN>S,D</PLAN>\n<answer>…</answer>`，`has_cot=False`。KEEP 放在 `type_probs` 里会报错。没配 `cf_index` 时不调任何 `random()`，和原来完全一样 |
| C3 | `DataCollator` | 加 `action_start_id` 字段（`run_sft.py` 从 config 传）。`skip_text_loss` 的样本把 `< action_start_id` 的 label 全设 -100。标记不放进 batch（`AutoVLA.forward` 会把 batch 里的其余键原样传给 VLM） |
| C3' | `models/autovla.py` | `train_text_loss` / `val_text_loss` 改成**无条件** log：原来 `if _tm.any()` 时才 log，解耦样本（batch_size=1 时整条没有文字 token）会让这个 rank 跳过 sync_dist 的 all-reduce，DDP 死锁（09-02 挂过同类的）。没有文字 token 时记 0 |
| C4 | `sft_dataset.py` | 像素读 `model.video`，没写时仍是 28·28·128 |
| C5 | `config/training/1026/` | `qwen2.5-vl-3B-nuplan-cf-sft-navtrain.yaml`（正式，数据路径暂时是本机的）、`…-cf-sft-debug-2gpu.yaml`（本机两卡调试：40 步、每 20 步验证 8 条、csv）。`run_sft.py` 加可选的 `training.max_steps` / `val_check_interval` / `limit_val_batches`，不写时和原来一样 |
| C6 | `sft_dataset.py` 的 `__main__` | `python -m dataset_utils.sft_dataset --config training/1026/<name> --n 200`。逐条查：结尾只有 report + suggestion、report 和答案相符、照做 ⇔ suggestion 等于本类、KEEP 只当解耦、`skip_text_loss` ⇔ 解耦、答案无 `<think>`、`<PLAN>` 在词表里且和类别一致、gt 轨迹就是 cf 轨迹、collator 对解耦样本只留 10 个 action label。200 条 0 错误 |

类别比例（10-27 定的默认值，在 config 里改）：照做样本 STOP / DECELERATE / ACCELERATE 各 25%、LANE_LEFT / LANE_RIGHT 各 12.5%；解耦样本（cf 的 10%）STOP / DECELERATE / KEEP / ACCELERATE 各 20%、换道各 10%。换道池子只有约 2.9k 条，每 epoch 约 6k 条换道样本，每条重复约 2 次。

本机调试：`bash scripts/1027/run_cf_sft_debug_2gpu.sh`（GPU 1,2；NCCL P2P / SHM 都关）。本机原来没有基座权重，10-27 下到 `/scratch/mh2803/vla/checkpoints/Qwen2.5-VL-3B-Instruct`，仓库里 `./Qwen2.5-VL-3B-Instruct` 是软链接（已在 .gitignore）。

## 5. 验证

1. **数据**：§3 D6 的统计和示例图，加上 §4 C6 的 smoke test。
2. **试训**：在本机单卡跑 300 步，确认 loss 在降、没有 NaN、`train_text_loss` 正常，再 push 到 cluster。
3. **开环**（val，贪心解码）：
   - 照做率：生成的 `<PLAN>` 等于 suggestion 的比例；
   - 运动学一致：`plan_from_track(解码出的轨迹)` 等于 suggestion 的比例，按类别分开统计；换道要单独看 2–3 s 内的横移是否 ≥ 0.6 个车道宽；
   - shortcut 检查：teacher forcing 一个和 suggestion 不同的 `<PLAN>`，看轨迹跟着 `<PLAN>` 走还是跟着 suggestion 走；
   - report 检查：固定 suggestion，只换 report（7 句轮流换），看 `<PLAN>` 和轨迹变不变；把结果记下来，预期是基本不变（见 §6）；
   - 不带 suggestion 时，`val_action_top1` 不低于 103k CoT 基线。
4. **闭环**：
   - navtest PDMS 不退化（1022 §9）；
   - NavSafe 在 ztao-SCU 上用 `autovla_ours` adapter 加 suggestion，先跑 1025 §6 的 oracle 场景。

## 6. 已知风险

- **LEFT/RIGHT 一词两义**：现有数据里基本是路口转弯，cf 里是换道，只能靠导航 command（keep forward 还是 turn）区分。D1 修完标签后，要在开环里分开统计这两种情况。
- **PDM 只检查 4 s**，日志车辆不会反应。换道"擦着旁边车切进去"也算不撞，所以 D3 的间距条件不能省。
- **"看见了却不减速"不一定是数据的问题**（1025 §4）。cf SFT 只保证"给了指令会照做"，指令对不对、来得及不及时，要靠 VLM。
- **report 可能被忽略**：照做样本里 suggestion 和答案一一对应，report 对答案是多余的，模型可能完全不看它（只有解耦样本里 report 有用，而解耦样本的文字 loss 是关的）。10-27 用户定**和 simlingo 一致，这一版不加对策**：simlingo cf SFT 也没加（`doc_drive_search/simlingo_training/dataloader/dataset_counterfactual.py:310-317`，只在 `docs/1019/cf_sft_v1_code_changes.md:191` 记了风险），让 report 起作用的是后面的 OPD（故意给错的 suggestion，teacher 按每种 report 的规则打分）。这一版只在开环里做 §5 的 report 检查，记录 report 是否被忽略，留给以后的 OPD。备选（没采用）：加 10–15% 只有 report、没有 suggestion 的 cf 样本。

## 7. 待定

| # | 问题 | 建议 |
|---|---|---|
| 1 | 专家真实换道的方向标签要不要从 STRAIGHT 改成 LEFT/RIGHT（D1） | 搁置：D4 不做，没有地图检测器。原建议是只改检测器确认过的真换道（约 1%），弯道不动。示例图见 [lc_label_examples/](lc_label_examples/README.md) |
| 2 | suggestion 在 prompt 里的写法和位置 | ✅ 10-27 定：base 后面接 `Scene report: {句子} Suggested action: {动词}.`，后面不再接任何指令；答案只有 `<PLAN>` 和动作，没有 `<think>`（见 §2） |
| 3 | 速度反事实重新算出的标签和档位不一致时怎么办（D2） | 丢掉 |
| 4 | 要不要加 scene report | ✅ 10-27 用户定：加。simlingo 式按动作随机配、不看画面；只给 cf 样本；固定 7 句（见 §2） |
| 5 | NavSafe 会传 `unknown` 导航 command，navtrain 里没有 | 训练时普通样本按 5% 的概率把 keep forward 换成 unknown（只对 keep forward，因为 NavSafe 是把 LANEFOLLOW 映射成 unknown） |
| 6 | ViT 冻结还是解冻 | 跟 103k CoT 一样冻结，方便和基线对比；之后再单独做一组解冻 |
| 7 | cluster 上 navtrain 的 sensor_blobs 放在哪、用几张 GPU | 需要用户提供，决定 C5 里的路径和 accum |
