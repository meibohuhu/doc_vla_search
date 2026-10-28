# D6 cf 索引：训练样本示例

2026-10-27。数据：`/scratch/mh2803/vla/nuplan/cf_navtrain_v1/cf_navtrain_v1_train.npz`（D6 导出，速度 + 换道合在一起）。画图：[tools/cf_data/render_index.py](../../../tools/cf_data/render_index.py)（`--pick 2`：每种（类别，方向）随机挑 2 条真反事实，seed 0）。计划：[cf_sft_autovla_plan.md](../cf_sft_autovla_plan.md) §3 D6。

都是**真反事实**：速度类的速度词和专家不同，换道类专家没换道。按训练时看到的样子写标题：
- 第二行 `prompt`：user prompt 末尾加的 suggestion（只有 5 个动词；KEEP 不当 suggestion）；
- 第三行 `answer`：答案 `<think>` 里的决策句，后面跟 10 个 action token。

每张图从左到右：
1. t = 0 前视图；
2. 大 BEV：他车的框是 **t = 0** 的位置，紫色细线是它接下来 5 s 往哪开（末端圆点 = 5 s）；灰点 = 专家 GT，彩色 = cf 轨迹，黑叉 = 编码成 action token 再解码回来的轨迹（训练真正学的）；灰色块是行人和静止物体（杆子、花盆、锥桶）；
3. **两个同一时刻的快照**（t = 2.5 s、t = 5 s）：彩色框 = cf 自车，灰虚线框 = 专家自车，紫色 = 这一刻的车。标题是这一刻 cf 离最近的车多远。**看会不会撞看这里**；
4. 每 0.5 s 的平均车速（灰 = 专家）；换道图是 cf 相对专家的横向偏移（左为正）。

标题第一行最后是 5 s 内 cf 和专家各自离最近的车 / 自行车的最小距离（车框到车框）。

⚠️ 大 BEV 上轨迹是 0.5–5 s 的，他车框只是 t = 0，轨迹线穿过某个车框不等于撞上（它早开走了）。前两版图把不同时刻的框叠在一起，`lane_right_0`（`c18eb`）和 `lane_left_0`（`e98b336770535de8`）都被看成"撞了 / 不行"：`e98b` 往左并到白车后面，同一时刻离车 2.5 s 时 8.4 m、5 s 时 5.2 m，所以保留。

## 速度

| 图 | v0 | prompt | answer | 专家 | 看什么 |
|---|---|---|---|---|---|
| `stop_straight_0` | 4.7 | come to a stop | come to a stop, STRAIGHT | ACCELERATE | |
| `stop_straight_1` | 2.3 | come to a stop | come to a stop, STRAIGHT | KEEP | 专家起步左转，cf 原地刹停。见下面"修掉的问题" |
| `stop_left_0` / `1` | 5.6 / 5.0 | come to a stop | come to a stop and bear left | ACCELERATE / KEEP | 左转途中刹停，停在路口里 |
| `stop_right_0` / `1` | 5.6 / 6.4 | come to a stop | come to a stop and bear right | KEEP | 右转途中刹停 |
| `decelerate_straight_0` / `1` | 7.0 / 5.7 | slow down | slow down, STRAIGHT | KEEP | |
| `decelerate_left_0` / `1` | 3.7 / 8.5 | slow down | slow down and bear left | KEEP / ACCELERATE | 减速左转（专家是匀速 / 加速转过去）|
| `decelerate_right_0` / `1` | 5.9 / 4.2 | slow down | slow down and bear right | ACCELERATE | |
| `keep_straight_0` / `1` | 10.9 / 10.2 | （无）| hold my current speed, STRAIGHT | ACCELERATE / DECELERATE | `_1`：专家从 10 m/s 减到 6.5，cf 保持 10 m/s |
| `keep_left_0` / `1`、`keep_right_0` / `1` | 2.5–6.2 | （无）| hold my current speed and bear left/right | ACCELERATE | 专家转弯时在加速，cf 匀速 |
| `accelerate_straight_0` / `1` | 7.0 / 6.5 | speed up | speed up, STRAIGHT | KEEP | |
| `accelerate_left_0` / `1` | 6.1 / 3.6 | speed up | speed up and bear left | KEEP | |
| `accelerate_right_0` / `1` | 0.4 / 5.1 | speed up | speed up and bear right | KEEP | `_0`：路口从静止起步右转，专家起步慢（5 s 才到 2.5 m/s），cf 按 +1.0 加到 5 m/s |

方向词从 cf 轨迹算：suggestion 只给速度，过路口时答案是 `… and bear left/right: <PLAN>X,LEFT/RIGHT</PLAN>`。

## 换道

| 图 | v0 | prompt | answer | 看什么 |
|---|---|---|---|---|
| `lane_left_0` / `1` / `2` | 11.6 / 13.7 / 8.6 | change lanes to the left | slow down / hold my current speed / slow down and change lanes to the left | |
| `lane_right_0` / `1` / `2` | 1.5 / 10.5 / 5.3 | change lanes to the right | speed up / hold my current speed / speed up and change lanes to the right | `_0`：拉斯维加斯路口起步，穿过路口时往右并一个车道；`_1`、`_2`：新加坡弯道，两条同向车道之间是白虚线 |

换道的速度词从轨迹算，纵向照抄专家，所以 navtrain 里多是 DECELERATE。换道图的右栏画的是 cf 相对专家的横向偏移（左为正），不是车速；点线是 ±3.5 m（大约一个车道宽）。

🔴 用户看出来两张"不太像能往右换道"的，各对应一个漏掉的问题，D6 导出时都已去掉：
1. 第一版 `lane_right_0`（`f2f3fc476dc45ea5`）：cf 末端只离专家 0.12 m，因为专家自己就在往右换道（t=0 还在车道中间，车头已经朝右偏约 6°）。这种"专家本来就换进目标车道"的有 702 条（14%）。规则：末端和专家的横向差 < 半个车道宽就丢。
2. 第二版 `lane_right_0`（`f80f84c1127d5a59`）：确实往右偏了 3.6 m，但并进去时离右车道的车最近只有 1.0 m，之后一直贴着 1.5 m，PDM TTC = 0。合成时只要求 NC / DAC / DDC，没要求 TTC。规则：所有类别 TTC 都要满分，速度去掉 18,679 条、换道 802 条。

这一版的图都是两条规则之后挑的；被 TTC 去掉的 5 张（`stop_left_0`、`keep_straight_0`、`accelerate_left_0`、`lane_right_0`、`lane_right_1`）同组重挑了。

3. 第三版 `lane_right_0`（`c18eb96197aa5b1c`）用户问"是不是直接撞了"：没撞（见上面的 ⚠️，最近 2.2 m、专家 2.7 m），但顺着查出 TTC 管不到的一类：并道时贴着旁边或前面的车，或者第 5 s 才贴上（PDM 只查 4 s）。规则：5 s 内离最近的车 / 自行车 < 2.5 m、且比专家同一帧的最小车距还近就丢，换道去掉 514 条（`c18eb` 本身也在内，用户要求丢）。

[lc_tight/](lc_tight/) 是 4 个例子，文件名里写了新规则下的结果。第一版规则把行人和静止物体也算进去，用户看出 `tight_1`、`tight_3` "不用去掉"：离得近的是路边的杆子（0.09 m）和酒店门口的花盆，于是改成只算车和自行车，这两张现在保留（`_kept`）。`tight_0`（往左并道时贴上从后面 11 m/s 开上来的车，0.07 m）和 `tight_2`（斜着插进左车道排队的车之间，1.1 m，专家 4.0 m）仍然去掉（`_dropped`），看快照能看出来。

现在的 `lane_right_0`（`039064e3ab615ba4`）是第四版：路口起步往右并一个车道，5 s 内离车最近 5.6 m。注意它是一边穿过路口一边换道，原型阶段定的规则允许（导航直行、两条车道都直）。

## 出这组图时修掉的问题

`stop_straight_1` 当前车速 2.3 m/s，第一版答案写的是 "remain stopped"。原因是 STOP 的动词按轨迹前 0.5 s 的平均速度选（和 CoT 一样），−2.5 m/s² 的刹车把它拉到 1.7。现在改成按当前车速选：v0 ≥ 2 是 "come to a stop"，否则 "remain stopped"。影响 4,789 条（v0 = 2.0–2.6 m/s 的 STOP），已重新导出。
