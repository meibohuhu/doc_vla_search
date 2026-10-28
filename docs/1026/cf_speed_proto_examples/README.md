# D2 速度反事实：原型示例

2026-10-27。代码：[tools/cf_data/speed.py](../../../tools/cf_data/speed.py)。计划：[cf_sft_autovla_plan.md](../cf_sft_autovla_plan.md) §3 D2。

navtrain 随机 300 帧（seed 0，和换道原型同一批），每帧四类各一条，共 1200 个候选。

## 全量结果

输出在 `/scratch/mh2803/vla/nuplan/cf_navtrain_v1/speed/{navtrain,navtrain_val}/`：`ok.jsonl`（通过的，带轨迹、token id、PDM 子分、`plan`、`start_from_stop`）和 `candidates.jsonl`（全部候选，失败的只留原因）。68 进程，navtrain 10 分钟，val 26 秒。

| | 帧数 | 通过 | STOP | DECELERATE | KEEP | ACCELERATE（其中从静止起步）|
|---|---|---|---|---|---|---|
| navtrain | 101,288 | **224,897** | 89,825 | 53,053 | 49,083 | 32,936（7,443）|
| navtrain_val | 2,000 | **4,444** | 1,789 | 1,066 | 948 | 641（152）|

- 每帧通过率和原型 300 帧一致（STOP 89%、DECELERATE 52%、KEEP 48%、ACCELERATE 33%）。
- 和专家速度词不同的（真正的反事实）：STOP 70,420、DECELERATE 36,996、KEEP 21,519、ACCELERATE 11,290。
- 通过的 v0（min / 中位 / max，m/s）：STOP 0 / 4.2 / 10.1，DECELERATE 3.6 / 6.5 / 17.1，KEEP 2.0 / 5.7 / 17.1，ACCELERATE 0 / 1.9 / 14.2。
- 方向词：STRAIGHT 占 81%，其余是路口转弯（LEFT / RIGHT）。
- 专家自己被判闯红灯 940 帧（0.93%）。红灯拒掉 3,382 条。
- token 往返误差 p50 0.04 / p90 0.08 / max 1.45 m。> 0.5 m 的 222 条（0.1%），> 1.0 m 的 15 条，多是快速弯道里横向加速度接近 3.0 的 ACCELERATE，解码后的标签仍对得上。D6 导出时去掉 > 1.0 m 的。
- 第一次全量发现一个 bug，修完重跑（上面是重跑的数）：专家停着时 GT 会前后抖几厘米，合成的停车轨迹沿倒退的那一段取点，朝向翻 180°，解码后侧移 1.5 m。现在建路径前只留沿专家朝向往前走 > 5 cm 的点（`forward_only`）。修完 STOP 多通过 684 条（原来解码后标签对不上）。

## 怎么合成的

路径用专家的（去掉专家停着时前后抖动的点），不够长时沿 PDM 的路线中心线续；只换快慢，恒定加速度：

| 类别 | 加速度（m/s²），按顺序试 | 说明 |
|---|---|---|
| STOP | −2.5 | 减速侧 p99.9（9 月在 navtest 上重标的）|
| DECELERATE | −1.0，不行再试 −0.5 | v0 < 5 m/s 时 −1.0 在 3 s 后已低于 2 m/s，会被标成 STOP |
| KEEP | 0 | 不当 suggestion，只出现在答案里 |
| ACCELERATE | +1.0，加到限速为止 | navtest \|a\| p90 1.29；原来的 +1.5 持续 5 s 太猛 |

**标签用 `plan_from_track` 在合成轨迹上重新算**，取第一档标签对得上的；都对不上就丢掉（`label_mismatch`）。所以每条通过的样本，答案里的速度词就是它的类别。方向词也从合成轨迹算（路口转弯时是 LEFT / RIGHT）。

然后逐关检查：

| 检查 | 原因 |
|---|---|
| 横向加速度 ≤ 3.0 m/s²（`too_fast_for_curve`）| 专家减速转弯的帧，KEEP / ACCELERATE 会开过专家路径终点、沿路线中心线高速拐进路口（`e_curve`：11.2 m/s 拐 90°）。PDM 不拦（路径在车道里），tokenizer 也表示不了（原型里出现过 8 m 的往返误差）。阈值取专家 GT p99 2.56 和 p99.9 3.31 之间 |
| PDM NC / DAC / DDC 全为 1 | 和换道一样 |
| 红灯（`red_light`）| 见下 |
| 限速（`at_speed_limit`）| ACCELERATE 加到地图限速就不再加；v0 已经在限速附近、加不上去的丢掉 |
| 解码后的标签对得上（`token_label`）| 编码成 10 个 action token 再解码，训练学的是解码后的轨迹 |

### 红灯：metric cache 里没有

🔴 navsim v1 建 metric cache 时没把红绿灯传进 PDMObservation（`metric_cache_processor.py` 只调了 `update_detections_tracks`），300 帧里一个红灯都没有，**PDM 打分从来不管红灯**。所以从 `navsim_logs` 取每帧（2 Hz）的 `traffic_lights`，只看路线上的 lane connector（和 PDM 原本的设计一样，否则横穿路口时会碰到横向车流的红灯 connector），往红灯 connector 里开得比 t = 0 时更深就算闯。t = 0 已经过了停止线的不算。

- 300 帧里有 131 帧 t = 0 时就有红灯。
- 专家自己被判闯的有 3 帧（1%）：一帧是红灯右转（美国合法），另两帧没细看，可能是灯在 0.5 s 采样间隔里变了色。检查偏保守，可以接受。

### 限速：只看 lane

地图限速覆盖：拉斯维加斯、匹兹堡全有，波士顿 7%，新加坡没有。缺的按城市默认：波士顿 25 mph（11.2 m/s），新加坡 50 km/h（13.9 m/s）。

只取 lane 上的限速，不取路口里的 lane connector（那上面多是转弯限速，转弯该多快由横向加速度管）。拉斯维加斯 lane 上本身就有不少 4.5 / 2.2 m/s（10 / 5 mph）的限速，多在酒店接送区，专家在 196 帧里有 53 帧比它快 1 m/s 以上。按地图办，所以这些区域的 ACCELERATE 少。

## 统计（1200 个候选）

| 类别 | 通过 | 主要失败原因 | 通过的 v0（min / 中位 / max，m/s）| 和专家的速度词不同 |
|---|---|---|---|---|
| STOP | **260** | label_mismatch 22（v0 > ~9.5，3 s 内停不下来）| 0 / 4.1 / 10.1 | 200 |
| DECELERATE | **153** | label_mismatch 125（v0 < ~3.5，减速就是停车）| 3.7 / 6.5 / 14.1 | 112 |
| KEEP | **148** | label_mismatch 72（v0 < 2，保持就是停着）、pdm_nc 34、too_fast_for_curve 24 | 2.0 / 5.6 / 14.1 | 66 |
| ACCELERATE | **94** | at_speed_limit 84、pdm_nc 54（追尾前车）、too_fast_for_curve 47 | 0 / 1.8 / 13.9 | 34 |

- 合计 655 条，按比例全量 navtrain 约 22 万条（STOP ~8.8 万、DECELERATE ~5.2 万、KEEP ~5.0 万、ACCELERATE ~3.2 万），远多于换道（~5000）。训练时按类别配比抽，不用全用。
- 红灯拒掉 16 条（STOP 1、DECELERATE 3、KEEP 6、ACCELERATE 6）。
- 从静止起步（v0 < 0.5 且 ACCELERATE 通过）22 条，全量约 7000 条。
- token 往返误差 p50 0.04 / p90 0.09 / max 0.38 m。
- "和专家的速度词不同"才是真正的反事实。ACCELERATE 里 94 条只有 34 条和专家不同，因为能加速的帧多半专家自己也在加速。

## 示例图

每张图：左边前视图；中间 BEV（紫色是日志车辆，实色 t = 0，淡色 t = 2.5 s；红色是 5 s 内亮过红灯的路线 connector；实心点是通过的，空心点是没通过的）；右边是每 0.5 s 的平均车速（灰色是专家，实线是通过的，虚线是没通过的，黑色虚线是限速）。标题第二行是每类的结果。

| 图 | v0 | 看什么 |
|---|---|---|
| `a_all_ok` | 7.4 | 四类都通过 |
| `b_high_speed` | 13.9 | 高速：STOP 3 s 内停不下来，标签对不上；ACCELERATE 加到限速 15.6 为止 |
| `c_red_light` | 7.4 | KEEP 4.8 s 冲进红灯路口；ACCELERATE 追尾 |
| `d_red_light` | 8.0 | 前方 2 s 后变红灯。专家在停止线前停住；只有 STOP 通过，DECELERATE / KEEP / ACCELERATE 都会冲过去 |
| `e_curve` | 11.2 | 专家减速右转。DECELERATE（−1.0）、KEEP 拐弯太快；ACCELERATE 转进去的车道限速 4.5 |
| `f_start_from_stop` | 0.4 | 从静止起步：STOP（remain stopped）和 ACCELERATE 通过 |
| `g_accel_collision` | 5.0 | 前车在减速，ACCELERATE 追尾；STOP / DECELERATE / KEEP 都通过 |
| `h_low_speed` | 0.6 | 低速只剩 STOP 和 ACCELERATE 两种 |

## 已知的局限

- 其他车按日志回放，不会对自车做出反应。STOP / DECELERATE 时后车照样按日志开过来，PDM 不算自车的责任，这里也不查。所以"有车紧跟还急刹"的 STOP 会通过。
- 恒定加速度，没有加加速度限制，速度曲线是折线。
- 专家路径和路线中心线续接处有一点横向偏差（通过的样本里最大 0.86 m）。偏差大的都被横向加速度挡掉了。
- 专家几乎不动时，路径从原点斜着连到中心线，停车距离 < 0.5 m 的 STOP 朝向可能偏 0.15–0.4 rad（全量 18,758 条里 19 条）。
- 高速（v0 > ~10 m/s）没有 STOP 样本。可以给 STOP 加一档 −3.5，代价是比专家日志里的减速猛；10-27 定先不加。
