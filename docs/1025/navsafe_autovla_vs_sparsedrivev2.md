# NavSafe-∞ 小规模评测：AutoVLA / SparseDriveV2 / 我们的 AutoVLA SFT-CoT

评测日期：2026-10-25。机器：ztao-SCU（2 × RTX A6000 48 GB）。代码：`/local1/mhu/NavSafe-infty`（commit `bd68458`，另加 `autovla_ours` adapter，见第 5 节）。

## 1. 结论

SparseDriveV2 在这 10 个场景上明显好于 AutoVLA：平均 DS 53.8 对 21.5，成功 4 个对 0 个。我们自己的 CoT-SFT checkpoint 介于两者之间：DS 37.3，成功 2 个。

| | 官方 AutoVLA | SparseDriveV2 | 我们的 AutoVLA SFT-CoT |
|---|---|---|---|
| checkpoint | `AutoVLA_PDMS_89.ckpt`（官方 GRPO-RFT，NAVSIM PDMS 89） | `sparsedrive_navsimv1_92p2.ckpt`（官方 NAVSIMv1，PDMS 92.2） | HF `PhoenixHu/AutoVLA-SFT-cot-103k`，`epoch=4-loss=0.4436.ckpt`（navtrain CoT SFT，ViT 冻结） |
| 平均 DS | 21.5 | **53.8** | 37.3 |
| SR | 0/10 | **4/10** | 2/10 |
| comfort | 0.70 | 0.94 | 0.81 |

- SparseDriveV2 的优势主要在于不开出可行驶区域、开得更稳：官方 AutoVLA 有 3 次 `off_drivable`，SparseDriveV2 0 次，我们的模型 1 次。
- 三个模型都处理不好静止障碍和突然出现的 VRU：两个 I-3 场景三个模型全部失败；R-4 `02015675e4585611` 三个模型都撞上动物；R-2 `00c1e4eb4a045f20` 三个模型都追尾同一辆车。这几类正好是我们想用"减速 / 停车 / 换道"指令提升的场景。
- 我们的模型**经常能看见危险，但决策没有跟上**（见第 4 节）：例如 R-3 里连续说 "I see pedestrians 14 meters ahead"，结论却是 "I should speed up"。
- 每类只有 2 个场景，结果只能作为定性参考。

## 2. 设置

| 项 | 内容 |
|---|---|
| 场景 | R-4、I-3、R-3、R-2、V-10 各 2 个，共 10 个（token 见第 3 节）。其中 9 个插入了 actor；V-10 `07846b829b3a575e` 的 recipe 实际上没有插入任何 actor（NavSafe README 写的是 1 个，与 recipe 不符），按原始日志评测，场景是新加坡 one-north 施工区封闭车道处的并道 |
| 协议 | `navsafe benchmark` 的完整协议：半反应式交通、LQR 跟踪、120 s 路线时间预算、seed 0、harmonizer 开启、按 recipe 插入 actor |
| 运行方式 | 两条 lane，每张 GPU 各跑一条，渲染器、仿真器和 policy 放在同一张卡上，显存峰值约 29 GB |
| 可视化 | 开启（`NAVSAFE_ENABLE_VIS=1`） |
| 我们模型的 prompt | 与训练一致：`prompt_spec.SYSTEM_COT` + `user_text(..., cot_mode="reason_then_act")`，video 像素 28·28·140（hparams 的 `model.video`），temperature 0.01 |

## 3. 逐场景结果

DS / 终止原因 / 帧数（10 Hz）。✓ 表示成功。

| 类型 | token | 官方 AutoVLA | SparseDriveV2 | 我们的 SFT-CoT |
|---|---|---|---|---|
| R-4 动物 | `005f87dd980253a5` | 16.9，路口左转开上人行道（f75） | **100 ✓**（f592） | 12.5，追尾日志车辆（f75） |
| R-4 动物 | `02015675e4585611` | 8.7，撞上动物（f38） | 7.8，撞上动物（f30） | 8.7，撞上动物（f31） |
| I-3 静止车辆堵路 | `03b66343e1ac5d68` | 7.3，撞上堵路车（f37） | 7.0，撞上堵路车（f48） | 7.0，撞上堵路车（f33） |
| I-3 静止车辆堵路 | `050284885b2059e9` | 21.2，开出可行驶区域（f169） | 37.5，追尾日志车辆（f426） | 22.4，开出可行驶区域（f262） |
| R-3 人群过马路 | `02379e524f105926` | 18.5，追尾日志车辆（f104） | 6.7，撞上行人（f51） | 6.7，撞上行人（f47） |
| R-3 人群过马路 | `0777f5a7263758be` | 15.4，撞上机器人（f98） | **100 ✓**（f313） | **100 ✓**（f265） |
| R-2 自行车 | `00c1e4eb4a045f20` | 15.8，追尾日志车辆（f92） | 14.1，追尾同一辆车（f103） | 15.8，追尾同一辆车（f91） |
| R-2 自行车 | `20cc0fdb7e2d5c3f` | 29.8，撞上骑车人（f258） | **100 ✓**（f403） | 29.8，撞上骑车人（f221） |
| V-10 并道（未插入 actor） | `07846b829b3a575e` | 44.1，追尾日志车辆（f298） | **100 ✓**（f406） | **100 ✓**（f408） |
| V-10 并道抢行 | `13805e7752fe5685` | 37.2，开出可行驶区域（f153） | 64.9，被并道车追尾，不归责于自车（f301） | 69.7，被并道车追尾，不归责于自车（f296） |

终止原因统计：

| | 撞上插入的 actor | 追尾日志车辆 | 开出可行驶区域 | 被撞（不归责） | 到达终点 |
|---|---|---|---|---|---|
| 官方 AutoVLA | 4（动物、堵路车、机器人、骑车人） | 3 | 3 | 0 | 0 |
| SparseDriveV2 | 3（动物、堵路车、行人） | 2 | 0 | 1 | 4 |
| 我们的 SFT-CoT | 4（动物、堵路车、行人、骑车人） | 2 | 1 | 1 | 2 |

## 4. 我们模型的 reasoning（闭环中每 0.5 s 一次）

reasoning 记录在 `scenarios/<token>/plan_records.json` 的 `predictions[*].reasoning` 里。几个失败场景里，撞上之前它说了什么：

| 场景 | reasoning 摘录 | 问题 |
|---|---|---|
| I-3 `03b66343e1ac5d68` | f13 "vehicle 20 m ahead … moving at 7 m/s … slow down" → f18 "… come to a stop" → f23–33 "vehicle 13 → 9 m ahead … speed up and bear right" | 看到了前车，但把**静止的堵路车判断成在行驶**（4–7 m/s）；先决定停车，随后改为加速并向右绕行，绕行太晚，撞上 |
| R-4 `02015675e4585611` | 全程 "hold my current speed and bear right" | reasoning 里**从未提到动物**，属于感知层面的遗漏 |
| R-3 `02379e524f105926` | f13–43 "I see pedestrians 15 → 9 meters ahead of me. I should speed up" | **看到了行人，结论却是加速**，reasoning 与决策不一致 |
| R-2 `20cc0fdb7e2d5c3f` | 大部分时间 "I see a cyclist 14 m ahead … hold / speed up"，偶尔 "stop"；f218 "cyclist 10 m ahead … vehicle 5 m ahead … speed up" | 看到了骑车人，但在停车和加速之间来回切换，最后在加速时撞上 |

这说明在这些长尾场景上，主要问题不只是"看不见"，而是"看见了却不减速"。显式地给出减速、停车或换道指令，有可能直接补上这一环，这支持我们方法的动机。

## 5. 注意事项

- **`autovla_ours` adapter**：NavSafe 自带的 `autovla` server 写死了上游原版的 CoT prompt，和我们训练用的 `prompt_spec.py` 不同。所以新增了 `navsafe/policy/sensor/autovla_ours.py` 和 `navsafe/modelzoo/navsim/vla_server/autovla_ours_server.py`，并在 `navsafe/engine/registry.py` 里注册。相机、帧历史、command 文字和轨迹解码都沿用官方 adapter，只是 prompt 改为直接 import `doc_vla_search/dataset_utils/prompt_spec.py`。checkpoint 加载结果为 825 个 key，0 missing，两边的动作 codebook md5 相同。
- **像素设置**：`sft_dataset.py` 训练时用的是 28·28·128，而 hparams 和 `models/autovla.py::get_prompt` 推理时用的是 28·28·140，这里按推理配置用了 140。可通过 `NAVSAFE_AUTOVLA_OURS_PIXELS=100352` 改成 128。
- **导航 command**：NavSafe 把 LANEFOLLOW 映射为 `unknown`，而我们的训练数据里没有这个值。三个模型用的是同一套映射（feasibility 文档第 3 节的待定项）。
- **驱动版本**：本机驱动 570.124.06，低于 Isaac Sim 6.0 要求的 580.95.05，日志里有 `rtx driver verification failed`。相机画面由 NuRec 渲染，目前看不影响评测，但和官方环境不同。
- **可复现性**：官方 AutoVLA 在 R-4 `005f87dd980253a5` 上跑了两次，DS 分别是 18.1 和 16.9，终止原因和帧数几乎相同。seed 0 下仍有少量不确定性。

## 6. 文件

| 内容 | 路径 |
|---|---|
| 运行结果 | `/local1/mhu/navsafe_runs/{autovla,sparsedrivev2,autovla_sft_cot_103k}_5types_x2/` |
| 汇总报告 | 上面各目录下的 `report.md` |
| 每个场景的动图 | `scenarios/<token>/visualization/combined.gif` |
| 前视逐帧图 | `scenarios/<token>/cam/cam_f0_<帧号>.jpg`（带叠加层：红点是日志参考路线，青→黄点是模型规划，模型本身看不到这些标记） |
| 每次规划的 reasoning | `scenarios/<token>/plan_records.json`（`predictions[*].reasoning`） |
| token 列表 | `/local1/mhu/navsafe_runs/tokens_5types_x2.txt` |
| 运行脚本 | `/local1/mhu/navsafe_runs/run_policy.sh <model-type> <ckpt> <tokens> <run名>`（我们的模型：`autovla_ours`） |
