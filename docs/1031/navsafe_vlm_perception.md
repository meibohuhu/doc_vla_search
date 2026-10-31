# NavSafe VLM 感知模块：Qwen3-VL prompt 调试结果（2026-10-30 ~ 31）

目标：参考 simlingo agent（`doc_drive_search/team_code/agent_simlingo_cluster_target_point_cot_egohistory_f2d.py`）里 VLM 的用法，给 NavSafe 做一个判断前方危险的 VLM 模块。要处理四种场景：

- 人（R-1、R-3）
- 自行车（R-2）
- 动物（R-4）
- 前方停着的车堵路（I-3）

判断结果以后会变成 cf SFT 模型的 scene report 和 suggestion。

## 1. 最终方案：两步调用

| 步骤 | 模型 | prompt | 输入图 | 怎么用回答 |
|---|---|---|---|---|
| 1. 人 / 自行车 / 动物 | **Qwen3-VL-8B** | `tools/navsafe_vlm/prompts/v16_vru.txt` | 三张：前（CAM_F0）、左前（CAM_L0）、右前（CAM_R0），按这个顺序 | `PEDESTRIAN` / `CYCLIST` / `ANIMAL` 直接用；`NONE` 进入第 2 步 |
| 2. BLOCKED | **Qwen3-VL-32B** | `tools/navsafe_vlm/prompts/v7.txt` | 只给前视图 | 只看是不是 `BLOCKED`，其他回答一律当 `NONE` |

```
label = ask(8B, v16_vru, [f0, l0, r0])
if label == NONE:
    label = BLOCKED if ask(32B, v7, [f0]) == BLOCKED else NONE
```

为什么这样分：

- **第 1 步用 8B**：8B 判人、自行车、动物 98.3%，没有一个误报，每帧 0.45 s，大约是 32B 的 1/3。
- **第 2 步要用 32B**：8B 判不出 BLOCKED。在旧的单图数据上，8B 用什么 prompt 都是 0/13。
- **第 2 步的 prompt 要用五类的 v7，不用只判 BLOCKED 的 prompt**：
  - 只给 BLOCKED / NONE 两个选项时，模型会把一切挡在车道上的东西都答成 BLOCKED，包括过马路的人、骑车人和送餐机器人（误报 30 多帧）。
  - v7 里人、自行车、动物有单独的选项，能把这些情况"吸走"，BLOCKED 才不乱报。
- **第 2 步只给前视图**：三张图能多抓几帧 BLOCKED（23/26 对 19/26），但误报也多（9 对 5）。BLOCKED 误报会让车无故停下或绕行，所以选前视图。
- **备选**：第 1 步也可以用 32B + `v13_vru.txt`，是 98.5%。`v16` 只在 8B 上测过。

以下两组作为参考配置保留：

- **v7 三张图**：`v7.txt`，图按前、左前、右前的顺序给，一次判五类。
- **v10 + v7 前视图**：先用 `v10_3cam.txt` 给三张图判人、自行车、动物，答 NONE 时再用 `v7.txt` 只给前视图判 BLOCKED。

## 2. 调用设置

这些设置要和测试时完全一样，换机器时照抄：

- **模型**：
  - `Qwen/Qwen3-VL-8B-Instruct`
  - `Qwen/Qwen3-VL-32B-Instruct`：这台机器在 `/scratch/mh2803/vla/checkpoints/Qwen3-VL-32B-Instruct`
  - 两个都用 bf16、`attn_implementation="sdpa"`、`AutoModelForImageTextToText`。
  - 放在同一张 97 GB 的卡上大约占 80 GB。
- **环境**：torch 2.11、transformers 5.19。这台机器的环境是 `/scratch/mh2803/vla/simlingo/conda_envs/qwen3vl`。
- **对话格式**：一轮 user 消息，先放所有图，后放 prompt 文字。
  - `apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)`
  - 贪心解码（`do_sample=False`），`max_new_tokens=8`。
- **`{speed}`**：prompt 里的这个占位符要替换成自车速度（m/s），保留一位小数，例如 `4.2`。
- **图片**：
  - 渲染器原图是 1920×1080，按原宽高比缩到宽 1280。
  - 前视图必须**不带 NavSafe 可视化叠加**（走廊、轨迹点、文字）。在 agent 里直接用渲染器给的原图就行。
  - 这次测试用的 run 里 `frames/<f>/cam_f0.jpg` 带着叠加，是用 `tools/navsafe_vlm/clean_overlay.py` 去掉的。
- **解析**：在回答里找 `PEDESTRIAN|CYCLIST|ANIMAL|BLOCKED|NONE`，取最先出现的那个。见 `probe.py` 的 `parse()`。
- **耗时**：在 Blackwell 97 GB 上，单张卡、一次处理一帧。
  - 8B 三张图：0.45 s/帧。
  - 32B 前视图：0.62 s/帧。
  - 32B 三张图：1.3 s/帧。
  - 大多数帧第 1 步就会答 NONE，所以平均每帧约 1.1 s。放进闭环里要降频调用。

## 3. Prompt 全文

### 3.1 `v16_vru.txt`（第 1 步，8B）

```
You get three photos taken at the same moment by cameras on a car. The first photo is the main view: it looks straight ahead through the windshield. The second looks to the front-left and the third to the front-right; they only show what is just outside the main view. The car with the cameras is not in the photos. The car is driving at {speed} m/s.
Look only at the car's path: the stretch of road the car will drive over in the next few seconds, starting right in front of the car. When the car is turning, its path bends toward the turn and may continue into the second or third photo. Ignore the sidewalk, cars parked along the curb, other lanes, the oncoming lane, and anything that stays there.
In the second and third photos, people standing or walking on the sidewalk, at a building entrance or at the corner are normal and do not count. They count only if they are already on the road surface and moving into the car's path.
Something very close to the car may show up only partly, at the bottom or side edge of a photo, and may look very large. It still counts.
Answer with exactly one token:
<PEDESTRIAN> if a person is in the car's path or is walking into it, for example crossing the road in front of the car. A small delivery robot moving with people counts as a person.
<CYCLIST> if a person riding a bicycle or scooter is ahead in the car's lane, riding along it or crossing it
<ANIMAL> if an animal is in the car's path or is moving into it, even if only part of it is visible, or if an animal is standing on the road or crosswalk where the car is about to turn
<NONE> if none of the above: the path is clear, or there are only vehicles, cars parked at the curb, people on the sidewalk, or traffic cones and road work along the side
If several apply, answer the one closest to the car. Output only the token.
```

`v13_vru.txt`（32B 用的备选）只在第 4 行少了 `, and may look very large`，其他完全一样。

### 3.2 `v7.txt`（第 2 步，32B，只给前视图）

```
This photo was taken by a camera on the windshield of a car, looking forward. The car with the camera is not in the photo. The car is driving at {speed} m/s.
Look only at the car's path: the stretch of road the car will drive over in the next few seconds, starting right in front of the camera. When the car is turning, its path bends toward the turn. Ignore the sidewalk, cars parked along the curb, other lanes, the oncoming lane, and anything that stays there.
Something very close to the car may show up only partly, at the bottom or side edge of the photo. It still counts.
Answer with exactly one token:
<PEDESTRIAN> if a person is in the car's path or is walking into it, for example crossing the road in front of the car. A small delivery robot moving with people counts as a person.
<CYCLIST> if a person riding a bicycle or scooter is ahead in the car's lane, riding along it or crossing it
<ANIMAL> if an animal is in the car's path or is moving into it, even if only part of it is visible
<BLOCKED> if vehicles are stuck in the road right ahead and block the car's lane, for example two cars stopped close together, side by side or one slightly behind the other, often standing at an angle like after a minor crash, with empty road beyond them. The car would have to stop behind them or drive around them. A single car driving ahead of the car in normal traffic does not count.
<NONE> if none of the above: the path is clear, or there are only vehicles driving normally or in other lanes, cars parked at the curb, or traffic cones and road work along the side
If several apply, answer the most important one: people, cyclists and animals come before blocked vehicles. Output only the token.
```

## 4. 测试数据

**图片来源**：在 NavSafe 那台机器上用 `--vis-cameras CAM_L0,CAM_R0` 重跑了 10 个场景，拷到 `/scratch/mh2803/vla/navsafe/navsafe_runs/` 下：

- `sparsedrivev2_5types_x2_3cam`（下文 SD）
- `autovla_sft_cot_103k_5types_x2_l0r0`（下文 AV）

| 场景 | token | 内容 |
|---|---|---|
| R-3 人 | `02379e524f105926`、`0777f5a7263758be` | 人群（3 人 + 2 个送餐机器人）过马路 |
| R-2 自行车 | `00c1e4eb4a045f20`、`20cc0fdb7e2d5c3f` | 骑车人在自车车道前方 |
| R-4 动物 | `005f87dd980253a5`（狗）、`02015675e4585611`（马） | 狗过人行横道；马站在自车右转要进的路上 |
| I-3 堵路 | `03b66343e1ac5d68` | 两辆车斜着停在一起，像刚发生小事故 |
| I-3 | `050284885b2059e9` | 两次重跑都开出了可行驶区域，只用前面一段当 NONE |
| 负样本 | `07846b829b3a575e`、`13805e7752fe5685` | V-10 并道：正常车流、施工锥桶、人行道上的行人 |

**标注**：`tools/navsafe_vlm/labels_v2.py`，人工看三张图标的，共 413 帧。

- 判断标准：危险出现在任何一张图里，并且在自车路径上或正要进入路径，就算。
- 人行道上的行人、路边停的车都算 NONE。
- 危险还远、刚开始进入路径、或者被挡住看不见的帧，两种答案都算对。

| 标注 | 帧数 |
|---|---|
| NONE | 170 |
| 自行车 | 63 |
| 动物 | 41 |
| 人 | 40 |
| BLOCKED | 26 |
| 两种答案都算对 | 73 |

和单前视图标注（`labels_v1.py`）的主要区别：

- 马在右前视图里一直看得到，所以整段都标 ANIMAL。
- 骑车人被货车完全挡住的帧（AV 20cc f160–192），和狗钻到车头底下的帧（AV 005f f50–52），改成两种答案都算对。

**SD 和 AV 的区别**：SD 重跑的轨迹和原来几乎一样，AV 每次跑会差 1–2 m。所以再跑新的 run 时，要先对一遍帧号和标注。

## 5. 结果

以下都是 413 帧、`labels_v2`。

### 5.1 第 1 步：人 / 自行车 / 动物（`--task vru`，BLOCKED 帧算 NONE）

| 模型 + prompt | 总体 | 人 /40 | 自行车 /63 | 动物 /41 | NONE 答对 /202 | 每帧 |
|---|---|---|---|---|---|---|
| 32B v7，只给前视图 | 95.6% | 40 | 62 | 29 | 199 | 0.64 s |
| 32B v7，三张图 | 96.6% | 40 | 62 | 41 | 192 | 1.26 s |
| 32B v10_3cam | 99.0% | 40 | 63 | 41 | 199 | 1.35 s |
| 32B v13_vru | 98.5% | 40 | 63 | 41 | 198 | 1.30 s |
| 32B v14_vru（v13 + 两句难例提示） | 98.1% | 40 | 63 | 41 | 196 | 1.29 s |
| 8B v13_vru | 97.8% | 40 | 61 | 34 | 202 | 0.43 s |
| **8B v16_vru** | **98.3%** | 40 | 61 | 36 | **202** | 0.45 s |
| 8B v15_vru（v16 + 转弯那句） | 96.1% | 37 | 50 | 41 | 202 | 0.44 s |
| 8B v17_vru（v13 + 转弯那句） | 92.7% | 37 | 38 | 39 | 202 | 0.43 s |

- **侧视图对动物至关重要**：只给前视图时，马一出画面就漏了。
- **8B 偏保守**：从不误报，但会漏掉离得特别近、占满侧视图的马（02015675 f26–30）。
- **转弯那句对 8B 有害**："When the car is turning toward the second or third photo, a person or animal standing close by in that photo is in the car's path." 加了它马全救回来了，但自行车大量漏判，8B 好像因此只盯着侧视图。
- **32B 剩下的错误都是误报**：右边那条路上骑电动小车的人（07846 f119）、几处站在路沿上的行人。

### 5.2 第 2 步：BLOCKED（只给前视图）和两步合起来

| 第 2 步用的 prompt | BLOCKED 抓到（只判 BLOCKED 时） | 误报 | 两步合起来的总体（第 1 步 = 32B v13_vru） |
|---|---|---|---|
| **v7（五类，只取 BLOCKED）** | 19/26 | 5 | **95.6%** |
| v7，三张图 | 23/26 | 9 | 95.6% |
| b1_blocked（只有 BLOCKED / NONE） | 18/26 | 30 | 93.2% |
| b2_blocked（b1 + 旁边车道的车不算） | 19/26 | 37 | 93.2% |

一次判五类的单个 prompt，最好的是 v7 三张图（前视图放第一张），91.0%。加了"人行道上的人不算"这类句子，误报会少，但 BLOCKED 每次都掉到 10–13/26。这就是要拆成两步的原因。

### 5.3 其他结论（来自旧的单前视图数据，`labels_v1`）

- **8B 判不出 BLOCKED**：试过 v1–v6 的各种说法，包括是 / 否问法和"前方有车"。
- **32B 判 BLOCKED 靠措辞**：v7 里"两辆车挨着停、斜着、像刚发生小事故、后面是空路"这种具体描述才有效。用 simlingo 原来的 taskA prompt 只有 64%。
- **这几种做法都不好**：
  - 先描述再判断（v8_describe）：81.6%，而且每帧 2.5 s。
  - 多给一帧历史帧：动物误报变多。
  - 裁出前方路面：动物漏判变多。

## 6. 还没解决的问题

- **BLOCKED 正样本太少**：只有 03b66343 一个场景、两次 run，共 26 帧，调措辞没意义。要多跑几个 I-3 场景，比如 `84acc78d…`、`b023b7bc…`，也要带 `--vis-cameras CAM_L0,CAM_R0`。
- **BLOCKED 还剩的错误**：
  - 漏判：03b66343 刚靠近那几帧（f12–24，两辆车大约在 40 m 外）。要定一下这几帧该不该标成"两种答案都算对"。
  - 误报：20cc 左边车道有一辆货车和前车挨在一起，自车从右边车道开过去，形状很像堵路。
- **前视图叠加**：这次重跑设了 `NAVSAFE_NO_OVERLAY=1`，但 `cam_f0` 还是带叠加，需要查一下 NavSafe 为什么没生效。agent 里用原图就不受影响。
- **v16 在 32B 上没测**。
- **测试数据测不出差别了**：人、自行车、动物这一块在这批数据上已经到顶。要继续调，得加更难的场景，比如夜间、遮挡、多个危险同时出现。

## 7. 文件

都在 `tools/navsafe_vlm/` 下，目前都没提交。

| 文件 | 作用 |
|---|---|
| `probe.py` | 主测试脚本：加载模型，按 prompt 逐帧提问，输出混淆矩阵和错帧列表 |
| `rescore.py` | 改了标注以后，用已有回答重新打分，不重跑模型 |
| `labels_v1.py` / `labels_v2.py` | 人工标注：单前视图的旧 run / 三张图的新 run |
| `clean_overlay.py` | 去掉前视图上的 NavSafe 可视化叠加 |
| `contact_sheet.py` | 把若干帧拼成缩略图用来标注；`--views l0,f0,r0` 时每帧一行三张图 |
| `prompts/*.txt` | 所有试过的 prompt：v1–v17、b1–b2、simlingo 原版 |

`probe.py` 的选项：

- `--model qwen3vl-8b|qwen3vl-32b`
- `--views f0,l0,r0`：给哪几张图，按什么顺序
- `--labels labels_v2`
- `--task all|vru|blocked`
- `--prepare`：只建图片缓存，要 cv2；qwen3vl 环境没有 cv2，得在 `autovla_codeclean` 环境里跑

图片缓存在 `/scratch/mh2803/vla/navsafe/vlm_probe/images`，每次的回答在 `/scratch/mh2803/vla/navsafe/vlm_probe/results/<model>/*.jsonl`。

复现第 1 步：

```bash
# 先建图片缓存（要 cv2）
/home/mh2803/miniconda3/envs/autovla_codeclean/bin/python tools/navsafe_vlm/probe.py \
    --prompts x --labels labels_v2 --views l0,f0,r0 --prepare
# 8B 判人 / 自行车 / 动物
/scratch/mh2803/vla/simlingo/conda_envs/qwen3vl/bin/python tools/navsafe_vlm/probe.py \
    --model qwen3vl-8b --labels labels_v2 --task vru --views f0,l0,r0 --device cuda:0 \
    --prompts tools/navsafe_vlm/prompts/v16_vru.txt
```
