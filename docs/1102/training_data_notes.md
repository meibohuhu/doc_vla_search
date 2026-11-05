# CoLT-Drive 训练数据调研笔记（2026-10-30）

## 1. 下载的数据只有评测集

- HF 数据集 `tangzx2024/CoLT-Drive` 已下载到 `data/`（4.8G），按代码 README 的约定放在仓库内，`.gitignore` 已排除 `data/`。
- 只有两个 split：`vfull`（自然背景车流）和 `vclean`（去掉背景车），各 1,768 个样本，和 manifest 一致。
- 两个 split 的目标物体、道路几何完全相同，只是背景不同，是同一批测试样本的两个版本，**不是 train/test 划分**。
- 每个样本：`front_camera.jpg`、`prompt.txt`、`gt.json`（可接受的动作对）、`meta.json`（障碍物类型、类别、位置）。
- **没有官方训练数据**。代码 README 原话："Trained weights are not distributed: they derive from a driving corpus that is not public." 作者在 HF 上也只有这一个数据集。
- 这 1,768 个样本不能拿来训练，否则评测结果没有可比性。

## 2. 仓库里没有转换代码

GitHub 上只有一个提交（`11bc12a`），本地已是最新。

- `colt_drive/`：`prompts.py`、`inference.py`、`judge.py`，都只用于评测。
- `kpa/`：SLERP 合并和 RegMoE 训练。`kpa/data.py` 只读一个现成的 JSONL（"Bring your own driving data"），不负责从原始驾驶数据生成。
- `kpa/configs/kpa_qwen3vl_2b.yaml` 里训练数据路径是占位符：`data.manifest: /path/to/train.jsonl`。
- SLERP 那一步要的 driving LoRA adapter（`--finetuned`）也没有提供。

## 3. 仓库定死的训练数据格式

JSONL，每行一帧：

```json
{"image": "images/000001.jpg",
 "prompt": "<driving context>\n\n<COMPLEX_PROMPT>",
 "target": "Longitudinal: slow down\nLateral: keep lane"}
```

- `image`：前视相机图，绝对路径或相对 `root`（默认是 manifest 所在目录）。
- `regime`：可选。不填的话由 `kpa/regimes.py::infer_regime` 从 `target` 推出来。

### 3.1 `prompt`

评测时模型的输入是 `prompt.txt + "\n\n" + COMPLEX_PROMPT`（`colt_drive/inference.py:145`，system prompt 是 `"You are a helpful autonomous driving assistant."`）。训练时 `prompt` 也应该这样拼，和推理一致。

`prompt.txt` 的格式：

```
## Ego Motion History (past 2 seconds)
  t=-2.0s: speed=4.94 m/s, accel=-0.15 m/s²
  t=-1.5s: speed=4.86 m/s, accel=-0.03 m/s²
  t=-1.0s: speed=4.84 m/s, accel=-0.07 m/s²
  t=-0.5s: speed=4.83 m/s, accel=-0.2 m/s²
  t=+0.0s: speed=4.71 m/s, accel=-0.25 m/s²

## Navigation
  Route: GO STRAIGHT
```

nuPlan/NAVSIM 的 ego status 和 driving command 可以直接生成。

### 3.2 `target`：不必用 benchmark 的 9 个词

benchmark 的动作空间：

| 轴 | 动作 |
|---|---|
| 纵向 | `keep_speed`, `slow_down`, `yield`, `creep`, `stop`, `speed_up` |
| 横向 | `keep_lane`, `nudge_left`, `nudge_right` |

但训练的 `target` 是自由文本，只需要带 `Longitudinal:` 和 `Lateral:` 两行。有两处可以说明这一点：

- `kpa/regimes.py` 用 `"turn"`、`"lane change"` 判定 MANEUVERER，而 benchmark 的动作空间里没有转弯和变道。说明作者的训练语料里有这两类，标签是从轨迹算的通用 meta-action，不是专门为 benchmark 标的。
- `colt_drive/judge.py` 用 LLM 加同义词表把自由文本映射到 canonical 动作：`turn left to avoid` / `lane change left` → `nudge_left`，`decelerate` → `slow_down`，`accelerate` → `speed_up`。模型输出通用说法也能对上分。

所以 nudge、yield、creep 这几类从轨迹上不好区分的动作，训练这一侧可以不专门标。

### 3.3 regime 推断规则（`kpa/regimes.py`）

先判 MANEUVERER，再判 REACTOR，其余是 CRUISER：

- MANEUVERER（1）：横向包含 `turn` 或 `lane change`
- REACTOR（2）：横向包含 `nudge`，或纵向包含 `stop / slow / decelerate / brake / reduce speed / yield / adjust / creep / crawl / inch forward`
- CRUISER（0）：其他情况

## 4. 可以用 nuPlan 训练

- 29 个场景里，`NP01`–`NP20` 的描述都是 Pittsburgh、Singapore、Las Vegas，也就是 nuPlan 的城市，应该是用 nuPlan 的帧当背景再插入障碍物。`R1`–`R14` 是通用道路类型（城市多车道、高速、匝道、施工区等），来源不明。
- 视觉域和 prompt 格式都能和 navtrain 对上。

注意事项：

1. **背景帧可能和训练集重合**：这 20 个 nuPlan 场景的背景可能来自 navtrain，但 `meta.json` 里没有 nuPlan 的 log 或 token，没法精确排除。benchmark 考的是插入的障碍物，泄漏有限；要严谨的话可以在论文里说明，或用图像相似度找出最像的训练帧剔除。
2. **图像尺寸**：benchmark 是 1376×768，nuPlan 前视原图是 1920×1080，宽高比基本一致（≈16:9）。训练时最好缩放到和评测相近的分辨率。
3. **标签分布很偏**：nuPlan 大多数帧是 `keep speed + keep lane`，benchmark 考的是遇到障碍物时的反应。要按标签重新采样，增加 stop、slow down 这类帧的比例，尤其是前方有行人、车辆或静态物体的帧。

## 5. 转换方案（待实现）

标签复用 `doc_vla_search` 里的 `plan_from_track`（`models/utils/rl_rewards.py:54`）。它和 CoT 训练标签用的是同一套规则：3s 窗口、2Hz，输出 (v0, SPEED, PATH)。

| `plan_from_track` | target 文本 |
|---|---|
| `STOP` / `DECELERATE` / `KEEP` / `ACCELERATE` | `stop` / `slow down` / `keep speed` / `speed up` |
| `STRAIGHT` / `LEFT` / `RIGHT` | `keep lane` / `turn left` / `turn right`（或按横向位移区分出 `change lane to left/right`） |

步骤：

1. 遍历 navtrain，每帧取前视图（缩放到约 1376×768）。
2. 用过去 2s 的 ego 速度、加速度和 driving command 按 3.1 的模板生成 `prompt`，后面拼上 `COMPLEX_PROMPT`。
3. 用 GT 未来轨迹经 `plan_from_track` 生成 `target`；`regime` 不填，交给 `infer_regime`。
4. 输出 JSONL。先抽几百帧看标签分布，再定重采样比例。

另一个参考：AutoVLA 预处理里的 `get_action_instruction`（`dataset_utils/preprocessing/nuplan_dataset.py:253`）可以输出 `turn left` / `change lane to left` / `a deceleration` 等更细的词，阈值和 `plan_from_track` 不同，两者不能混用。
