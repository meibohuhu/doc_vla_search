# AutoVLA on NAVSIM v2（navhard 两阶段，EPDMS）

leaderboard：<https://huggingface.co/spaces/AGC2025/e2e-driving-navhard>（NAVSIM 官方 README 里的 "Public Leaderboard v2 (navhard)"）。
官方代码：<https://github.com/autonomousvision/navsim> 的 main 分支（v2.2），提交说明在它的 `docs/submission.md`。

仓库里的 `navsim/` 是 v1.1，只能跑 navtest / PDMS。这个目录下的脚本另外用一份 v2 devkit，两套互不影响。

**在论文里的定位（2026-11-06 定）**：只放 supplementary，和 Bench2Drive 一样，用来说明加了我们的方法以后，正常驾驶能力没有下降。navhard 不是长尾 benchmark：它是从 nuPlan 里挑出的难场景（无保护转弯、密集车流），再加上第二阶段"开偏后能否纠正"的测试，场景里几乎不会触发 VLM 的危险判断。所以报的是同一个底座 ckpt 加和不加我们的方法的对比，不拿来排名。

## 文件

| 文件 | 做什么 | 在哪个 navsim 下跑 |
|---|---|---|
| `env.sh` | 路径、ckpt、环境变量；所有脚本都 source 它，每项都能用环境变量覆盖 | — |
| `setup_data.sh` | clone v2 devkit（固定 commit `0a380a9`），下载 split，接上本地 test logs / 图 | — |
| `build_inputs.py` | v2 SceneLoader → 每个 token 一行 AutoVLA 输入（`inputs.jsonl`），只取图片路径 | v2 |
| `infer_autovla.py` | AutoVLA 推理，单卡一片、可续跑 → `preds_s*.jsonl`（轨迹 + 生成文本） | 仓库版（v1.1） |
| `make_submission.py` | preds → 官方格式 `submission.pkl` | v2 |
| `score_local.sh` | 官方 `run_metric_caching.py` + `run_pdm_score_from_submission.py` 本地算 EPDMS | v2 |
| `run.sh` | 上面四步串起来 | — |

推理不在 v2 下跑，是因为 `models/` 依赖装在环境里的仓库版 navsim；读场景和打分只用 v2 的官方代码。两边只通过 jsonl 交接。

## 用法

```bash
bash tools/navsim_v2/setup_data.sh warmup                         # 1.2 GB
GPUS="0 2" bash tools/navsim_v2/run.sh                            # warmup：建输入 → 推理 → submission.pkl → EPDMS

bash tools/navsim_v2/setup_data.sh navhard                        # 约 32 GB
SPLIT=navhard_two_stage GPUS="0 2" bash tools/navsim_v2/run.sh

# 换 ckpt：
MODEL_DIR=... CKPT=... CONFIG=.../hparams.yaml TAG=xxx GPUS="0 2" bash tools/navsim_v2/run.sh
# 只给已有的 submission.pkl 打分：
bash tools/navsim_v2/score_local.sh warmup_two_stage <submission.pkl>
```

默认 ckpt：`model_zoo/autovla_l0r0_cot_turn40_freeze/epoch=4-val_action_loss=2.1827.ckpt`。config 用它自带的 `lightning_logs/version_0/hparams.yaml`（L0R0 侧视、use_cot）。

目录（`V2_ROOT=/scratch/mh2803/vla/navsim_v2`）：

```
navsim-devkit/                 官方 v2 代码
data/                          = OPENSCENE_DATA_ROOT
  navsim_logs -> /scratch/mh2803/vla/nuplan/navsim_logs     第一阶段场景的 logs（test）
  sensor_blobs -> /scratch/mh2803/vla/nuplan/sensor_blobs   第一阶段的图（test）
  warmup_two_stage/            synthetic_scene_pickles/ + sensor_blobs/（合成场景的图）
exp/
  metric_cache/<split>/
  <split>/inputs/              inputs.jsonl、inputs_summary.json
  <split>/<TAG>/               preds/、submission.pkl、logs/
  score_<split>_<TAG>/<时间>/   打分 csv
```

## 输入怎么构造

和 navtest 评测（`run_pdm_score_cot.py` 读的 JSON，来源是 `navsim/navsim/agents/vla_agent.py`）一致，只是直接从 v2 的 `AgentInput` 取：

- 相机：前视 F0 + 侧视（按 config 的 `nuplan_side_cams`，这个 ckpt 是 L0 / R0），各 4 帧，旧 → 新。
- 速度 / 加速度：当前帧 `ego_velocity` / `ego_acceleration` 的二维向量，prompt 里取模。
- 导航指令：`driving_command` one-hot → turn left / keep forward / turn right / unknown。
- prompt：默认 `reason_then_act`，和 navtest 一样。

输出取 AutoVLA 10 个点（0.5 s 间隔）里的前 8 个，也就是官方默认的 4 s。打分只看 4 s 以内，所以和 navtest 时给 10 个点等价。

## 已经核实的

- **合成场景（第二阶段）**：每个是一个 `Scene` pickle，4 帧历史。8 路相机每帧都有单独渲染的图（3DGS，沿扰动后的轨迹）。每帧都有 `ego_status`，包括 `driving_command`。所以输入的字段和真实场景完全一样。
- **warmup 第一阶段的图**：warmup 包只带合成场景的图，第一阶段的图要去 OpenScene test 的 `sensor_blobs` 找。本地的 `sensor_blobs/test` 只留了 navtest 用到的帧和 F0 / L0 / R0 / L1 / R1。有 5 个 token 缺最早一帧（-1.5 s），`build_inputs.py` 用同一路相机最近的一帧补上，记在 `inputs_summary.json` 里。所以 **warmup 的本地分和 HF warmup 不会完全一样**；它只用来验证流程。
- **navhard**：下载包里带第一阶段的图（`curr_sensors` / `hist_sensors`）。`build_inputs.py` 第一阶段先找 `sensor_blobs/test`，找不到再去 split 自己的 `sensor_blobs` 找。下载后看 `inputs_summary.json` 的补图数是不是 0。
- **本地打分流程**：用匀速直行的 submission（`make_submission.py` 不给 preds，就全部用匀速直行）验证过，220 个 token 都有效。

## 结果

| split | 模型 | 第一阶段 | 第二阶段 | **EPDMS** |
|---|---|---|---|---|
| warmup（220） | 人类轨迹（只有第一阶段；第二阶段没有真值，用匀速直行） | 95.51 | 29.67 | 28.40 |
| warmup（220） | ⚠️ AutoVLA `l0r0_cot_sft_navtrain_navtest_vit_unfreeze` ep4（**训过 navtest**，见下） | 88.78 | 39.85 | 32.51 |
| warmup（220） | ⚠️ AutoVLA `l0r0_cot_sft_navtrain_navtest_vit_freeze` ep4（**训过 navtest**） | 66.17 | 36.49 | 21.94 |
| warmup（220） | AutoVLA `l0r0_cot_sft_navtrain_vit_unfreeze` ep4（没训过 navtest；HF 仓库里没有 hparams，用 `config/training/1028_tiger/` 的同名 config） | 50.88 | 41.65 | 17.98 |
| warmup（220） | LTF（官方基线 `ltf_seed_0`，`run_ltf.sh`） | 65.91 | 34.28 | 27.37 |
| warmup（220） | AutoVLA 官方 RFT `AutoVLA_PDMS_89`（L1R1） | 65.94 | 39.17 | 23.45 |
| warmup（220） | 匀速直行 | 46.03 | 33.41 | 18.54 |
| warmup（220） | AutoVLA `l0r0_cot_turn40_freeze` ep4 | 45.01 | 36.82 | 15.75 |

warmup 只有 16 个第一阶段场景，一个场景的乘数项不合格，第一阶段就掉约 6 分，所以这些数只用来验证流程。

**navtest 泄漏**：navhard 和 warmup 都是从 OpenScene test 里挑的，和 navtest 大量重叠。navhard 第一阶段 450 个 token 里有 367 个（82%）在 navtest 里，76 个 log 全部和 navtest 共用；warmup 是 16 个里有 11 个。
训过 navtest 的 ckpt（HF `hellooPheonix/autovla-l0r0-cot-sft-navtrain-navtest-vit-unfreeze-adamw8bit-tigris8n`，本地 `/scratch/mh2803/vla/checkpoints/` 下）在 navhard 上的分数不能当成测试结果；NAVSIM 的规则也不允许这样做。
warmup 第一阶段按是否在 navtest 里拆开（每个场景第一阶段分的平均）：

| 模型 | 在 navtest 里（11 个） | 不在（5 个） |
|---|---|---|
| L0R0 ViT 解冻，训过 navtest | 0.949 | 0.752 |
| L0R0 ViT 解冻，没训过 navtest（同配方） | 0.516 | 0.493 |
| L0R0 ViT 冻结，训过 navtest | 0.700 | 0.578 |
| AutoVLA_PDMS_89 | 0.777 | 0.400 |
| turn40_freeze | 0.435 | 0.484 |
| LTF | 0.607 | 0.775 |
| 匀速直行 | 0.408 | 0.574 |

不在 navtest 里的 5 个 token，正好就是最早一帧缺图、被补上的那 5 个（本地 `sensor_blobs/test` 只保留了 navtest 的帧），所以这一组同时受两个因素影响。

同配方、只差有没有训过 navtest 的两个 ckpt（ViT 解冻）对比：EPDMS 32.51 vs 17.98，第一阶段 88.78 vs 50.88。
在 navtest 里的场景从 0.516 涨到 0.949；不在 navtest 里的场景也从 0.493 涨到 0.752。这 5 个不在 navtest 里的 token 和 navtest 来自同一批 log，是同一段路前后相邻的帧，所以也算见过。
结论：训过 navtest 的 ckpt 在 warmup / navhard 第一阶段的高分，基本来自泄漏。
AutoVLA 第一阶段 DAC 只有 8/16 合格，人类轨迹是 16/16。模型的轨迹离人类轨迹不远（4 s 处横向差 0.3–1.4 m），主要问题是左弯时多打了方向，比如 `ed3db88f` 横向 3.15 m（人类 1.76 m）、`0dc54a8c` 2.92 m（人类 2.24 m），蹭到了路边。
第二阶段 AutoVLA 比匀速直行高（36.8 vs 33.4），低分项是 NC / TTC（0.72）和 LK（0.39）。

EPDMS 取 csv 里 `extended_pdm_score_combined` 那一行。官方打分脚本最后打印的 "Final extended pdm score" 就是它；服务器用的是同一个脚本，所以 leaderboard 主列应该就是这个数（space 页面读不到，没有直接对过）。

combined 的算法（`run_pdm_score_from_submission.py` 里的 `calculate_individual_mapping_scores`）：
1. 第一阶段场景成对出现：当前帧和前一帧（`reactive_all_mapping` 里的 `orig_token, prev_token`），各自对应一组第二阶段合成场景。
2. 每组分数 = 第一阶段分 × 第二阶段分的加权平均。权重按合成场景的起点离模型第一阶段 4 s 终点的距离算（高斯核）。
3. 两组取平均，再对所有场景取平均。

所以 combined 不等于 `stage_one × stage_two`，后两行只用来分析。

## 提交到 leaderboard

1. 带上队伍信息重新生成 pkl：`TEAM_NAME=... AUTHORS=... EMAIL=... INSTITUTION=... COUNTRY=... SPLIT=navhard_two_stage bash tools/navsim_v2/run.sh`。推理已经跑完的话会直接跳过。
2. 在 HF 上 New Model（可以设成 private），上传 `submission.pkl`，文件名不能改。
3. 打开 space，点左下角 Login with Hugging Face，再点 New Submission，粘贴 `用户名/模型名`。结果在 My Submissions 里看。

规则：`test` / `navtest` / `navhard_two_stage` / `warmup_two_stage` 都不能用来训练。这个 ckpt 训练用的是 navtrain + trainval 166k 的 navfilt_turn40，不含 test。

## 用时

warmup 共 220 个 token，GPU0、GPU2 两张卡同时跑（卡上还有别人的进程），每个 token 约 12–16 s。
navhard 是 450 + 5462 = 5912 个 token，按这个速度两张卡大约 10–13 小时。metric cache 在 warmup 上不到 1 分钟。
