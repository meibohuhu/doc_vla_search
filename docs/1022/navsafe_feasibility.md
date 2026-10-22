# NavSafe-∞ 可行性调研：我们的方法能否在 NavSafe 上做 SFT 和评测

调研日期：2026-10-22。代码：`/home/mh2803/vla/NavSafe-infty`（commit `bd68458`）。OPD 暂不考虑。

## 1. 结论

可以，但要换 VLA，SFT 数据也要重新造：

- **VLA 用我们自己训练的 AutoVLA，不用 SimLingo。** NavSafe 是 nuPlan 真实日志的 3DGS 重建，相机是 NAVSIM 那一套。SimLingo 只在 CARLA 上训练过，画面和相机都对不上。
- **SFT 数据要在 navtrain 上重新造。** dreamer 依赖 CARLA 的特权信息，不能直接搬过来。可以改用现成的 `navtrain_metric_cache`，配合 PDM scorer 检查候选轨迹（见第 4 节）。
- **评测通过自己写一个 policy adapter 接入**，不需要改 benchmark。
- **目前卡在渲染器**：渲染器需要 docker，而本机账户没有权限安装（见第 6、7 节）。

## 2. NavSafe 概要

| 项 | 内容 |
|---|---|
| 场景 | 28 类事件，每类 10 个，共 280 个，全部来自 nuPlan **test** split。其中 62 个用 recipe 插入或删除了 actor（如 R-4 动物、R-3 过马路的人群、I-3 堵在本车道的静止车辆、C-7 窄路会车），其余 218 个直接用原始日志 |
| 仿真器 | Isaac Sim 6.0 + Isaac Lab，是 pip 包，运行在 `navsafe eval` 进程里。负责自车动力学（LQR 跟踪规划轨迹）、交通参与者（日志回放、半反应式 IDM、行人 social force、定时急刹或切入）和碰撞检测 |
| 渲染器 | NVIDIA NuRec（3DGS），以 docker 镜像 `nvcr.io/nvidia/nre/nre-ga:26.04` 的形式提供，通过 gRPC 渲染相机画面 |
| policy 接口 | 每 0.5 s 重新规划一次。输入是 NAVSIM 相机、自车状态和导航 command（左转 / 直行 / 右转 / unknown），输出 4–5 s 的轨迹 |
| 评测协议 | 半反应式交通，route 预算 120 s，碰撞即终止 |
| 指标 | DS、SR（规则和 Bench2Drive 相同）、DE、comfort。碰撞、原地不动 5 s、开出可行驶区域、逆行都会结束 episode |

和 CARLA 的区别：
- 每个场景是一段日志的重建，范围有限。自车偏离原轨迹太远时，画面质量会下降。
- 部署上不需要每条 route 起关仿真服务器，但渲染器只有 docker 镜像这一种形式。

## 3. 我们的 AutoVLA 接入 NavSafe

AutoVLA 的代码在 `/home/mh2803/vla/doc_vla_search`，checkpoint 在 `/scratch/mh2803/vla/checkpoints/autovla-nocot-sft-navtrain-vit-unfreeze-adamw8bit-tigris8n/`（no-CoT SFT，在 navtrain 上训练）。NavSafe 自带的 `autovla` adapter 是给官方 checkpoint 写的，用到我们的模型上要改这几处：

| 项 | 现状 | 处理 |
|---|---|---|
| 相机 | 一致：训练时 nuPlan 分支也用 `CAM_L1` / `CAM_R1` 作为左前 / 右前相机（`autovla_agent.py:257`），和 adapter 相同 | 不改 |
| prompt | NavSafe 的 `autovla_server.py` 写死了上游的 CoT system prompt（`USE_COT=True`）；我们是 no-CoT 训练，用的是 `prompt_spec.SYSTEM_NOCOT`，user text 也不带后缀 | 写一个 server 变体，直接 import `dataset_utils/prompt_spec.py` |
| 动作 codebook | server 从 repo 目录读 `codebook_cache/agent_vocab.pkl` | 设置 `NAVSAFE_AUTOVLA_REPO=/home/mh2803/vla/doc_vla_search` |
| 导航 command | NavSafe 把 LANEFOLLOW 映射成 `"unknown"`；navtrain 里只有 turn left / keep forward / turn right，抽查 3000 条没有一条 unknown | **待定**：映射成 `keep forward`，还是保留 NavSafe 的做法。baseline 和我们的方法要用同一种，并在结果中注明 |
| checkpoint | Lightning 格式，server 会去掉 `autovla.` / `vlm.` 前缀 | 第一次运行时检查日志里打印的 missing keys 数量 |

## 4. 我们的方法在 NavSafe 上的形态

### 4.1 评测

新写一个 adapter，比如注册成 `autovla_cmd`，继承 `SensorPolicy`：
- 每次重新规划时，用前视相机调用 Qwen3-VL，可以复用 `scripts_a6000/1001/vlm_agent/vlm_server.py`。
- 把得到的指令拼在导航 instruction 后面。AutoVLA 先输出 meta-action，再输出动作 token。
- meta-action 写进 `parse_output` 返回的 `reasoning` 字段，用来统计闭环下指令和动作是否一致。
- 不读 `scenario_data` 里 recipe 或 actor 的特权信息，和 F2D 的规则相同。

和指令对应的事件类型：

| 指令 | 事件类型 |
|---|---|
| 减速 / 停车 | R-4 动物、R-3 人群、R-1 行人、R-2 自行车、V-10 并道抢行 |
| 换道 / 等空当 | I-3 静止车辆堵住本车道、I-1 路障、I-2 施工区、C-7 窄路会车 |
| 不发指令（只确认没有退化） | 掉头、逆行、红灯、超速、跟车等其余类型 |

和 F2D 不同的一点：那 218 个原始日志场景里的危险（锥桶、行人）在 navtrain 里本来就出现过，长尾程度比 F2D 低。最接近长尾的是插入了 actor 的 62 个场景。

### 4.2 SFT（在 AutoVLA 上做）

- **碰撞和可行性检查**：用 `navtrain_metric_cache`（35 GB，已有）配合 PDM scorer，对任意候选轨迹检查与日志车辆是否碰撞、是否在可行驶区域、行驶方向是否正确，代替 dreamer 的 OBB 检查。车道几何、限速、红绿灯从 nuPlan 地图读。
- **候选轨迹**：
  - 减速 / 停车：沿专家路径重新分配速度。
  - 加速：速度上限取道路限速；会撞前车或遇到红灯的样本丢掉。
  - 换道：平移到地图上相邻车道的中心线，从 0 m 开始横移，用 PDM 检查。
  - 等待：保持车道并减速。
- **轨迹转成 token**：AutoVLA 输出 10 个 codebook token（每个 0.5 s）。反事实轨迹要用 action tokenizer 编码，并先检查换道轨迹编码后的误差。
- **输出格式**：meta-action 放在动作 token 前面的文字段，代码已经支持"先文字后动作"（`reasoning_given` / `<answer>`）。指令写进 `prompt_spec.user_text`，训练和推理共用这一个函数。
- **"一致指令"数据**：用 nuPlan 的 scenario tag 挖，比如 `changing_lane`、`waiting_for_pedestrian_to_cross`、`near_construction_zone_sign`。

### 4.3 数据泄漏

NavSafe 的场景来自 nuPlan test split，navtrain 来自 trainval，两者的日志不重叠。NavSafe 的 asset、recipe、bundle 都不能进训练集；checkpoint 选择继续用从 navtrain 划出来的验证集。

## 5. 运行 NavSafe 需要的机器条件

| 类别 | 要求 | 说明 |
|---|---|---|
| 系统 | Linux x86-64，装好 NVIDIA 驱动 | Isaac Sim 和渲染器都不支持 ARM（如 GH200） |
| GPU | 带 RT core 的 RTX 显卡；渲染器单场景最多占 24 GB 显存，仿真器和 policy 另外占用 | Isaac Sim 渲染需要 RT core，A100 / H100 这类数据中心卡不行。官方用两张 24 GB 的 3090 测试；显存够大时可以全部放在一张卡上。我们还要额外放 AutoVLA（3B）和 Qwen3-VL-8B |
| CPU / 内存 | ≥ 4 核；内存 32 GB（评测）+ 16 GB（渲染器） | |
| 磁盘 | Python 环境约 25 GB（放本地盘，不要放网络文件系统）；渲染器镜像 28 GB；每个场景约 8.5 GB，280 个共约 2.2 TB；另加 model zoo 权重 | 每个场景的大小几乎全是 4 个 5 s 的 NuRec 重建文件（`.usdz`，每个约 2.1 GB）。共享的 `asset/` 和 `gait_bank/` 加起来只有约 0.14 GB；官方 AutoVLA 权重 16.3 GB，加上 Qwen2.5-VL-3B 基座 7.5 GB |
| 软件 | Python 3.12 + uv；VLA 类模型另需一个独立的 Python 环境 | |
| 容器 | Docker + NVIDIA Container Toolkit，或者能连到一个已经在运行的 NuRec gRPC 渲染器 | 渲染器和评测程序必须用同一个绝对路径读数据，所以最好放在同一台机器上，或者共享文件系统 |
| 账号 | NGC API key（免费，在 ngc.nvidia.com 注册）；接受 NVIDIA EULA（设置 `ACCEPT_EULA=Y`、`OMNI_KIT_ACCEPT_EULA=YES`） | |
| 权限 | 能使用 docker：root、docker 组，或者 rootless docker（需要 `uidmap`，且系统允许普通用户创建 user namespace） | |
| 网络 | 能访问 Hugging Face（下载数据集和权重）和 nvcr.io（拉镜像） | |

## 6. 本机（gcis-zxtics-rl）对照

| 条件 | 本机 | 是否满足 |
|---|---|---|
| 架构 | x86_64 | ✓ |
| GPU | 3 × RTX PRO 6000 Blackwell，每张 96 GB | ✓ |
| 驱动 | 595.91.07 | 应该可以；和 Isaac Sim 6.0、NRE 26.04 的兼容性没有实测 |
| CPU / 内存 | 72 核，251 GB | ✓ |
| 磁盘 | `/scratch` 剩 1.1 TB，`/` 剩 223 GB | 部分满足：不够放全部 280 个场景，先下载需要的几类（每类 10 个场景约 85 GB；R-4、I-3、C-7 共 30 个场景约 260 GB），也可以按类分批下载、评测完删除 |
| uv | 没装 | 可以自己装，不需要 root |
| Docker / 容器 | 没有 docker、podman、apptainer | ✗ |
| sudo | `mh2803` 不在 `sudo` 组（组里只有 `gccisadmin`），`sudo` 要密码 | ✗，以 `sudo -l` 的结果为准 |
| rootless 容器 | 系统开了 `kernel.apparmor_restrict_unprivileged_userns=1`，普通用户连 `unshare -Ur` 都会失败；也没有 `newuidmap` / `newgidmap` | ✗ |
| NGC key / EULA | 还没有 | 自己注册即可 |

## 7. 解决渲染器问题的三种办法

1. **请管理员（`gccisadmin`）安装**，二选一：
   - 装 docker-ce 和 nvidia-container-toolkit，把 `mh2803` 加进 docker 组。最省事，但 docker 组等同 root 权限。
   - 装 `uidmap`、`docker-ce-rootless-extras` 和 nvidia-container-toolkit，并按 Docker 文档给 rootlesskit 加 AppArmor profile。之后用户自己运行 `dockerd-rootless-setuptool.sh install`。
2. **不用容器跑渲染器**（不需要 root，但不确定能否跑通）：用 crane 或 skopeo 拿 NGC key 拉下镜像层，解包到 `/scratch`，然后直接在宿主机上运行 `serve-grpc`（用 `LD_LIBRARY_PATH` 或 proot）。镜像里可能有写死的路径，或依赖容器运行时注入驱动。这条路不在 NVIDIA 的支持范围内，跑通后要用官方模型跑一个场景，和 leaderboard 的数字核对。
3. **换一台有 docker 或 k8s 的 x86 RTX 机器**，整套评测都放在那台机器上。

## 8. 下一步

不依赖 docker、现在就能做的：
1. 装 uv，用 `uv sync` 装 NavSafe 的 Python 环境，初始化 `third_party/AutoVLA` submodule，装 VLA 环境。
2. 写 AutoVLA 的 server 变体（prompt、codebook 路径）。
3. 在 navtrain 上写反事实数据生成器（PDM 检查、地图读取相邻车道、轨迹编码成 token）。

渲染器能用之后：
1. 用官方 AutoVLA 跑 1–2 个场景，核对 leaderboard 数字。
2. 用我们的 no-CoT checkpoint 跑 baseline。
3. 实现 `autovla_cmd` adapter，加上 VLM，跑相关的几类事件。

待定：导航 command 的映射方式。

## 9. 决定（2026-10-22）

NavSafe 先不作为主要 benchmark：
- 闭环长尾评测用 Fail2Drive。
- 真实图像上的评测用 CoLT-Drive（开环，meta-action 成对评测）。
- AutoVLA 加指令后正常驾驶是否退化，用 NAVSIM navtest 检查。

如果之后需要"真实图像 + 闭环"的结果，只跑 R-4（动物）和 I-3（静止车辆堵住本车道）两类，共 20 个场景，约 170 GB。场景 token 取自 `navsafe/benchmark/recipes/benchmark/` 下的 recipe 文件名：

| 事件类型 | 场景 token |
|---|---|
| R-4 动物 | `005f87dd980253a5` `02015675e4585611` `0371700bf65b51e0` `05f1a5cbc0905d8e` `061e7e1700945b03` `173369dc059d5fe9` `225eb6e22af55972` `25b5cecdb3b75e7c` `2ee44628526e524c` `a2c2e046132e5596` |
| I-3 静止车辆堵路 | `03b66343e1ac5d68` `050284885b2059e9` `131a036a111e54f3` `4fff5f5a86be53d5` `76da778ff251508d` `84acc78da95f56d3` `89e02236312d5038` `a43dbb1d34665f24` `a6b9a83019915658` `b023b7bcbab05bcb` |

下载方式（需要时再执行；两类的场景都需要插入 actor，所以 `asset/` 和 `gait_bank/` 也要下载）：

```bash
export NAVSAFE_DATA_ROOT=/scratch/mh2803/vla/navsafe_data
for TOKEN in <上表 20 个 token>; do
  navsafe fetch --token "$TOKEN" --out "$NAVSAFE_DATA_ROOT"
done
hf download c13752hz/NavSafe --repo-type dataset --local-dir "$NAVSAFE_DATA_ROOT" \
  --include "asset/*" "gait_bank/*"
```

前置条件不变：渲染器需要 docker（见第 7 节）。
