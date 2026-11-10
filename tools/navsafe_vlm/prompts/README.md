# VLM prompts：哪个给哪个 benchmark

prompt 文件会被原样读进去当 prompt，所以说明写在这里，不写进 `.txt`。`{speed}` 由脚本填成当前车速（m/s）。

## CoLT-Drive

| 文件 | 用途 | 状态 |
|---|---|---|
| **`v28_single_f0.txt`** | **CoLT 测试用**。只看前视图，一次调用，输出 7 个标签之一 | Qwen3-VL-32B，vfull 73.9%（`tools/navsafe_vlm/colt_probe.py --single`）。AutoVLA 测评（`tools/colt/eval_autovla.py`）用的是它的标签 |
| **`v28_single_f0_paper.txt`** | **CoLT 论文版**。先输出类别、再输出动作两个 token，动作词表和 Fail2Drive 的 `taskA_v2_action_paper.txt` 一致 | — |

说明见 `docs/1104/colt_eval.md` §2。

## NavSafe

| 文件 | 用途 | 状态 |
|---|---|---|
| `v18_vru_f0.txt` | 只看前视图，只判行人 / 骑车的人 / 动物 / NONE | NavSafe 闭环在用（Qwen3-VL-8B，如 `vru12_l0r0cot_vlm_hold`，`run_vru_lane.sh` 的默认） |
| `v16_vru.txt` | 三张图（F0 / L0 / R0）版，标签同上 | `autovla_ours_vlm.py` 不设 `NAVSAFE_VLM_PROMPT` 时的默认 |
| **`v29_single_f0.txt`** | **NavSafe 用**。v28 的格式加上两句：转弯时路径跟着弯，人行道 / 门口 / 路口的人不算 | Qwen3-VL-8B，labels_v2 前视 413 帧：只算人 / 骑车 / 动物 93.7%（v18 93.5%），没有误报人车动物；BLOCKED 召回 7/26，NONE 帧误报障碍 28/170。adapter 还要认 AROUND / BLOCKED |

## 其他

`simlingo_taskA_best.txt` 是 Fail2Drive（CARLA）那边的参考。其余 `v1`–`v27`、`b1` / `b2` 是调 prompt 过程中的版本，不再使用，结果见 `docs/1102/experiment_design.md` §4 和 `docs/1104/colt_eval.md` §2.2。
