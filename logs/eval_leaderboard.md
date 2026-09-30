# NAVSIM navtest PDMS 评测汇总

> 整理日期：2026-09-30
> 数据来源：扫描 `dataset/` 下所有 eval run 的 `*_merged.csv`（PDMS = `score` 列均值），
> checkpoint 归属从每个 run 的 `code/hydra/config.yaml` 里的 `checkpoint_path` 读出，
> 再对回现存文件（部分训练目录后来加过后缀，已做重映射）。
> 下表路径均相对 `dataset/`（= `/data/autovla_data/`）。
> **ckpt 已不存在的 run 不在表内**，见文末「已剔除」。

---

## 1. 总表

| PDMS | n | checkpoint | 训练 run | 评测结果目录 |
|---:|---:|---|---|---|
| **89.48** | 12125 | `checkpoints/AutoVLA_PDMS_89.ckpt` | 官方发布 | `eval/nuplan/pdms_shards/navtest_fast_full.csv` |
| **84.61** | 12126 | `checkpoints/sft/2026-09-13_18-41-37_unfreeze_cot_103k/epoch=4-loss=0.4064.ckpt` | `navtrain_cot_vit_unfreeze_adamw8bit_4gpu` | `nuplan/sft_eval_0913_ep4_cot_full/` |
| 84.22 | 12126 | `checkpoints/sft/2026-09-10_05-35-55_unfreeze_nocot_103k/epoch=4-loss=0.6315.ckpt` | `navtrain_vit_unfreeze_adamw8bit_4gpu` | `nuplan/sft_eval_0910_ep4_nocot_full/` |
| 84.22 | 12123 | `checkpoints/sft/2026-09-10_05-35-55_unfreeze_nocot_103k/epoch=5-loss=0.6759.ckpt` | 同上 | `nuplan/sft_eval_0910_ep5_nocot_full/` |
| 83.65 | 1000 | `checkpoints/sft/2026-09-10_05-35-55_unfreeze_nocot_103k/epoch=4-loss=0.6315.ckpt` | 同上 | `nuplan/sft_eval_0910_ep4_nocot/` |
| 83.49 | 12126 | `checkpoints/sft/2026-09-10_05-35-55_unfreeze_nocot_103k/epoch=3-loss=0.6536.ckpt` | 同上 | `nuplan/sft_eval_0910_ep3_nocot_full/` |
| 83.13 | 1000 | `checkpoints/sft/2026-09-10_05-35-55_unfreeze_nocot_103k/epoch=3-loss=0.6536.ckpt` | 同上 | `nuplan/sft_eval_0910_ep3_nocot/` |
| 81.95 | 12125 | `checkpoints/sft/2026-09-10_05-33-40/epoch=2-loss=0.7087.ckpt` | `navtrain_vit_unfreeze_adamw8bit_2gpu` | `nuplan/sft_eval_0533_ep2_nocot_full/` |
| 80.63 | 12123 | `checkpoints/sft/2026-07-27_18-05-16/epoch=4-loss=0.6984.ckpt` | `navtrain_vit_frozen_0727_100k` | `eval/nuplan/sft_eval/` |
| 80.45 | 12121 | `checkpoints/sft/2026-09-02_05-47-39_bf16_cot_103k/epoch=4-loss=0.4436.ckpt` | `navtrain_cot_sft_0902` | `eval/nuplan/cot_sft_greedy_eval/epoch_4_loss_0_4436/` |
| 80.19 | 998 | `checkpoints/sft/2026-09-02_05-47-39_bf16_cot_103k/epoch=4-loss=0.4436.ckpt` | 同上 | `eval/nuplan/cot_sft_greedy_eval/103k_ep4_1000subset/` |
| 79.90 | 12126 | `checkpoints/sft/2026-09-05_20-34-30/epoch=3-loss=0.6501.ckpt` | `mix166k_nocot_sft_4gpu_0905` | `eval/nuplan/cot_sft_greedy_eval/epoch_3_loss_0_6501/` |
| 79.89 | 12123 | `checkpoints/sft/2026-09-02_05-47-39_bf16_cot_103k/epoch=3-loss=0.4446.ckpt` | `navtrain_cot_sft_0902` | `eval/nuplan/cot_sft_greedy_eval/epoch_3_loss_0_4446/` |
| 79.48 | 999 | `checkpoints/sft/2026-09-02_05-47-39_bf16_cot_103k/epoch=2-loss=0.4732.ckpt` | 同上 | `eval/nuplan/cot_sft_greedy_eval/103k_ep2_1000subset/` |
| 79.43 | 12125 | `checkpoints/sft/2026-07-27_18-05-16/epoch=3-loss=0.6743.ckpt` | `navtrain_vit_frozen_0727_100k` | `eval/nuplan/sft_eval/` |
| 79.02 | 12120 | `checkpoints/sft/2026-09-02_05-47-39_bf16_cot_103k/epoch=2-loss=0.4732.ckpt` | `navtrain_cot_sft_0902` | `eval/nuplan/cot_sft_greedy_eval/cotgreedy_merged.csv`（根目录那份） |
| 77.88 | 1000 | `checkpoints/sft/2026-09-05_20-34-30/epoch=2-loss=0.6905.ckpt` | `mix166k_nocot_sft_4gpu_0905` | `eval/nuplan/cot_sft_greedy_eval/epoch_2_loss_0_6905/` |
| 77.00 | 12126 | `checkpoints/sft/2026-09-03_20-56-46/epoch=3-loss=0.3547.ckpt` | `trainval166k_cot_sft_0902` | `nuplan/sft_eval_0903_ep3_cot_full/` |
| 76.54 | 1000 | `checkpoints/sft/2026-09-03_20-56-46/epoch=3-loss=0.3547.ckpt` | 同上 | `nuplan/sft_eval_0903_cot_1k/` |
| 74.65 | 1000 | `checkpoints/sft/2026-09-03_20-56-46/epoch=2-loss=0.3617.ckpt` | 同上 | `eval/nuplan/cot_sft_greedy_eval/epoch_2_loss_0_3617/` |
| 74.40 | 1000 | `checkpoints/sft/2026-09-03_20-56-46/epoch=2-loss=0.3617.ckpt` | 同上 | `nuplan/sft_eval_0903_cot_1k/` |
| 60.54 | 12126 | `checkpoints/sft/2026-07-23_04-41-59_bf16/epoch=4-loss=1.0498.ckpt` | bf16 master weight bug，见 `docs/0724/bf16_master_weights_bug.md` | `eval/nuplan/sft_eval_bf16full/` |
| 58.40 | 21 | `nuplan/lora_dryrun_merged.ckpt` | LoRA dry run，21 样本，**无参考价值** | `eval/nuplan/sft_eval_loradry/` |
| 27.51 | 555 | `checkpoints/AutoVLA_PDMS_89.ckpt` | 官方发布，**只跑低分样本子集，不可与全量比** | `eval/nuplan/pdms_shards/navtest_pdms_merged.csv` |

n = 有效场景数。navtest 全量 12146，`n≈12120–12126` 为全量（少量场景评测失败）；`n≈1000` 为 seed=42 随机子集，方差较大，只能同子集内横比。

---

## 2. 按训练 run 汇总

| 训练 run | 数据 | CoT | ViT | optimizer | 最佳 PDMS（全量） |
|---|---|---|---|---|---:|
| `2026-09-13_18-41-37_unfreeze_cot_103k` | navtrain 103k | ✅ | 解冻 | adamw8bit | **84.61** (ep4) |
| `2026-09-10_05-35-55_unfreeze_nocot_103k` | navtrain 103k | ❌ | 解冻 | adamw8bit | **84.22** (ep4/ep5) |
| `2026-09-10_05-33-40` | navtrain 103k | ❌ | 解冻 | adamw8bit (2gpu) | 81.95 (ep2) |
| `2026-07-27_18-05-16` | navtrain 103k | ❌ | 冻结 | — | 80.63 (ep4) |
| `2026-09-02_05-47-39_bf16_cot_103k` | navtrain 103k | ✅ | 冻结 | — | 80.45 (ep4) |
| `2026-09-05_20-34-30` | mix 166k | ❌ | 冻结 | — | 79.90 (ep3) |
| `2026-09-03_20-56-46` | trainval 166k | ✅ | 冻结 | — | 77.00 (ep3) |
| `2026-07-23_04-41-59_bf16` | navtrain 103k | ❌ | 冻结 | — | 60.54 (ep4)，bf16 bug |

尚未评测的训练 run：`2026-07-26_17-41-02_llmlora`、`2026-08-10_19-13-10_nuscenes`（走 nuScenes L2/Collision，不产 PDMS）、`2026-09-05_02-48-28`、`2026-09-05_14-11-00`、`2026-07-24_05-27-55`（空目录）。

---

## 3. 结论

**ViT 解冻是目前最大的单点增益。** 同为 103k no-CoT：冻结 80.63 → 解冻 84.22，+3.6 PDMS。

**CoT 只在解冻 + 103k 下才为正。** 同为 103k：冻结下 CoT 80.45 vs no-CoT 80.63（持平偏负）；解冻下 CoT 84.61 vs no-CoT 84.22（+0.4）。

**166k 数据是个回退。** CoT 166k 只有 77.00，比 CoT 103k 的 80.45 低 3.5；mix 166k no-CoT 79.90 也低于 103k no-CoT 80.63。同一批 CoT 数据换到 103k + 解冻能到 84.61，所以问题更像在 166k 数据本身（分布/质量），而不是 CoT 这个设计。相关分析见 `docs/0901/166k_vs_103k_distribution.md`。

**epoch 已到顶。** `unfreeze_nocot_103k` 的 ep3/ep4/ep5 = 83.49 / 84.22 / 84.22，ep4 之后完全平掉。

**离官方 89.48 还差 4.9。** 官方那份是 SFT + RFT(GRPO) 的结果，本地所有条目都只到 SFT。

---

## 4. 已剔除（ckpt 不存在）

4 个 RFT/GRPO 结果的权重记录在 `/home/nvidia/workspace/doc_drive_search/other_repo/AutoVLA/runs/grpo/2026-09-03_05-02-39/`，该仓库根目录已不存在，全盘也搜不到任何 `rft-step*.ckpt`，**无法复现**：

| PDMS | n | 原 checkpoint | CSV（仍在） |
|---:|---:|---|---|
| 80.24 | 12123 | `runs/grpo/2026-09-03_05-02-39/rft-step500-reward8.8125.ckpt` | `eval/nuplan/cot_sft_greedy_eval/rft_step500_reward8_8125/` |
| 79.25 | 12124 | `rft-step1000-reward8.6250.ckpt` | `.../rft_step1000_reward8_6250/` |
| 78.63 | 12124 | `rft-step1500-reward8.8125.ckpt` | `.../rft_step1500_reward8_8125/` |
| 78.53 | 12124 | `rft-step2000-reward8.5000.ckpt` | `.../rft_step2000_reward8_5000/` |

这批 RFT 的 SFT 起点是 `2026-09-02_05-47-39_bf16_cot_103k/epoch=4-loss=0.4436.ckpt`（80.45）。四个 step 越训越低，且没有一个超过起点 —— 当时的 GRPO 配置是失败的。

---

## 5. nuScenes（另一套指标）

nuScenes 走 L2 + Collision Rate，不产 PDMS，唯一一份结果在
`logs/0810_nvidia/eval_mix_ep012_2026-08-12_18-43-54/FINAL_ep012_results.txt`
（ckpt：`2026-08-10_19-13-10_nuscenes` 的 ep0/ep1/ep2，STP3 与 UniAD 两种定义，0.5–3.0s）。

---

## 6. 复现方式

PDMS = `*_merged.csv` 的 `score` 列均值 ×100。CSV schema：

```
Unnamed: 0, token, valid, no_at_fault_collisions, drivable_area_compliance,
ego_progress, time_to_collision_within_bound, comfort,
driving_direction_compliance, score
```

checkpoint 归属：
```bash
cat <eval_run>/<ckpt名>/shard0/<ckpt名>_s0/<时间戳>/code/hydra/config.yaml | grep checkpoint_path
```

注意两处坑：
1. 9 月之前的 run 写在 `dataset/eval/nuplan/`，9 月之后直接写在 `dataset/nuplan/`；后者里的 `sft_eval`、`sft_eval_bf16full`、`sft_eval_loradry`、`pdms_shards` 等是**软链接指回前者**，直接 glob 两个目录会出重复行。
2. hydra 里记的 `checkpoint_path` 是当时的绝对路径，训练目录后来加过后缀（如 `2026-09-10_05-35-55` → `2026-09-10_05-35-55_unfreeze_nocot_103k`），需按 `时间戳前缀 + 文件名` 重新匹配。
