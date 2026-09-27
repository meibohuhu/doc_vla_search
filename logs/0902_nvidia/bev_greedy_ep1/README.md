# greedy rollout 的 BEV（epoch=1 权重，navtest）

生成:`python logs/0902/render_bev.py --jsonl logs/0902/bev_samples.jsonl --out logs/0902/bev_greedy --n 12 --pick any`
轨迹来自 `logs/0902/bev_samples.jsonl`（12 个挑出来的 token，T=0.01、G=1、带 poses）。

图里画的东西全部取自 **metric_cache 本身** —— 也就是 PDMScorer 判分时实际看到的那份，
所以图上看到的就是判定依据，不会出现"图和分数对不上"。

```
浅灰面     可行驶区（判 DAC 用的那份）
灰框       其他 agent:实线 t=0，淡色 t=2s
蓝虚线     centerline（route）
绿粗线     GT 轨迹
红/绿粗线  模型预测（红 = 判不可行）+ 每隔一个位姿的【车身框】
细点线     4 条反事实速度剖面，✗ = 不在 D 里
```

⚠️ **必须看车身框，不能只看中心线。** DAC 判的是车身四角在不在可行驶区里 ——
有的图中心线明明在路上，但右前角已经出界了（如 07）。

| 图 | token | PDMS | NC | DAC | said `<PLAN>` | D | said∈D |
|---|---|---|---|---|---|---|---|
| [00_078bc1027dde5d1a_g0.png](00_078bc1027dde5d1a_g0.png) | `078bc1027dde5d1a` | 0.000 | 1 | 0 | ACCELERATE/RIGHT | ['HARD_BRAKE', 'BRAKE', 'KEEP'] | False |
| [01_0d63314a528159c9_g0.png](01_0d63314a528159c9_g0.png) | `0d63314a528159c9` | 0.000 | 1 | 0 | ACCELERATE/LEFT | ['HARD_BRAKE', 'BRAKE'] | False |
| [02_25dbdd29ce325538_g0.png](02_25dbdd29ce325538_g0.png) | `25dbdd29ce325538` | 0.000 | 0 | 1 | DECELERATE/STRAIGHT | ['HARD_BRAKE', 'BRAKE'] | True |
| [03_2a3ab48ae28f5e90_g0.png](03_2a3ab48ae28f5e90_g0.png) | `2a3ab48ae28f5e90` | 0.953 | 1 | 1 | ACCELERATE/LEFT | ['HARD_BRAKE', 'BRAKE', 'KEEP', 'ACCEL'] | True |
| [04_2d2a1f08e4895258_g0.png](04_2d2a1f08e4895258_g0.png) | `2d2a1f08e4895258` | 1.000 | 1 | 1 | ACCELERATE/STRAIGHT | ['HARD_BRAKE', 'BRAKE', 'ACCEL'] | True |
| [05_566bd5e6d09153e0_g0.png](05_566bd5e6d09153e0_g0.png) | `566bd5e6d09153e0` | 0.000 | 1 | 0 | KEEP/STRAIGHT | ['HARD_BRAKE', 'BRAKE'] | False |
| [06_807f6b90b60d5685_g0.png](06_807f6b90b60d5685_g0.png) | `807f6b90b60d5685` | 0.000 | 0 | 1 | KEEP/STRAIGHT | ['HARD_BRAKE', 'BRAKE'] | False |
| [07_9900336c08095d0f_g0.png](07_9900336c08095d0f_g0.png) | `9900336c08095d0f` | 0.000 | 1 | 0 | KEEP/LEFT | ['HARD_BRAKE', 'BRAKE'] | False |
| [08_9b5c31b36c5b5aa9_g0.png](08_9b5c31b36c5b5aa9_g0.png) | `9b5c31b36c5b5aa9` | 0.000 | 0 | 1 | ACCELERATE/STRAIGHT | ['HARD_BRAKE', 'BRAKE'] | False |
| [09_a6229e66c0e656d8_g0.png](09_a6229e66c0e656d8_g0.png) | `a6229e66c0e656d8` | 0.000 | 1 | 0 | DECELERATE/RIGHT | ['HARD_BRAKE', 'BRAKE'] | True |
| [10_b5f57f5a6b5b5244_g0.png](10_b5f57f5a6b5b5244_g0.png) | `b5f57f5a6b5b5244` | 0.000 | 1 | 0 | KEEP/STRAIGHT | ['HARD_BRAKE', 'BRAKE'] | False |
| [11_bcbe67b78b825aec_g0.png](11_bcbe67b78b825aec_g0.png) | `bcbe67b78b825aec` | 1.000 | 1 | 1 | STOP/STRAIGHT | ['HARD_BRAKE', 'BRAKE'] | True |

## 两个值得单独看的

**`00_078bc1027dde5d1a`** —— 转弯半径太大，从路口右侧甩出可行驶区。
说的是 `ACCELERATE,RIGHT`，**方向判断是对的**（GT 也是右转），错的是转得太宽。
但 D = [HARD_BRAKE, BRAKE, KEEP]，ACCEL 不在里面 → 落进 `chg`，
teacher 会把 ACCELERATE 改成 KEEP —— **教它减速，而真正的问题是转向。**

**`07_9900336c08095d0f`** —— 更隐蔽。中心线几乎贴着 GT，但从 x≈18 开始
车身右前角已经压出路沿（看红色车框）。说的是 `KEEP,LEFT`，**同样是对的**，
D = [HARD_BRAKE, BRAKE] → 又落进 `chg`，teacher 又会教减速。

这两张就是 `step0_gonogo.md` 里"69.5% 的触发是纯横向失效"那条结论的直观版本，
也是"触发器只认 NC=0"这个改动的理由。
