# greedy rollout 的 BEV（**epoch=2** 权重 = 训了 3/6，navtest）

    python logs/0902/render_bev.py --jsonl logs/0902/navtest600_greedy_G1_ep2_poses.jsonl \
        --tokens <token 列表> --out logs/0902/bev_greedy --n 12 --pick any

ep1（2/6）的同一批帧在 `../bev_greedy_ep1/`，可逐帧对照。

图里的东西全部取自 **metric_cache 本身** —— PDMScorer 判分时实际看到的那份，
所以图上看到的就是判定依据。

```
浅灰面     可行驶区（判 DAC 用的那份）
灰框       其他 agent:实线 t=0s / 中等 t=2s / 虚线 t=4s
蓝虚线     centerline（route）
绿粗线     GT 轨迹
红/绿粗线  模型预测（红 = 判不可行）+ 每隔一个位姿的【车身框】+ 1~5s 时间刻度
粗点线     反事实速度剖面中【可行】的那几档，带终点方块 + 车身框虚线
细点线灰   不可行的那几档（✗）
```

## ⚠️ 两个看图必读

**① agent 是随时间移动的。** ego 轨迹跨 5s，只看 t=0 的框会把"轨迹穿过某个框"
误读成碰撞。`a6229e66c0e656d8` 就是这样:看着 GT 撞上 (18.8,−4.7) 的车，
实际 GT 是 `NC=1, PDMS=0.96` —— 那辆车（track 4661649c）在 t=0 在那儿，
ego 4.0~4.5s 才到，届时它已经开走。**对时间刻度看。**

**② 必须看车身框，不能只看中心线。** DAC 判的是车身四角在不在可行驶区里。

| 图 | token | PDMS | NC | DAC | said `<PLAN>` | D | said∈D |
|---|---|---|---|---|---|---|---|
| [00_b5f57f5a6b5b5244_g0.png](00_b5f57f5a6b5b5244_g0.png) | `b5f57f5a6b5b5244` | 0.862 | 1 | 1 | KEEP/STRAIGHT | ['HARD_BRAKE', 'BRAKE', 'KEEP', 'ACCEL'] | True |
| [01_566bd5e6d09153e0_g0.png](01_566bd5e6d09153e0_g0.png) | `566bd5e6d09153e0` | 1.000 | 1 | 1 | KEEP/STRAIGHT | ['HARD_BRAKE', 'BRAKE', 'KEEP', 'ACCEL'] | True |
| [02_078bc1027dde5d1a_g0.png](02_078bc1027dde5d1a_g0.png) | `078bc1027dde5d1a` | 0.000 | 1 | 0 | KEEP/RIGHT | ['HARD_BRAKE', 'BRAKE', 'KEEP'] | True |
| [03_9900336c08095d0f_g0.png](03_9900336c08095d0f_g0.png) | `9900336c08095d0f` | 0.000 | 1 | 0 | KEEP/LEFT | ['HARD_BRAKE', 'BRAKE'] | False |
| [04_0d63314a528159c9_g0.png](04_0d63314a528159c9_g0.png) | `0d63314a528159c9` | 0.000 | 1 | 0 | ACCELERATE/LEFT | ['HARD_BRAKE', 'BRAKE'] | False |
| [05_a6229e66c0e656d8_g0.png](05_a6229e66c0e656d8_g0.png) | `a6229e66c0e656d8` | 0.000 | 1 | 0 | DECELERATE/RIGHT | ['HARD_BRAKE', 'BRAKE'] | True |
| [06_25dbdd29ce325538_g0.png](06_25dbdd29ce325538_g0.png) | `25dbdd29ce325538` | 0.000 | 0 | 1 | KEEP/STRAIGHT | ['HARD_BRAKE', 'BRAKE'] | False |
| [07_807f6b90b60d5685_g0.png](07_807f6b90b60d5685_g0.png) | `807f6b90b60d5685` | 0.856 | 1 | 1 | KEEP/STRAIGHT | ['HARD_BRAKE', 'BRAKE', 'KEEP'] | True |
| [08_9b5c31b36c5b5aa9_g0.png](08_9b5c31b36c5b5aa9_g0.png) | `9b5c31b36c5b5aa9` | 0.000 | 0 | 1 | ACCELERATE/STRAIGHT | ['HARD_BRAKE', 'BRAKE'] | False |
| [09_2d2a1f08e4895258_g0.png](09_2d2a1f08e4895258_g0.png) | `2d2a1f08e4895258` | 1.000 | 1 | 1 | ACCELERATE/STRAIGHT | ['HARD_BRAKE', 'BRAKE', 'ACCEL'] | True |
| [10_bcbe67b78b825aec_g0.png](10_bcbe67b78b825aec_g0.png) | `bcbe67b78b825aec` | 1.000 | 1 | 1 | STOP/STRAIGHT | ['HARD_BRAKE', 'BRAKE'] | True |
| [11_2a3ab48ae28f5e90_g0.png](11_2a3ab48ae28f5e90_g0.png) | `2a3ab48ae28f5e90` | 1.000 | 1 | 1 | ACCELERATE/LEFT | ['HARD_BRAKE', 'BRAKE', 'KEEP', 'ACCEL'] | True |

## 论文备选（见 ../step0_gonogo.md 末节）

- **`08_9b5c31b36c5b5aa9`** ⭐ 主图候选。NC=0 真碰撞:说 `ACCELERATE` 撞上停放车辆，
  而 `HARD_BRAKE`/`BRAKE` 两条反事实**可行**（紫/蓝，带终点方块）。teacher 判"更慢"，对症。
- **`03_9900336c08095d0f`** 弃权案例。说 `KEEP,LEFT` 完全正确，失效是横向跟踪漂移。
- **`00_b5f57f5a6b5b5244`** ep1 PDMS 0.000 → ep2 0.862，训练收敛的对照。
