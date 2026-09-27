# NAVSIM reasoning —— 按类别抽样(**真实训练 token 集**)

日期 2026-09-02 · 数据源 `cot_navtrain.json`(100,222 条)· 生成器 [`build_navsim_cot.py`](../../../Impromptu-VLA/data_qa_generate/data_engine/datasets/navsim/build_navsim_cot.py)

> 🔴 **和 v1/v2/v4 的关键区别**:那几份是在 log 里【等间隔取帧】抽的,
> 而这一份抽自 AutoVLA 真正训练用的 token 集(NAVSIM 按场景筛选过、`has_route: true`)。
> **两者的场景分布差很多**,以这一份为准。

## 真实训练集的分布(n=100,222)

```
子模式  none 67.6%   stay_behind 29.5%   cross_path 1.0%   stop_sign 1.0%   yield_vru 0.5%   gap_big 0.4%
SPEED   KEEP 34.7%   ACCELERATE 28.2%   STOP 19.4%   DECELERATE 17.6%
PATH    STRAIGHT 70.1%   LEFT 20.7%   RIGHT 9.3%
灯      不可评 86.1%   GREEN 8.5%   RED 5.4%
```

⚠️ 对照按 log 等间隔取帧的旧样本(v1/v2/v4):`STOP 46.7%→16.8%`、`red_light 5.8%→0.6%`。
**红灯这条线在真实训练集里只覆盖 0.6%。**

### 🔴 red_light 已砍掉(2026-09-02)

真实训练集里它只占 0.6%，而为它踩了**六个坑**:

```
① 字段语义反转——traffic_lights 的 flag=True 是【红】，不是绿
② lane_connector_id 与 roadblock_id 是两套 id，直接求交集恒为 0
③ "所有下游 connector + 全红才算红" → 路口处左转红/直行绿被判成红
④ 切换期不可靠——记录滞后 1 帧、随后消失 3 帧（像素统计逐帧验证过）
⑤ 路线(roadblock_ids)在多平行车道处选错 connector
   实测一帧 15 条灯记录【没有一条】在路线上；另一帧唯一"在路线"的红灯
   离 ego 实际轨迹 17.4m，而 ego 真正压过的那条(0.4m)是绿的
⑥ 改用 ego 实际轨迹匹配后有【选择效应】——红灯时 ego 停下、开不过去，
   恰好匹配不上，P(停或减|RED) 掉到 21%
```

人工抽查 15 个 `red_light` 样本，**连着抓到 3 个画面是绿灯**。
留着它 = 在 0.6% 的帧上注入一个已知会错、且读起来完全通顺的假陈述。
**理由从句和观察句一起砍** —— 灯态不可靠对两者一视同仁。
json 里的 `traffic_light` 字段保留做元数据，但不进句子。

`stop_sign` 保留:静态地图元素，上面六个坑一个都不适用。

### 🔴 标签判据的三处修正

```
SPEED   STOP ⇔ v[4]<2  且 ( v_min<0.5 或 v_end<2 )
        旧版只有 v[4]<2 —— 把【起步初期】误判成 STOP，占 STOP 的 20.5%、全量 4.2%
        两种误判形态:
          纯起步    v = [0.57, 0.68, 0.76, 0.89, 1.23, 1.69, 2.20, …, 4.10]
          减速再走   v = [2.99, 2.71, 2.37, 2.08, 1.86, 1.82, 2.20, …, 3.81]
        真实 token 集实测:误判 126→0，漏判持平，DECELERATE 质量保住(19.6%)

PATH    横向判据用【横纵比 ≥0.20】，不是绝对横移 ≥3m
        旧版:5 秒纵向走 33.6m、横向偏 3.71m（航向仅 +11°、导航 STRAIGHT）
              被标成 LEFT —— 那是【顺着弯道走】，绝对阈值与前进距离无关
        实测:精度 82%→95%，代价是召回 97.0%→91.7%
        ⚠️ 漏掉的是【中高速变道】(10 m/s 变道横纵比只有 0.07)。
           试过用"航向形状"(变道 S 形回正 vs 弯道单调)区分——【不成立】:
           |lat|≥3 的帧里只有 1.1% 呈现末段回正，因为变道要 4~6 秒，
           5 秒窗口装不下完整的 S 形。这是个没有好解的取舍，选了精度。
           PATH 是 teacher 从不碰的不变量，只影响句子自然度。
```

变动量:SPEED 2.0%、PATH 8.7%、句子 10.5%。

## 句式(四个槽，都可能为空)

```
[VRU]      I see a pedestrian N meters ahead of me.
[前车/切入]  The vehicle N meters ahead in my lane is stopped.
           ／ A vehicle N meters on my left will move into my path.
[理由+决策]  Because I need to stay behind it, I should come to a stop:
           <PLAN>STOP,STRAIGHT</PLAN>                        ← teacher 只改 SPEED 那一半
```

---

## 场景:`stay_behind` —— 跟车 —— 我车道里有前车　(全量占比 29.5%)

### 1. `DECELERATE,STRAIGHT`　v0=7.0 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.09.15.13.16.40_veh-28_00642_01267 · token `b8add10a033b5b6e`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/stay_behind_1_CAM_F0.jpg) | ![](image_v5/stay_behind_1_CAM_L0.jpg) | ![](image_v5/stay_behind_1_CAM_R0.jpg) | ![](image_v5/stay_behind_1_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 5 meters ahead in my lane is stopped. Because I need to stay behind it, I should slow down: <PLAN>DECELERATE,STRAIGHT</PLAN>

### 2. `STOP,STRAIGHT`　v0=1.6 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.05.12.22.28.35_veh-35_02138_02481 · token `b9e3016cb0ac517f`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/stay_behind_2_CAM_F0.jpg) | ![](image_v5/stay_behind_2_CAM_L0.jpg) | ![](image_v5/stay_behind_2_CAM_R0.jpg) | ![](image_v5/stay_behind_2_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 13 meters ahead in my lane is stopped. Because I need to stay behind it, I should remain stopped: <PLAN>STOP,STRAIGHT</PLAN>

### 3. `STOP,STRAIGHT`　v0=0.4 m/s　nav=STRAIGHT　灯=GREEN
<sub>2021.06.14.19.22.11_veh-38_01871_02040 · token `946b417c8afb5683`</sub>

```
q1/q2 最近 3 个对象（共 15）
    Vehicle    side right    (   1.7,  -3.6) m    0.0 m/s
    Vehicle    side right    (  -4.9,  -4.0) m    0.0 m/s
    Vehicle    side right    (  -1.5,  -6.9) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/stay_behind_3_CAM_F0.jpg) | ![](image_v5/stay_behind_3_CAM_L0.jpg) | ![](image_v5/stay_behind_3_CAM_R0.jpg) | ![](image_v5/stay_behind_3_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 8 meters ahead in my lane is stopped. Because I need to stay behind it, I should remain stopped: <PLAN>STOP,STRAIGHT</PLAN>

### 4. `STOP,STRAIGHT`　v0=1.9 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.06.14.17.26.26_veh-38_01860_02729 · token `e953857739f05797`</sub>

```
q1/q2 最近 3 个对象（共 17）
    Vehicle    front center  (   7.7,   3.3) m    1.7 m/s
    Vehicle    rear right    ( -10.9,  -6.2) m    4.8 m/s
    Vehicle    front center  (  13.3,  -0.2) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/stay_behind_4_CAM_F0.jpg) | ![](image_v5/stay_behind_4_CAM_L0.jpg) | ![](image_v5/stay_behind_4_CAM_R0.jpg) | ![](image_v5/stay_behind_4_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 7 meters ahead in my lane is moving slowly. Because I need to stay behind it, I should remain stopped: <PLAN>STOP,STRAIGHT</PLAN>

### 5. `DECELERATE,STRAIGHT`　v0=6.7 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.06.08.18.18.30_veh-38_06017_06142 · token `23a61e9352c35052`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/stay_behind_5_CAM_F0.jpg) | ![](image_v5/stay_behind_5_CAM_L0.jpg) | ![](image_v5/stay_behind_5_CAM_R0.jpg) | ![](image_v5/stay_behind_5_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 25 meters ahead in my lane is moving at 3 m/s. Because I need to stay behind it, I should slow down: <PLAN>DECELERATE,STRAIGHT</PLAN>

### 6. `STOP,STRAIGHT`　v0=1.8 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.06.14.18.33.41_veh-35_03575_03668 · token `1cb47ab853245446`</sub>

```
q1/q2 最近 3 个对象（共 75）
    Vehicle    side right    (  -4.0,  -3.8) m    5.0 m/s
    Vehicle    side right    (  -0.9, -10.4) m    0.0 m/s
    Pedestrian side right    (   3.7,  -9.8) m    1.3 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/stay_behind_6_CAM_F0.jpg) | ![](image_v5/stay_behind_6_CAM_L0.jpg) | ![](image_v5/stay_behind_6_CAM_R0.jpg) | ![](image_v5/stay_behind_6_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 14 meters ahead in my lane is stopped. Because I need to stay behind it, I should remain stopped: <PLAN>STOP,STRAIGHT</PLAN>

### 7. `DECELERATE,STRAIGHT`　v0=3.9 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.07.16.01.22.41_veh-14_04315_07102 · token `3c8b0ae2f2e95cbe`</sub>

```
q1/q2 最近 3 个对象（共 16）
    Vehicle    front center  (   5.4,   3.6) m    9.0 m/s
    Vehicle    front left    (  12.5,   6.4) m    8.5 m/s
    Pedestrian front right   (  10.6, -10.0) m    1.1 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/stay_behind_7_CAM_F0.jpg) | ![](image_v5/stay_behind_7_CAM_L0.jpg) | ![](image_v5/stay_behind_7_CAM_R0.jpg) | ![](image_v5/stay_behind_7_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 5 meters ahead in my lane is moving at 8 m/s. Because I need to stay behind it, I should slow down: <PLAN>DECELERATE,STRAIGHT</PLAN>

### 8. `STOP,STRAIGHT`　v0=1.5 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.06.09.20.26.11_veh-35_00825_00942 · token `12808b86e11b5684`</sub>

```
q1/q2 最近 3 个对象（共 27）
    Vehicle    side right    (   0.3,  -3.5) m    0.0 m/s
    Vehicle    front center  (   6.8,  -3.9) m    0.0 m/s
    Vehicle    side left     (   3.9,   9.3) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/stay_behind_8_CAM_F0.jpg) | ![](image_v5/stay_behind_8_CAM_L0.jpg) | ![](image_v5/stay_behind_8_CAM_R0.jpg) | ![](image_v5/stay_behind_8_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 6 meters ahead in my lane is stopped. Because I need to stay behind it, I should remain stopped: <PLAN>STOP,STRAIGHT</PLAN>

### 9. `STOP,STRAIGHT`　v0=3.9 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.07.16.02.53.40_veh-17_00016_01588 · token `d6adceab73e8503f`</sub>

```
q1/q2 最近 3 个对象（共 25）
    Vehicle    rear left     (  -8.6,   1.0) m    4.4 m/s
    Vehicle    front center  (  10.0,   2.7) m    0.0 m/s
    Vehicle    front center  (  16.4,   1.8) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/stay_behind_9_CAM_F0.jpg) | ![](image_v5/stay_behind_9_CAM_L0.jpg) | ![](image_v5/stay_behind_9_CAM_R0.jpg) | ![](image_v5/stay_behind_9_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 9 meters ahead in my lane is stopped. Because I need to stay behind it, I should come to a stop: <PLAN>STOP,STRAIGHT</PLAN>

### 10. `DECELERATE,RIGHT`　v0=5.7 m/s　nav=RIGHT　灯=无记录
<sub>2021.05.12.22.28.35_veh-35_02138_02481 · token `8da9f349061c5f93`</sub>

```
q1/q2 最近 3 个对象（共 5）
    Vehicle    front center  (  16.8,  -2.4) m    4.9 m/s
    Vehicle    front left    (  13.4,  14.0) m    9.4 m/s
    Vehicle    rear left     ( -24.1,   3.7) m    7.7 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/stay_behind_10_CAM_F0.jpg) | ![](image_v5/stay_behind_10_CAM_L0.jpg) | ![](image_v5/stay_behind_10_CAM_R0.jpg) | ![](image_v5/stay_behind_10_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 16 meters ahead in my lane is moving at 4 m/s. Because I need to stay behind it, I should slow down and bear right: <PLAN>DECELERATE,RIGHT</PLAN>

### 11. `STOP,STRAIGHT`　v0=1.0 m/s　nav=LEFT　灯=无记录
<sub>2021.07.09.23.23.48_veh-26_04648_06327 · token `32b1e375e496597d`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/stay_behind_11_CAM_F0.jpg) | ![](image_v5/stay_behind_11_CAM_L0.jpg) | ![](image_v5/stay_behind_11_CAM_R0.jpg) | ![](image_v5/stay_behind_11_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 10 meters ahead in my lane is moving slowly. Because I need to stay behind it, I should remain stopped: <PLAN>STOP,STRAIGHT</PLAN>

### 12. `STOP,STRAIGHT`　v0=0.2 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.07.16.16.08.35_veh-35_04744_06051 · token `30bec010ffd951cf`</sub>

```
q1/q2 最近 3 个对象（共 10）
    Vehicle    side left     (  -2.4,   6.4) m    8.4 m/s
    Vehicle    rear right    (  -7.8,  -0.0) m    0.0 m/s
    Vehicle    front center  (  10.8,   0.0) m    1.2 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/stay_behind_12_CAM_F0.jpg) | ![](image_v5/stay_behind_12_CAM_L0.jpg) | ![](image_v5/stay_behind_12_CAM_R0.jpg) | ![](image_v5/stay_behind_12_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 10 meters ahead in my lane is moving slowly. Because I need to stay behind it, I should remain stopped: <PLAN>STOP,STRAIGHT</PLAN>

### 13. `STOP,STRAIGHT`　v0=2.9 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.07.09.17.06.37_veh-35_02609_05015 · token `05cd45426dd55fb6`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/stay_behind_13_CAM_F0.jpg) | ![](image_v5/stay_behind_13_CAM_L0.jpg) | ![](image_v5/stay_behind_13_CAM_R0.jpg) | ![](image_v5/stay_behind_13_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 10 meters ahead in my lane is stopped. Because I need to stay behind it, I should come to a stop: <PLAN>STOP,STRAIGHT</PLAN>

### 14. `DECELERATE,STRAIGHT`　v0=6.7 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.06.23.15.18.10_veh-26_00165_02848 · token `47d6c868f2cf5b50`</sub>

```
q1/q2 最近 3 个对象（共 7）
    Pedestrian front right   (  11.0,  -8.8) m    1.2 m/s
    Vehicle    front left    (  16.9,  10.3) m    3.2 m/s
    Vehicle    front left    (  22.8,   7.1) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/stay_behind_14_CAM_F0.jpg) | ![](image_v5/stay_behind_14_CAM_L0.jpg) | ![](image_v5/stay_behind_14_CAM_R0.jpg) | ![](image_v5/stay_behind_14_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 28 meters ahead in my lane is moving at 2 m/s. Because I need to stay behind it, I should slow down: <PLAN>DECELERATE,STRAIGHT</PLAN>

### 15. `STOP,STRAIGHT`　v0=3.1 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.07.16.18.19.22_veh-35_00869_03454 · token `164554700350586b`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/stay_behind_15_CAM_F0.jpg) | ![](image_v5/stay_behind_15_CAM_L0.jpg) | ![](image_v5/stay_behind_15_CAM_R0.jpg) | ![](image_v5/stay_behind_15_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 17 meters ahead in my lane is stopped. Because I need to stay behind it, I should come to a stop: <PLAN>STOP,STRAIGHT</PLAN>

---

## 场景:`yield_vru` —— 让行 VRU —— 正前走廊 ≤20m 有行人/骑行者　(全量占比 0.5%)

### 1. `STOP,STRAIGHT`　v0=3.1 m/s　nav=LEFT　灯=无记录
<sub>2021.06.09.12.39.51_veh-26_00609_01168 · token `b3ee54e0344658d7`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/yield_vru_1_CAM_F0.jpg) | ![](image_v5/yield_vru_1_CAM_L0.jpg) | ![](image_v5/yield_vru_1_CAM_R0.jpg) | ![](image_v5/yield_vru_1_CAM_B0.jpg) |

**生成的 reasoning:**

> I see pedestrians 19 meters ahead of me. Because I need to yield to them, I should come to a stop: <PLAN>STOP,STRAIGHT</PLAN>

### 2. `DECELERATE,STRAIGHT`　v0=3.3 m/s　nav=LEFT　灯=无记录
<sub>2021.05.12.23.36.44_veh-35_01735_01957 · token `f5d4db945cd3573b`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/yield_vru_2_CAM_F0.jpg) | ![](image_v5/yield_vru_2_CAM_L0.jpg) | ![](image_v5/yield_vru_2_CAM_R0.jpg) | ![](image_v5/yield_vru_2_CAM_B0.jpg) |

**生成的 reasoning:**

> I see a pedestrian 19 meters ahead of me. Because I need to yield to them, I should slow down: <PLAN>DECELERATE,STRAIGHT</PLAN>

### 3. `STOP,STRAIGHT`　v0=2.1 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.06.14.16.32.09_veh-35_01781_02379 · token `f9316a3c17ff5dd5`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/yield_vru_3_CAM_F0.jpg) | ![](image_v5/yield_vru_3_CAM_L0.jpg) | ![](image_v5/yield_vru_3_CAM_R0.jpg) | ![](image_v5/yield_vru_3_CAM_B0.jpg) |

**生成的 reasoning:**

> I see pedestrians 14 meters ahead of me. Because I need to yield to them, I should come to a stop: <PLAN>STOP,STRAIGHT</PLAN>

### 4. `STOP,STRAIGHT`　v0=3.2 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.06.09.17.37.09_veh-12_02324_02434 · token `796905cc89e05d4d`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/yield_vru_4_CAM_F0.jpg) | ![](image_v5/yield_vru_4_CAM_L0.jpg) | ![](image_v5/yield_vru_4_CAM_R0.jpg) | ![](image_v5/yield_vru_4_CAM_B0.jpg) |

**生成的 reasoning:**

> I see pedestrians 15 meters ahead of me. Because I need to yield to them, I should come to a stop: <PLAN>STOP,STRAIGHT</PLAN>

### 5. `STOP,STRAIGHT`　v0=4.9 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.07.16.18.49.56_veh-26_00833_03384 · token `a01a34a4ee2950b3`</sub>

```
q1/q2 最近 3 个对象（共 98）
    Vehicle    side right    (   0.8, -10.7) m    0.0 m/s
    Vehicle    side right    (  -3.8, -10.5) m    0.0 m/s
    Pedestrian front right   (   8.0, -10.2) m    1.5 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/yield_vru_5_CAM_F0.jpg) | ![](image_v5/yield_vru_5_CAM_L0.jpg) | ![](image_v5/yield_vru_5_CAM_R0.jpg) | ![](image_v5/yield_vru_5_CAM_B0.jpg) |

**生成的 reasoning:**

> I see pedestrians 18 meters ahead of me. Because I need to yield to them, I should come to a stop: <PLAN>STOP,STRAIGHT</PLAN>

### 6. `STOP,STRAIGHT`　v0=2.3 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.07.16.20.45.29_veh-35_00600_01084 · token `294f998310d357a6`</sub>

```
q1/q2 最近 3 个对象（共 133）
    Vehicle    side right    (   2.8,  -3.3) m    0.0 m/s
    Pedestrian front left    (   6.8,   8.9) m    1.3 m/s
    Pedestrian rear left     (  -9.9,   5.7) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/yield_vru_6_CAM_F0.jpg) | ![](image_v5/yield_vru_6_CAM_L0.jpg) | ![](image_v5/yield_vru_6_CAM_R0.jpg) | ![](image_v5/yield_vru_6_CAM_B0.jpg) |

**生成的 reasoning:**

> I see pedestrians 12 meters ahead of me. Because I need to yield to them, I should come to a stop: <PLAN>STOP,STRAIGHT</PLAN>

### 7. `STOP,STRAIGHT`　v0=0.8 m/s　nav=LEFT　灯=无记录
<sub>2021.07.16.02.53.40_veh-17_00016_01588 · token `36683cf7c1745d2a`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/yield_vru_7_CAM_F0.jpg) | ![](image_v5/yield_vru_7_CAM_L0.jpg) | ![](image_v5/yield_vru_7_CAM_R0.jpg) | ![](image_v5/yield_vru_7_CAM_B0.jpg) |

**生成的 reasoning:**

> I see pedestrians 10 meters ahead of me. Because I need to yield to them, I should remain stopped: <PLAN>STOP,STRAIGHT</PLAN>

### 8. `STOP,STRAIGHT`　v0=2.7 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.06.09.17.37.09_veh-12_01386_01454 · token `4fa19b20f26a5caf`</sub>

```
q1/q2 最近 3 个对象（共 116）
    Vehicle    rear right    ( -11.6,  -5.6) m    3.6 m/s
    Pedestrian front left    (   7.4,  12.7) m    1.4 m/s
    Vehicle    rear right    ( -11.8, -10.4) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/yield_vru_8_CAM_F0.jpg) | ![](image_v5/yield_vru_8_CAM_L0.jpg) | ![](image_v5/yield_vru_8_CAM_R0.jpg) | ![](image_v5/yield_vru_8_CAM_B0.jpg) |

**生成的 reasoning:**

> I see pedestrians 15 meters ahead of me. Because I need to yield to them, I should come to a stop: <PLAN>STOP,STRAIGHT</PLAN>

### 9. `STOP,STRAIGHT`　v0=2.6 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.07.16.16.27.22_veh-26_01536_02260 · token `e05d86da0bd65c4b`</sub>

```
q1/q2 最近 3 个对象（共 62）
    Vehicle    rear right    (  -6.8,  -5.8) m    6.9 m/s
    Pedestrian side right    (   0.1, -10.1) m    0.9 m/s
    Pedestrian front center  (  11.0,   2.9) m    0.5 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/yield_vru_9_CAM_F0.jpg) | ![](image_v5/yield_vru_9_CAM_L0.jpg) | ![](image_v5/yield_vru_9_CAM_R0.jpg) | ![](image_v5/yield_vru_9_CAM_B0.jpg) |

**生成的 reasoning:**

> I see pedestrians 10 meters ahead of me. Because I need to yield to them, I should come to a stop: <PLAN>STOP,STRAIGHT</PLAN>

### 10. `DECELERATE,STRAIGHT`　v0=5.9 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.07.16.00.51.05_veh-17_03264_05261 · token `e600873f19025daf`</sub>

```
q1/q2 最近 3 个对象（共 8）
    Pedestrian front right   (   5.7,  -4.7) m    0.9 m/s
    Pedestrian front center  (   6.6,  -3.7) m    1.0 m/s
    Pedestrian front right   (   7.0,  -4.1) m    1.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/yield_vru_10_CAM_F0.jpg) | ![](image_v5/yield_vru_10_CAM_L0.jpg) | ![](image_v5/yield_vru_10_CAM_R0.jpg) | ![](image_v5/yield_vru_10_CAM_B0.jpg) |

**生成的 reasoning:**

> I see a pedestrian 6 meters ahead of me. Because I need to yield to them, I should slow down: <PLAN>DECELERATE,STRAIGHT</PLAN>

### 11. `STOP,STRAIGHT`　v0=0.8 m/s　nav=LEFT　灯=无记录
<sub>2021.06.23.21.56.29_veh-35_00220_00936 · token `efca95aef7615995`</sub>

```
q1/q2 最近 3 个对象（共 23）
    Vehicle    side right    (   0.4,  -4.2) m    0.0 m/s
    Vehicle    side left     (  -3.4,   3.8) m    0.6 m/s
    Pedestrian side left     (   4.6,   2.5) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/yield_vru_11_CAM_F0.jpg) | ![](image_v5/yield_vru_11_CAM_L0.jpg) | ![](image_v5/yield_vru_11_CAM_R0.jpg) | ![](image_v5/yield_vru_11_CAM_B0.jpg) |

**生成的 reasoning:**

> I see pedestrians 5 meters ahead of me. Because I need to yield to them, I should remain stopped: <PLAN>STOP,STRAIGHT</PLAN>

### 12. `STOP,STRAIGHT`　v0=0.9 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.06.14.17.26.26_veh-38_01860_02729 · token `dfbfec6906dc5199`</sub>

```
q1/q2 最近 3 个对象（共 11）
    Vehicle    side right    (  -2.8,  -3.9) m    0.5 m/s
    Vehicle    rear left     (  -9.0,   0.7) m    0.8 m/s
    Pedestrian side right    (  -3.4, -11.2) m    1.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/yield_vru_12_CAM_F0.jpg) | ![](image_v5/yield_vru_12_CAM_L0.jpg) | ![](image_v5/yield_vru_12_CAM_R0.jpg) | ![](image_v5/yield_vru_12_CAM_B0.jpg) |

**生成的 reasoning:**

> I see a pedestrian 13 meters ahead of me. Because I need to yield to them, I should remain stopped: <PLAN>STOP,STRAIGHT</PLAN>

### 13. `STOP,STRAIGHT`　v0=0.3 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.06.14.16.48.02_veh-12_03091_03461 · token `e5d75b108e545346`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/yield_vru_13_CAM_F0.jpg) | ![](image_v5/yield_vru_13_CAM_L0.jpg) | ![](image_v5/yield_vru_13_CAM_R0.jpg) | ![](image_v5/yield_vru_13_CAM_B0.jpg) |

**生成的 reasoning:**

> I see pedestrians 5 meters ahead of me. Because I need to yield to them, I should remain stopped: <PLAN>STOP,STRAIGHT</PLAN>

### 14. `STOP,STRAIGHT`　v0=3.8 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.06.14.13.28.41_veh-12_01313_01541 · token `94ac86884e5e5009`</sub>

```
q1/q2 最近 3 个对象（共 80）
    Vehicle    side right    (  -4.9, -10.3) m    0.0 m/s
    Vehicle    rear right    ( -11.6, -10.7) m    0.0 m/s
    Pedestrian front right   (  11.8, -10.7) m    0.1 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/yield_vru_14_CAM_F0.jpg) | ![](image_v5/yield_vru_14_CAM_L0.jpg) | ![](image_v5/yield_vru_14_CAM_R0.jpg) | ![](image_v5/yield_vru_14_CAM_B0.jpg) |

**生成的 reasoning:**

> I see pedestrians 17 meters ahead of me. Because I need to yield to them, I should come to a stop: <PLAN>STOP,STRAIGHT</PLAN>

### 15. `STOP,STRAIGHT`　v0=2.5 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.06.09.12.51.31_veh-35_05024_05275 · token `b8e801e741f354ec`</sub>

```
q1/q2 最近 3 个对象（共 68）
    Pedestrian front center  (  14.5,  -1.5) m    1.4 m/s
    Pedestrian front center  (  14.9,  -1.8) m    1.5 m/s
    Pedestrian front left    (  13.8,   6.1) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/yield_vru_15_CAM_F0.jpg) | ![](image_v5/yield_vru_15_CAM_L0.jpg) | ![](image_v5/yield_vru_15_CAM_R0.jpg) | ![](image_v5/yield_vru_15_CAM_B0.jpg) |

**生成的 reasoning:**

> I see pedestrians 14 meters ahead of me. Because I need to yield to them, I should come to a stop: <PLAN>STOP,STRAIGHT</PLAN>

---

## 场景:`cross_path` —— 切入 —— 从我前方穿过中线，或并入我车道　(全量占比 1.0%)

### 1. `STOP,STRAIGHT`　v0=5.3 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.08.17.16.57.11_veh-08_01200_01636 · token `9b35623cc4f05352`</sub>

```
q1/q2 最近 3 个对象（共 5）
    Vehicle    rear left     ( -16.6,   3.3) m   10.3 m/s
    Vehicle    rear right    ( -19.5,  -0.2) m    5.5 m/s
    Vehicle    front center  (  30.2,   1.2) m    9.2 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/cross_path_1_CAM_F0.jpg) | ![](image_v5/cross_path_1_CAM_L0.jpg) | ![](image_v5/cross_path_1_CAM_R0.jpg) | ![](image_v5/cross_path_1_CAM_B0.jpg) |

**生成的 reasoning:**

> Because the vehicle on my center will move into my path, I should come to a stop: <PLAN>STOP,STRAIGHT</PLAN>

### 2. `DECELERATE,LEFT`　v0=5.0 m/s　nav=LEFT　灯=无记录
<sub>2021.07.16.00.51.05_veh-17_01938_03243 · token `1017a35dfa815362`</sub>

```
q1/q2 最近 3 个对象（共 32）
    Pedestrian side left     (  -3.6,   6.6) m    0.6 m/s
    Vehicle    side left     (  -1.1,   7.6) m    0.0 m/s
    Vehicle    side right    (  -1.2,  -7.7) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/cross_path_2_CAM_F0.jpg) | ![](image_v5/cross_path_2_CAM_L0.jpg) | ![](image_v5/cross_path_2_CAM_R0.jpg) | ![](image_v5/cross_path_2_CAM_B0.jpg) |

**生成的 reasoning:**

> Because the pedestrian on my left will move into my path, I should slow down and bear left: <PLAN>DECELERATE,LEFT</PLAN>

### 3. `STOP,STRAIGHT`　v0=5.8 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.08.09.18.37.41_veh-28_00053_00548 · token `3a0916b93da7551b`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/cross_path_3_CAM_F0.jpg) | ![](image_v5/cross_path_3_CAM_L0.jpg) | ![](image_v5/cross_path_3_CAM_R0.jpg) | ![](image_v5/cross_path_3_CAM_B0.jpg) |

**生成的 reasoning:**

> Because the pedestrian on my right will move into my path, I should come to a stop: <PLAN>STOP,STRAIGHT</PLAN>

### 4. `STOP,STRAIGHT`　v0=4.6 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.07.16.16.01.30_veh-38_03893_05253 · token `44bde6a7387f5120`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/cross_path_4_CAM_F0.jpg) | ![](image_v5/cross_path_4_CAM_L0.jpg) | ![](image_v5/cross_path_4_CAM_R0.jpg) | ![](image_v5/cross_path_4_CAM_B0.jpg) |

**生成的 reasoning:**

> Because the vehicle on my left will move into my path, I should come to a stop: <PLAN>STOP,STRAIGHT</PLAN>

### 5. `STOP,LEFT`　v0=2.8 m/s　nav=LEFT　灯=无记录
<sub>2021.07.09.23.23.48_veh-26_04648_06327 · token `b21601ee8cac5427`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/cross_path_5_CAM_F0.jpg) | ![](image_v5/cross_path_5_CAM_L0.jpg) | ![](image_v5/cross_path_5_CAM_R0.jpg) | ![](image_v5/cross_path_5_CAM_B0.jpg) |

**生成的 reasoning:**

> Because the pedestrian on my center will move into my path, I should come to a stop and bear left: <PLAN>STOP,LEFT</PLAN>

### 6. `DECELERATE,STRAIGHT`　v0=6.0 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.06.14.19.22.11_veh-38_01134_01389 · token `ddd3e5e129915ed9`</sub>

```
q1/q2 最近 3 个对象（共 5）
    Vehicle    rear left     ( -13.1,   3.4) m    7.8 m/s
    Pedestrian front left    (  25.4,   7.7) m    1.3 m/s
    Vehicle    rear right    ( -26.5,  -4.0) m    5.8 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/cross_path_6_CAM_F0.jpg) | ![](image_v5/cross_path_6_CAM_L0.jpg) | ![](image_v5/cross_path_6_CAM_R0.jpg) | ![](image_v5/cross_path_6_CAM_B0.jpg) |

**生成的 reasoning:**

> Because the pedestrian on my left will move into my path, I should slow down: <PLAN>DECELERATE,STRAIGHT</PLAN>

### 7. `STOP,STRAIGHT`　v0=1.5 m/s　nav=RIGHT　灯=无记录
<sub>2021.09.15.15.34.53_veh-28_00512_01084 · token `7af92d9b54845f44`</sub>

```
q1/q2 最近 3 个对象（共 31）
    Vehicle    side left     (  -0.3,   3.8) m    0.0 m/s
    Vehicle    rear left     (  -5.4,   0.4) m    0.0 m/s
    Vehicle    rear left     (  -7.6,   3.6) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/cross_path_7_CAM_F0.jpg) | ![](image_v5/cross_path_7_CAM_L0.jpg) | ![](image_v5/cross_path_7_CAM_R0.jpg) | ![](image_v5/cross_path_7_CAM_B0.jpg) |

**生成的 reasoning:**

> Because the pedestrian on my left will move into my path, I should remain stopped: <PLAN>STOP,STRAIGHT</PLAN>

### 8. `STOP,STRAIGHT`　v0=6.1 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.06.09.17.23.18_veh-38_00131_00294 · token `19c1fba8fe7d59d1`</sub>

```
q1/q2 最近 3 个对象（共 15）
    Vehicle    side left     (   1.0,  10.9) m    0.0 m/s
    Vehicle    rear left     (  -7.4,  11.0) m    0.0 m/s
    Vehicle    front left    (   8.9,  10.7) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/cross_path_8_CAM_F0.jpg) | ![](image_v5/cross_path_8_CAM_L0.jpg) | ![](image_v5/cross_path_8_CAM_R0.jpg) | ![](image_v5/cross_path_8_CAM_B0.jpg) |

**生成的 reasoning:**

> Because the vehicle on my left will move into my path, I should come to a stop: <PLAN>STOP,STRAIGHT</PLAN>

### 9. `STOP,STRAIGHT`　v0=4.1 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.06.09.17.23.18_veh-38_03425_04047 · token `380bec175f1e5e9f`</sub>

```
q1/q2 最近 3 个对象（共 18）
    Vehicle    rear left     ( -20.9,  11.8) m    1.2 m/s
    Pedestrian front left    (  25.4,   8.6) m    1.4 m/s
    Pedestrian front left    (  26.0,   7.4) m    1.4 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/cross_path_9_CAM_F0.jpg) | ![](image_v5/cross_path_9_CAM_L0.jpg) | ![](image_v5/cross_path_9_CAM_R0.jpg) | ![](image_v5/cross_path_9_CAM_B0.jpg) |

**生成的 reasoning:**

> Because the pedestrian on my left will move into my path, I should come to a stop: <PLAN>STOP,STRAIGHT</PLAN>

### 10. `STOP,LEFT`　v0=3.1 m/s　nav=LEFT　灯=无记录
<sub>2021.07.16.20.45.29_veh-35_00016_00589 · token `45d30ee25c515310`</sub>

```
q1/q2 最近 3 个对象（共 22）
    Vehicle    side left     (   4.1,  10.7) m    1.1 m/s
    Vehicle    front left    (   5.1,  16.5) m    0.0 m/s
    Pedestrian front left    (  17.1,   4.8) m    0.3 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/cross_path_10_CAM_F0.jpg) | ![](image_v5/cross_path_10_CAM_L0.jpg) | ![](image_v5/cross_path_10_CAM_R0.jpg) | ![](image_v5/cross_path_10_CAM_B0.jpg) |

**生成的 reasoning:**

> Because the pedestrian on my left will move into my path, I should come to a stop and bear left: <PLAN>STOP,LEFT</PLAN>

### 11. `DECELERATE,LEFT`　v0=3.7 m/s　nav=LEFT　灯=无记录
<sub>2021.06.14.11.44.56_veh-35_00059_00410 · token `8b503df1f81958b1`</sub>

```
q1/q2 最近 3 个对象（共 17）
    Vehicle    side right    (   1.5, -11.5) m    0.0 m/s
    Vehicle    side left     (  -3.6,  11.7) m    0.0 m/s
    Pedestrian side right    (  -3.3, -12.2) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/cross_path_11_CAM_F0.jpg) | ![](image_v5/cross_path_11_CAM_L0.jpg) | ![](image_v5/cross_path_11_CAM_R0.jpg) | ![](image_v5/cross_path_11_CAM_B0.jpg) |

**生成的 reasoning:**

> Because the pedestrian on my center will move into my path, I should slow down and bear left: <PLAN>DECELERATE,LEFT</PLAN>

### 12. `DECELERATE,STRAIGHT`　v0=7.3 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.06.23.17.31.36_veh-16_00634_01421 · token `34d94cf580135db5`</sub>

```
q1/q2 最近 3 个对象（共 3）
    Pedestrian rear left     ( -13.5,  34.1) m    1.3 m/s
    Pedestrian rear left     ( -14.5,  34.9) m    1.2 m/s
    Vehicle    front left    (  33.5,  19.4) m    6.7 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/cross_path_12_CAM_F0.jpg) | ![](image_v5/cross_path_12_CAM_L0.jpg) | ![](image_v5/cross_path_12_CAM_R0.jpg) | ![](image_v5/cross_path_12_CAM_B0.jpg) |

**生成的 reasoning:**

> Because the vehicle on my left will move into my path, I should slow down: <PLAN>DECELERATE,STRAIGHT</PLAN>

### 13. `DECELERATE,STRAIGHT`　v0=5.8 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.08.31.16.37.21_veh-40_00798_00955 · token `f38a2b8db76a5d26`</sub>

```
q1/q2 最近 3 个对象（共 11）
    Vehicle    side left     (  -1.2,   4.0) m    0.0 m/s
    Vehicle    side right    (  -0.5,  -4.6) m    0.0 m/s
    Pedestrian front left    (   7.4,   6.4) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/cross_path_13_CAM_F0.jpg) | ![](image_v5/cross_path_13_CAM_L0.jpg) | ![](image_v5/cross_path_13_CAM_R0.jpg) | ![](image_v5/cross_path_13_CAM_B0.jpg) |

**生成的 reasoning:**

> Because the vehicle on my center will move into my path, I should slow down: <PLAN>DECELERATE,STRAIGHT</PLAN>

### 14. `DECELERATE,STRAIGHT`　v0=5.5 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.07.16.18.49.56_veh-26_00833_03384 · token `450853d9122b589d`</sub>

```
q1/q2 最近 3 个对象（共 90）
    Vehicle    side right    (  -0.6, -10.4) m    0.0 m/s
    Vehicle    side right    (   4.1, -10.6) m    0.0 m/s
    Vehicle    side right    (  -4.6, -10.5) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/cross_path_14_CAM_F0.jpg) | ![](image_v5/cross_path_14_CAM_L0.jpg) | ![](image_v5/cross_path_14_CAM_R0.jpg) | ![](image_v5/cross_path_14_CAM_B0.jpg) |

**生成的 reasoning:**

> Because the pedestrian on my right will move into my path, I should slow down: <PLAN>DECELERATE,STRAIGHT</PLAN>

### 15. `DECELERATE,RIGHT`　v0=7.9 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.09.15.14.00.15_veh-28_00420_00578 · token `01c48ef7b0d8510f`</sub>

```
q1/q2 最近 3 个对象（共 17）
    Vehicle    side right    (   1.5,  -3.9) m    0.0 m/s
    Vehicle    side left     (  -2.1,   5.9) m    0.0 m/s
    Vehicle    side left     (   2.6,   5.8) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/cross_path_15_CAM_F0.jpg) | ![](image_v5/cross_path_15_CAM_L0.jpg) | ![](image_v5/cross_path_15_CAM_R0.jpg) | ![](image_v5/cross_path_15_CAM_B0.jpg) |

**生成的 reasoning:**

> Because the vehicle on my center will move into my path, I should slow down and bear right: <PLAN>DECELERATE,RIGHT</PLAN>

---

## 场景:`stop_sign` —— 停车让行标志 —— 正前 8~40m 有 STOP_SIGN　(全量占比 1.0%)

### 1. `STOP,STRAIGHT`　v0=4.3 m/s　nav=STRAIGHT　灯=无记录　stop_sign=18m
<sub>2021.07.16.16.27.22_veh-26_01536_02260 · token `0105a875bb32558c`</sub>

```
q1/q2 最近 3 个对象（共 8）
    Vehicle    side right    (  -1.8, -13.7) m    0.0 m/s
    Vehicle    side right    (  -4.7, -13.3) m    0.0 m/s
    Vehicle    front right   (   6.2, -13.8) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/stop_sign_1_CAM_F0.jpg) | ![](image_v5/stop_sign_1_CAM_L0.jpg) | ![](image_v5/stop_sign_1_CAM_R0.jpg) | ![](image_v5/stop_sign_1_CAM_B0.jpg) |

**生成的 reasoning:**

> Because there is a stop sign ahead, I should come to a stop: <PLAN>STOP,STRAIGHT</PLAN>

### 2. `DECELERATE,STRAIGHT`　v0=4.3 m/s　nav=STRAIGHT　灯=无记录　stop_sign=21m
<sub>2021.06.09.14.50.36_veh-26_02495_02669 · token `3eaf3473ae6d5e79`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/stop_sign_2_CAM_F0.jpg) | ![](image_v5/stop_sign_2_CAM_L0.jpg) | ![](image_v5/stop_sign_2_CAM_R0.jpg) | ![](image_v5/stop_sign_2_CAM_B0.jpg) |

**生成的 reasoning:**

> Because there is a stop sign ahead, I should slow down: <PLAN>DECELERATE,STRAIGHT</PLAN>

### 3. `STOP,STRAIGHT`　v0=4.1 m/s　nav=STRAIGHT　灯=无记录　stop_sign=11m
<sub>2021.06.09.11.54.15_veh-12_00689_01229 · token `ed54dcca822c50e6`</sub>

```
q1/q2 最近 3 个对象（共 10）
    Pedestrian rear right    (  -9.9, -12.7) m    0.0 m/s
    Pedestrian rear right    ( -11.0, -12.4) m    0.0 m/s
    Vehicle    rear left     ( -23.9,   9.1) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/stop_sign_3_CAM_F0.jpg) | ![](image_v5/stop_sign_3_CAM_L0.jpg) | ![](image_v5/stop_sign_3_CAM_R0.jpg) | ![](image_v5/stop_sign_3_CAM_B0.jpg) |

**生成的 reasoning:**

> A vehicle 37 meters on my right will move into my path. Because there is a stop sign ahead, I should come to a stop: <PLAN>STOP,STRAIGHT</PLAN>

### 4. `DECELERATE,STRAIGHT`　v0=4.9 m/s　nav=STRAIGHT　灯=无记录　stop_sign=21m
<sub>2021.07.16.16.27.22_veh-26_00016_01515 · token `bfd8c06703925eb1`</sub>

```
q1/q2 最近 3 个对象（共 10）
    Vehicle    side right    (  -1.5, -13.2) m    0.0 m/s
    Vehicle    side right    (   1.4, -13.6) m    0.0 m/s
    Vehicle    front left    (  13.1,   4.8) m    8.4 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/stop_sign_4_CAM_F0.jpg) | ![](image_v5/stop_sign_4_CAM_L0.jpg) | ![](image_v5/stop_sign_4_CAM_R0.jpg) | ![](image_v5/stop_sign_4_CAM_B0.jpg) |

**生成的 reasoning:**

> Because there is a stop sign ahead, I should slow down: <PLAN>DECELERATE,STRAIGHT</PLAN>

### 5. `STOP,STRAIGHT`　v0=2.9 m/s　nav=RIGHT　灯=无记录　stop_sign=29m
<sub>2021.06.09.14.50.36_veh-26_03874_04112 · token `67b4315c0ca95e3c`</sub>

```
q1/q2 最近 3 个对象（共 22）
    Pedestrian side right    (   3.4,  -3.8) m    1.2 m/s
    Pedestrian side right    (  -3.2,  -4.4) m    0.7 m/s
    Cyclist    side right    (  -3.2,  -4.5) m    0.7 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/stop_sign_5_CAM_F0.jpg) | ![](image_v5/stop_sign_5_CAM_L0.jpg) | ![](image_v5/stop_sign_5_CAM_R0.jpg) | ![](image_v5/stop_sign_5_CAM_B0.jpg) |

**生成的 reasoning:**

> A vehicle 7 meters on my left will move into my path. Because there is a stop sign ahead, I should come to a stop: <PLAN>STOP,STRAIGHT</PLAN>

### 6. `STOP,STRAIGHT`　v0=3.2 m/s　nav=STRAIGHT　灯=无记录　stop_sign=18m
<sub>2021.07.09.17.06.37_veh-35_05026_05593 · token `a55eb33f0d6756e7`</sub>

```
q1/q2 最近 3 个对象（共 18）
    Vehicle    side right    (   2.9,  -6.4) m    4.0 m/s
    Pedestrian side left     (   2.0,  10.4) m    1.3 m/s
    Vehicle    rear right    (  -5.2, -13.0) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/stop_sign_6_CAM_F0.jpg) | ![](image_v5/stop_sign_6_CAM_L0.jpg) | ![](image_v5/stop_sign_6_CAM_R0.jpg) | ![](image_v5/stop_sign_6_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 39 meters ahead in my lane is moving at 9 m/s. Because there is a stop sign ahead, I should come to a stop: <PLAN>STOP,STRAIGHT</PLAN>

### 7. `DECELERATE,STRAIGHT`　v0=5.5 m/s　nav=STRAIGHT　灯=无记录　stop_sign=23m
<sub>2021.06.09.14.03.17_veh-12_04129_04237 · token `d243f570f1615426`</sub>

```
q1/q2 最近 3 个对象（共 10）
    Vehicle    side right    (  -0.7,  -5.7) m    7.7 m/s
    Vehicle    rear right    ( -11.1, -12.8) m    0.0 m/s
    Pedestrian front right   (  21.1,  -8.3) m    1.1 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/stop_sign_7_CAM_F0.jpg) | ![](image_v5/stop_sign_7_CAM_L0.jpg) | ![](image_v5/stop_sign_7_CAM_R0.jpg) | ![](image_v5/stop_sign_7_CAM_B0.jpg) |

**生成的 reasoning:**

> Because there is a stop sign ahead, I should slow down: <PLAN>DECELERATE,STRAIGHT</PLAN>

### 8. `STOP,STRAIGHT`　v0=2.4 m/s　nav=STRAIGHT　灯=无记录　stop_sign=10m
<sub>2021.07.16.01.22.41_veh-14_04315_07102 · token `151db456a92b55bc`</sub>

```
q1/q2 最近 3 个对象（共 21）
    Vehicle    front left    (  10.3,   4.2) m    0.6 m/s
    Vehicle    front center  (  11.5,   1.1) m    1.0 m/s
    Pedestrian side right    (  -4.9, -11.4) m    0.9 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/stop_sign_8_CAM_F0.jpg) | ![](image_v5/stop_sign_8_CAM_L0.jpg) | ![](image_v5/stop_sign_8_CAM_R0.jpg) | ![](image_v5/stop_sign_8_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 11 meters ahead in my lane is moving slowly. Because there is a stop sign ahead, I should come to a stop: <PLAN>STOP,STRAIGHT</PLAN>

### 9. `STOP,STRAIGHT`　v0=3.3 m/s　nav=RIGHT　灯=无记录　stop_sign=28m
<sub>2021.06.14.13.28.41_veh-12_00906_01063 · token `f1994af0bd595b7f`</sub>

```
q1/q2 最近 3 个对象（共 18）
    Vehicle    side right    (  -1.5,  -3.0) m    0.0 m/s
    Vehicle    rear right    (  -7.1,  -4.0) m    0.0 m/s
    Vehicle    rear left     (  -9.8,   9.6) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/stop_sign_9_CAM_F0.jpg) | ![](image_v5/stop_sign_9_CAM_L0.jpg) | ![](image_v5/stop_sign_9_CAM_R0.jpg) | ![](image_v5/stop_sign_9_CAM_B0.jpg) |

**生成的 reasoning:**

> A vehicle 7 meters on my left will move into my path. Because there is a stop sign ahead, I should come to a stop: <PLAN>STOP,STRAIGHT</PLAN>

### 10. `DECELERATE,LEFT`　v0=5.0 m/s　nav=LEFT　灯=无记录　stop_sign=20m
<sub>2021.10.05.06.31.40_veh-52_01598_02013 · token `9d256c861ff35812`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/stop_sign_10_CAM_F0.jpg) | ![](image_v5/stop_sign_10_CAM_L0.jpg) | ![](image_v5/stop_sign_10_CAM_R0.jpg) | ![](image_v5/stop_sign_10_CAM_B0.jpg) |

**生成的 reasoning:**

> Because there is a stop sign ahead, I should slow down and bear left: <PLAN>DECELERATE,LEFT</PLAN>

### 11. `DECELERATE,STRAIGHT`　v0=6.4 m/s　nav=STRAIGHT　灯=无记录　stop_sign=29m
<sub>2021.10.06.13.21.47_veh-28_01127_01187 · token `b1dc0e044db4545d`</sub>

```
q1/q2 最近 3 个对象（共 10）
    Vehicle    front left    (   7.7,   6.1) m    0.0 m/s
    Vehicle    front left    (  13.0,   6.3) m    0.0 m/s
    Vehicle    front left    (  18.7,   6.1) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/stop_sign_11_CAM_F0.jpg) | ![](image_v5/stop_sign_11_CAM_L0.jpg) | ![](image_v5/stop_sign_11_CAM_R0.jpg) | ![](image_v5/stop_sign_11_CAM_B0.jpg) |

**生成的 reasoning:**

> Because there is a stop sign ahead, I should slow down: <PLAN>DECELERATE,STRAIGHT</PLAN>

### 12. `STOP,STRAIGHT`　v0=3.4 m/s　nav=STRAIGHT　灯=无记录　stop_sign=16m
<sub>2021.06.14.14.03.45_veh-38_04398_04488 · token `c6d62854cb885bc6`</sub>

```
q1/q2 最近 3 个对象（共 8）
    Vehicle    side right    (  -2.7,  -6.5) m    5.9 m/s
    Vehicle    rear left     ( -13.1,   7.7) m    3.9 m/s
    Vehicle    front right   (  21.2,  -7.5) m    3.7 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/stop_sign_12_CAM_F0.jpg) | ![](image_v5/stop_sign_12_CAM_L0.jpg) | ![](image_v5/stop_sign_12_CAM_R0.jpg) | ![](image_v5/stop_sign_12_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 22 meters ahead in my lane is moving at 4 m/s. Because there is a stop sign ahead, I should come to a stop: <PLAN>STOP,STRAIGHT</PLAN>

### 13. `DECELERATE,STRAIGHT`　v0=6.2 m/s　nav=STRAIGHT　灯=无记录　stop_sign=20m
<sub>2021.10.06.14.31.13_veh-28_00981_01226 · token `9144a4b381ea591e`</sub>

```
q1/q2 最近 3 个对象（共 15）
    Vehicle    side right    (   3.2,  -6.0) m    0.0 m/s
    Vehicle    side right    (  -4.5,  -6.8) m    0.0 m/s
    Vehicle    front right   (   5.9,  -6.2) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/stop_sign_13_CAM_F0.jpg) | ![](image_v5/stop_sign_13_CAM_L0.jpg) | ![](image_v5/stop_sign_13_CAM_R0.jpg) | ![](image_v5/stop_sign_13_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 26 meters ahead in my lane is stopped. Because there is a stop sign ahead, I should slow down: <PLAN>DECELERATE,STRAIGHT</PLAN>

### 14. `STOP,STRAIGHT`　v0=3.0 m/s　nav=STRAIGHT　灯=无记录　stop_sign=10m
<sub>2021.06.14.16.32.09_veh-35_03231_03426 · token `da2c0ee139fd5acb`</sub>

```
q1/q2 最近 3 个对象（共 18）
    Vehicle    rear left     ( -10.1,   1.2) m    3.5 m/s
    Vehicle    front center  (  14.5,   1.1) m    3.6 m/s
    Pedestrian rear left     (  -8.8,  16.2) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/stop_sign_14_CAM_F0.jpg) | ![](image_v5/stop_sign_14_CAM_L0.jpg) | ![](image_v5/stop_sign_14_CAM_R0.jpg) | ![](image_v5/stop_sign_14_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 14 meters ahead in my lane is moving at 3 m/s. Because there is a stop sign ahead, I should come to a stop: <PLAN>STOP,STRAIGHT</PLAN>

### 15. `DECELERATE,STRAIGHT`　v0=6.7 m/s　nav=STRAIGHT　灯=无记录　stop_sign=23m
<sub>2021.10.06.14.31.13_veh-28_00981_01226 · token `7ef6efc8eab85155`</sub>

```
q1/q2 最近 3 个对象（共 15）
    Vehicle    side right    (  -1.1,  -6.8) m    0.0 m/s
    Vehicle    front right   (   6.4,  -6.1) m    0.0 m/s
    Vehicle    front right   (   9.2,  -6.1) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/stop_sign_15_CAM_F0.jpg) | ![](image_v5/stop_sign_15_CAM_L0.jpg) | ![](image_v5/stop_sign_15_CAM_R0.jpg) | ![](image_v5/stop_sign_15_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 29 meters ahead in my lane is stopped. Because there is a stop sign ahead, I should slow down: <PLAN>DECELERATE,STRAIGHT</PLAN>

---

## 场景:`gap_big` —— 前方放空 —— 前车在加速且已拉开　(全量占比 0.4%)

### 1. `KEEP,STRAIGHT`　v0=4.5 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.06.09.17.23.18_veh-38_04782_05228 · token `709abfb23c9950f4`</sub>

```
q1/q2 最近 3 个对象（共 13）
    Vehicle    rear left     ( -14.2,   9.9) m    2.5 m/s
    Vehicle    front left    (   9.3,  18.9) m    0.0 m/s
    Vehicle    front left    (  20.9,   5.8) m    8.5 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/gap_big_1_CAM_F0.jpg) | ![](image_v5/gap_big_1_CAM_L0.jpg) | ![](image_v5/gap_big_1_CAM_R0.jpg) | ![](image_v5/gap_big_1_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 28 meters ahead in my lane is moving at 3 m/s. Because the gap in front of me is opening up, I should hold my current speed: <PLAN>KEEP,STRAIGHT</PLAN>

### 2. `ACCELERATE,STRAIGHT`　v0=0.2 m/s　nav=STRAIGHT　灯=RED
<sub>2021.09.15.11.49.23_veh-28_00767_00955 · token `5e419707e2ef5f68`</sub>

```
q1/q2 最近 3 个对象（共 21）
    Vehicle    side left     (  -3.4,   4.9) m   14.0 m/s
    Vehicle    rear right    (  -6.3,  -0.2) m    0.5 m/s
    Pedestrian side right    (  -3.4, -13.7) m    1.3 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/gap_big_2_CAM_F0.jpg) | ![](image_v5/gap_big_2_CAM_L0.jpg) | ![](image_v5/gap_big_2_CAM_R0.jpg) | ![](image_v5/gap_big_2_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 26 meters ahead in my lane is stopped. Because the gap in front of me is opening up, I should speed up: <PLAN>ACCELERATE,STRAIGHT</PLAN>

### 3. `KEEP,STRAIGHT`　v0=11.1 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.07.16.16.08.35_veh-35_04744_06051 · token `23eb8229a2e256e9`</sub>

```
q1/q2 最近 3 个对象（共 40）
    Vehicle    front left    (   5.1,   4.2) m    3.7 m/s
    Pedestrian side right    (   1.5, -11.6) m    1.3 m/s
    Pedestrian side right    (   2.0, -12.2) m    1.3 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/gap_big_3_CAM_F0.jpg) | ![](image_v5/gap_big_3_CAM_L0.jpg) | ![](image_v5/gap_big_3_CAM_R0.jpg) | ![](image_v5/gap_big_3_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 37 meters ahead in my lane is moving at 7 m/s. Because the gap in front of me is opening up, I should hold my current speed: <PLAN>KEEP,STRAIGHT</PLAN>

### 4. `ACCELERATE,LEFT`　v0=1.1 m/s　nav=LEFT　灯=无记录
<sub>2021.06.09.12.39.51_veh-26_00609_01168 · token `4d41bbb1ab1b5d42`</sub>

```
q1/q2 最近 3 个对象（共 15）
    Vehicle    rear right    (  -5.7,  -3.3) m    2.5 m/s
    Vehicle    front left    (   5.7,   4.6) m    7.3 m/s
    Vehicle    front right   (   5.4,  -6.8) m    4.5 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/gap_big_4_CAM_F0.jpg) | ![](image_v5/gap_big_4_CAM_L0.jpg) | ![](image_v5/gap_big_4_CAM_R0.jpg) | ![](image_v5/gap_big_4_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 31 meters ahead in my lane is moving at 8 m/s. Because the gap in front of me is opening up, I should speed up and bear left: <PLAN>ACCELERATE,LEFT</PLAN>

### 5. `KEEP,STRAIGHT`　v0=11.0 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.07.16.16.08.35_veh-35_04744_06051 · token `e477167805585323`</sub>

```
q1/q2 最近 3 个对象（共 40）
    Vehicle    side left     (   1.4,   4.4) m    3.4 m/s
    Pedestrian side right    (   1.9, -11.1) m    1.0 m/s
    Pedestrian side right    (   1.7, -11.7) m    1.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/gap_big_5_CAM_F0.jpg) | ![](image_v5/gap_big_5_CAM_L0.jpg) | ![](image_v5/gap_big_5_CAM_R0.jpg) | ![](image_v5/gap_big_5_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 35 meters ahead in my lane is moving at 7 m/s. Because the gap in front of me is opening up, I should hold my current speed: <PLAN>KEEP,STRAIGHT</PLAN>

### 6. `ACCELERATE,LEFT`　v0=7.4 m/s　nav=LEFT　灯=GREEN
<sub>2021.07.16.16.08.35_veh-35_03711_04709 · token `2bbd97b0c6015fd3`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/gap_big_6_CAM_F0.jpg) | ![](image_v5/gap_big_6_CAM_L0.jpg) | ![](image_v5/gap_big_6_CAM_R0.jpg) | ![](image_v5/gap_big_6_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 33 meters ahead in my lane is moving at 11 m/s. Because the gap in front of me is opening up, I should speed up and bear left: <PLAN>ACCELERATE,LEFT</PLAN>

### 7. `KEEP,STRAIGHT`　v0=8.3 m/s　nav=STRAIGHT　灯=GREEN
<sub>2021.06.23.15.56.12_veh-16_00839_01285 · token `a02242d31dfe5abf`</sub>

```
q1/q2 最近 3 个对象（共 16）
    Vehicle    side left     (   4.5,   3.0) m   11.2 m/s
    Vehicle    side right    (   3.5, -11.8) m    0.0 m/s
    Vehicle    side right    (  -0.1, -12.7) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/gap_big_7_CAM_F0.jpg) | ![](image_v5/gap_big_7_CAM_L0.jpg) | ![](image_v5/gap_big_7_CAM_R0.jpg) | ![](image_v5/gap_big_7_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 27 meters ahead in my lane is moving at 10 m/s. Because the gap in front of me is opening up, I should hold my current speed: <PLAN>KEEP,STRAIGHT</PLAN>

### 8. `ACCELERATE,STRAIGHT`　v0=0.4 m/s　nav=LEFT　灯=无记录
<sub>2021.08.17.17.17.01_veh-45_00207_00594 · token `f24459501ee95cf1`</sub>

```
q1/q2 最近 3 个对象（共 11）
    Vehicle    rear left     ( -12.4,   6.5) m    0.0 m/s
    Vehicle    front right   (  19.8,  -9.3) m    0.0 m/s
    Vehicle    front right   (  19.7, -13.2) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/gap_big_8_CAM_F0.jpg) | ![](image_v5/gap_big_8_CAM_L0.jpg) | ![](image_v5/gap_big_8_CAM_R0.jpg) | ![](image_v5/gap_big_8_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 38 meters ahead in my lane is moving at 4 m/s. Because the gap in front of me is opening up, I should speed up: <PLAN>ACCELERATE,STRAIGHT</PLAN>

### 9. `ACCELERATE,STRAIGHT`　v0=6.2 m/s　nav=STRAIGHT　灯=RED
<sub>2021.10.11.05.34.05_veh-50_00020_00149 · token `7a6ac7ff378b520a`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/gap_big_9_CAM_F0.jpg) | ![](image_v5/gap_big_9_CAM_L0.jpg) | ![](image_v5/gap_big_9_CAM_R0.jpg) | ![](image_v5/gap_big_9_CAM_B0.jpg) |

**生成的 reasoning:**

> The cyclist 33 meters ahead in my lane is moving at 10 m/s. Because the gap in front of me is opening up, I should speed up: <PLAN>ACCELERATE,STRAIGHT</PLAN>

### 10. `ACCELERATE,STRAIGHT`　v0=0.0 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.06.09.12.51.31_veh-35_01729_02626 · token `59a1bb2069d057ed`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/gap_big_10_CAM_F0.jpg) | ![](image_v5/gap_big_10_CAM_L0.jpg) | ![](image_v5/gap_big_10_CAM_R0.jpg) | ![](image_v5/gap_big_10_CAM_B0.jpg) |

**生成的 reasoning:**

> I see a pedestrian 10 meters ahead of me. The vehicle 28 meters ahead in my lane is moving at 3 m/s. Because the gap in front of me is opening up, I should speed up: <PLAN>ACCELERATE,STRAIGHT</PLAN>

### 11. `ACCELERATE,LEFT`　v0=2.4 m/s　nav=LEFT　灯=无记录
<sub>2021.08.24.14.25.28_veh-42_00333_00472 · token `197c947e49005343`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/gap_big_11_CAM_F0.jpg) | ![](image_v5/gap_big_11_CAM_L0.jpg) | ![](image_v5/gap_big_11_CAM_R0.jpg) | ![](image_v5/gap_big_11_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 32 meters ahead in my lane is moving at 13 m/s. Because the gap in front of me is opening up, I should speed up and bear left: <PLAN>ACCELERATE,LEFT</PLAN>

### 12. `KEEP,STRAIGHT`　v0=4.2 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.07.09.17.06.37_veh-35_00928_02567 · token `5d2299d94a405baf`</sub>

```
q1/q2 最近 3 个对象（共 19）
    Vehicle    side left     (  -4.5,   5.9) m    7.6 m/s
    Vehicle    front right   (   8.4,  -6.7) m    6.7 m/s
    Vehicle    side right    (   2.7, -13.1) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/gap_big_12_CAM_F0.jpg) | ![](image_v5/gap_big_12_CAM_L0.jpg) | ![](image_v5/gap_big_12_CAM_R0.jpg) | ![](image_v5/gap_big_12_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 31 meters ahead in my lane is moving at 3 m/s. Because the gap in front of me is opening up, I should hold my current speed: <PLAN>KEEP,STRAIGHT</PLAN>

### 13. `ACCELERATE,STRAIGHT`　v0=1.0 m/s　nav=STRAIGHT　灯=RED
<sub>2021.07.09.01.37.16_veh-26_01726_01793 · token `caafcdd4b7835eb0`</sub>

```
q1/q2 最近 3 个对象（共 86）
    Vehicle    rear left     (  -7.1,   0.1) m    0.0 m/s
    Pedestrian side right    (   3.9,  -6.2) m    1.5 m/s
    Pedestrian front right   (   5.9,  -5.4) m    0.1 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/gap_big_13_CAM_F0.jpg) | ![](image_v5/gap_big_13_CAM_L0.jpg) | ![](image_v5/gap_big_13_CAM_R0.jpg) | ![](image_v5/gap_big_13_CAM_B0.jpg) |

**生成的 reasoning:**

> I see pedestrians 9 meters ahead of me. The vehicle 26 meters ahead in my lane is moving slowly. Because the gap in front of me is opening up, I should speed up: <PLAN>ACCELERATE,STRAIGHT</PLAN>

### 14. `ACCELERATE,STRAIGHT`　v0=2.6 m/s　nav=STRAIGHT　灯=无记录　stop_sign=38m
<sub>2021.10.11.07.12.18_veh-50_00866_01534 · token `2ac9606d15d05f93`</sub>

```
q1/q2 最近 3 个对象（共 6）
    Vehicle    side right    (   2.0,  -3.5) m    1.6 m/s
    Pedestrian front left    (  24.0,   6.9) m    0.0 m/s
    Pedestrian front left    (  24.9,  11.3) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/gap_big_14_CAM_F0.jpg) | ![](image_v5/gap_big_14_CAM_L0.jpg) | ![](image_v5/gap_big_14_CAM_R0.jpg) | ![](image_v5/gap_big_14_CAM_B0.jpg) |

**生成的 reasoning:**

> The cyclist 31 meters ahead in my lane is moving at 9 m/s. Because the gap in front of me is opening up, I should speed up: <PLAN>ACCELERATE,STRAIGHT</PLAN>

### 15. `ACCELERATE,LEFT`　v0=7.3 m/s　nav=LEFT　灯=GREEN
<sub>2021.07.16.18.49.56_veh-26_00015_00235 · token `44cd0d7501e853a3`</sub>

```
q1/q2 最近 3 个对象（共 36）
    Vehicle    side left     (  -2.6,   4.8) m    4.9 m/s
    Vehicle    side left     (   2.9,  11.6) m    0.0 m/s
    Vehicle    rear left     ( -12.7,   2.9) m    6.1 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/gap_big_15_CAM_F0.jpg) | ![](image_v5/gap_big_15_CAM_L0.jpg) | ![](image_v5/gap_big_15_CAM_R0.jpg) | ![](image_v5/gap_big_15_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 30 meters ahead in my lane is moving at 10 m/s. Because the gap in front of me is opening up, I should speed up and bear left: <PLAN>ACCELERATE,LEFT</PLAN>

---

## 场景:`none` —— 没有理由从句　(全量占比 67.6%)

### 1. `KEEP,STRAIGHT`　v0=4.5 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.07.09.01.37.16_veh-26_03306_03373 · token `27cc20d9192052d8`</sub>

```
q1/q2 最近 3 个对象（共 12）
    Pedestrian side right    (   3.3,  -5.2) m    1.3 m/s
    Pedestrian front right   (   7.1,  -7.5) m    1.4 m/s
    Vehicle    rear left     ( -11.4,  17.1) m    2.5 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/none_1_CAM_F0.jpg) | ![](image_v5/none_1_CAM_L0.jpg) | ![](image_v5/none_1_CAM_R0.jpg) | ![](image_v5/none_1_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 27 meters ahead in my lane is moving at 7 m/s. I should hold my current speed: <PLAN>KEEP,STRAIGHT</PLAN>

### 2. `ACCELERATE,RIGHT`　v0=2.6 m/s　nav=RIGHT　灯=无记录
<sub>2021.10.05.07.49.39_veh-52_00934_01406 · token `f092e48179045493`</sub>

```
q1/q2 最近 3 个对象（共 1）
    Vehicle    side right    (  -0.3, -26.9) m    1.8 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/none_2_CAM_F0.jpg) | ![](image_v5/none_2_CAM_L0.jpg) | ![](image_v5/none_2_CAM_R0.jpg) | ![](image_v5/none_2_CAM_B0.jpg) |

**生成的 reasoning:**

> I should speed up and bear right: <PLAN>ACCELERATE,RIGHT</PLAN>

### 3. `DECELERATE,STRAIGHT`　v0=4.7 m/s　nav=LEFT　灯=无记录
<sub>2021.10.11.07.12.18_veh-50_00866_01534 · token `0c1d1e46e19c5afa`</sub>

```
q1/q2 最近 3 个对象（共 1）
    Vehicle    front left    (  31.7,   8.5) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/none_3_CAM_F0.jpg) | ![](image_v5/none_3_CAM_L0.jpg) | ![](image_v5/none_3_CAM_R0.jpg) | ![](image_v5/none_3_CAM_B0.jpg) |

**生成的 reasoning:**

> I should slow down: <PLAN>DECELERATE,STRAIGHT</PLAN>

### 4. `STOP,LEFT`　v0=2.3 m/s　nav=LEFT　灯=无记录
<sub>2021.10.11.05.34.05_veh-50_01281_01692 · token `b0e29931221c5820`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/none_4_CAM_F0.jpg) | ![](image_v5/none_4_CAM_L0.jpg) | ![](image_v5/none_4_CAM_R0.jpg) | ![](image_v5/none_4_CAM_B0.jpg) |

**生成的 reasoning:**

> I should come to a stop and bear left: <PLAN>STOP,LEFT</PLAN>

### 5. `KEEP,STRAIGHT`　v0=6.6 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.10.11.05.34.05_veh-50_01718_02261 · token `463815f761ca5d0f`</sub>

```
q1/q2 最近 3 个对象（共 2）
    Vehicle    side left     (   1.3,   3.8) m    0.0 m/s
    Vehicle    front center  (  31.4,   1.8) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/none_5_CAM_F0.jpg) | ![](image_v5/none_5_CAM_L0.jpg) | ![](image_v5/none_5_CAM_R0.jpg) | ![](image_v5/none_5_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 31 meters ahead in my lane is stopped. I should hold my current speed: <PLAN>KEEP,STRAIGHT</PLAN>

### 6. `KEEP,STRAIGHT`　v0=12.2 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.06.23.14.54.32_veh-16_01187_03336 · token `954025fa67215f54`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/none_6_CAM_F0.jpg) | ![](image_v5/none_6_CAM_L0.jpg) | ![](image_v5/none_6_CAM_R0.jpg) | ![](image_v5/none_6_CAM_B0.jpg) |

**生成的 reasoning:**

> I should hold my current speed: <PLAN>KEEP,STRAIGHT</PLAN>

### 7. `KEEP,RIGHT`　v0=4.7 m/s　nav=RIGHT　灯=无记录
<sub>2021.08.31.12.21.30_veh-40_00056_00155 · token `add36b4981ec5824`</sub>

```
q1/q2 最近 3 个对象（共 1）
    Vehicle    front left    (  27.7,   5.3) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/none_7_CAM_F0.jpg) | ![](image_v5/none_7_CAM_L0.jpg) | ![](image_v5/none_7_CAM_R0.jpg) | ![](image_v5/none_7_CAM_B0.jpg) |

**生成的 reasoning:**

> I should hold my current speed and bear right: <PLAN>KEEP,RIGHT</PLAN>

### 8. `KEEP,RIGHT`　v0=2.1 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.07.09.17.06.37_veh-35_00769_00907 · token `0fbe5f75c3915b0c`</sub>

```
q1/q2 最近 3 个对象（共 18）
    Vehicle    side left     (  -1.6,   6.4) m    8.7 m/s
    Vehicle    front center  (  10.3,  -2.9) m    5.1 m/s
    Pedestrian side right    (  -2.1, -13.1) m    0.8 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/none_8_CAM_F0.jpg) | ![](image_v5/none_8_CAM_L0.jpg) | ![](image_v5/none_8_CAM_R0.jpg) | ![](image_v5/none_8_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 10 meters ahead in my lane is moving at 5 m/s. I should hold my current speed and bear right: <PLAN>KEEP,RIGHT</PLAN>

### 9. `ACCELERATE,STRAIGHT`　v0=9.9 m/s　nav=STRAIGHT　灯=GREEN
<sub>2021.07.09.15.53.28_veh-38_02316_03434 · token `2ea84245a45c5551`</sub>

```
q1/q2 最近 3 个对象（共 14）
    Vehicle    side left     (   1.1,   3.8) m    7.3 m/s
    Vehicle    side right    (   4.6,  -3.6) m    7.5 m/s
    Vehicle    rear right    (  -7.1,  -6.8) m    8.5 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/none_9_CAM_F0.jpg) | ![](image_v5/none_9_CAM_L0.jpg) | ![](image_v5/none_9_CAM_R0.jpg) | ![](image_v5/none_9_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 22 meters ahead in my lane is moving at 10 m/s. I should speed up: <PLAN>ACCELERATE,STRAIGHT</PLAN>

### 10. `KEEP,STRAIGHT`　v0=8.3 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.08.17.18.54.02_veh-45_00665_01065 · token `c0a98b2d87c65c1b`</sub>

```
q1/q2 最近 3 个对象（共 1）
    Pedestrian front left    (  24.7,   7.0) m    1.3 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/none_10_CAM_F0.jpg) | ![](image_v5/none_10_CAM_L0.jpg) | ![](image_v5/none_10_CAM_R0.jpg) | ![](image_v5/none_10_CAM_B0.jpg) |

**生成的 reasoning:**

> I should hold my current speed: <PLAN>KEEP,STRAIGHT</PLAN>

### 11. `ACCELERATE,STRAIGHT`　v0=4.3 m/s　nav=STRAIGHT　灯=GREEN
<sub>2021.06.23.21.56.29_veh-35_00220_00936 · token `212effc037ef56f3`</sub>

```
q1/q2 最近 3 个对象（共 25）
    Vehicle    rear right    (  -5.7,  -3.4) m    4.9 m/s
    Vehicle    front center  (   6.5,   3.4) m    6.6 m/s
    Vehicle    rear left     (  -8.9,   3.6) m    5.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/none_11_CAM_F0.jpg) | ![](image_v5/none_11_CAM_L0.jpg) | ![](image_v5/none_11_CAM_R0.jpg) | ![](image_v5/none_11_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 6 meters ahead in my lane is moving at 6 m/s. I should speed up: <PLAN>ACCELERATE,STRAIGHT</PLAN>

### 12. `DECELERATE,STRAIGHT`　v0=5.1 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.10.11.08.31.07_veh-50_02360_02684 · token `2fb17d18ba345719`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/none_12_CAM_F0.jpg) | ![](image_v5/none_12_CAM_L0.jpg) | ![](image_v5/none_12_CAM_R0.jpg) | ![](image_v5/none_12_CAM_B0.jpg) |

**生成的 reasoning:**

> I should slow down: <PLAN>DECELERATE,STRAIGHT</PLAN>

### 13. `DECELERATE,STRAIGHT`　v0=6.4 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.10.05.06.57.40_veh-50_01131_01452 · token `f318d1c464de5eda`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/none_13_CAM_F0.jpg) | ![](image_v5/none_13_CAM_L0.jpg) | ![](image_v5/none_13_CAM_R0.jpg) | ![](image_v5/none_13_CAM_B0.jpg) |

**生成的 reasoning:**

> I should slow down: <PLAN>DECELERATE,STRAIGHT</PLAN>

### 14. `KEEP,STRAIGHT`　v0=10.6 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.08.24.18.30.46_veh-08_02327_02583 · token `12f7648c19e45d7c`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/none_14_CAM_F0.jpg) | ![](image_v5/none_14_CAM_L0.jpg) | ![](image_v5/none_14_CAM_R0.jpg) | ![](image_v5/none_14_CAM_B0.jpg) |

**生成的 reasoning:**

> I should hold my current speed: <PLAN>KEEP,STRAIGHT</PLAN>

### 15. `ACCELERATE,STRAIGHT`　v0=10.2 m/s　nav=LEFT　灯=GREEN
<sub>2021.06.23.17.31.36_veh-16_00634_01421 · token `52e68fb3819759b6`</sub>

```
q1/q2 最近 3 个对象（共 29）
    Vehicle    front left    (  21.9,   7.6) m   12.1 m/s
    Vehicle    rear right    ( -22.2,  -9.9) m    0.0 m/s
    Pedestrian front right   (  24.1,  -6.7) m    1.1 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/none_15_CAM_F0.jpg) | ![](image_v5/none_15_CAM_L0.jpg) | ![](image_v5/none_15_CAM_R0.jpg) | ![](image_v5/none_15_CAM_B0.jpg) |

**生成的 reasoning:**

> I should speed up: <PLAN>ACCELERATE,STRAIGHT</PLAN>

### 16. `KEEP,LEFT`　v0=5.5 m/s　nav=LEFT　灯=无记录　stop_sign=17m
<sub>2021.06.23.14.54.32_veh-16_01187_03336 · token `b194973d8f0953c6`</sub>

```
q1/q2 最近 3 个对象（共 15）
    Pedestrian side right    (   2.9,  -9.9) m    1.1 m/s
    Pedestrian side right    (   3.1, -10.8) m    1.2 m/s
    Pedestrian front right   (   7.5,  -8.5) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/none_16_CAM_F0.jpg) | ![](image_v5/none_16_CAM_L0.jpg) | ![](image_v5/none_16_CAM_R0.jpg) | ![](image_v5/none_16_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 17 meters ahead in my lane is stopped. I should hold my current speed and bear left: <PLAN>KEEP,LEFT</PLAN>

### 17. `KEEP,STRAIGHT`　v0=10.7 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.08.17.13.15.12_veh-45_02124_02293 · token `6b4e81d4ed615829`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/none_17_CAM_F0.jpg) | ![](image_v5/none_17_CAM_L0.jpg) | ![](image_v5/none_17_CAM_R0.jpg) | ![](image_v5/none_17_CAM_B0.jpg) |

**生成的 reasoning:**

> I should hold my current speed: <PLAN>KEEP,STRAIGHT</PLAN>

### 18. `KEEP,LEFT`　v0=3.3 m/s　nav=LEFT　灯=无记录
<sub>2021.06.09.11.54.15_veh-12_02540_02723 · token `7ab9fb3d224354f4`</sub>

```
q1/q2 最近 3 个对象（共 21）
    Pedestrian side right    (  -1.2,  -3.4) m    0.0 m/s
    Pedestrian side right    (   1.3,  -4.1) m    0.0 m/s
    Pedestrian side right    (   2.1,  -7.3) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/none_18_CAM_F0.jpg) | ![](image_v5/none_18_CAM_L0.jpg) | ![](image_v5/none_18_CAM_R0.jpg) | ![](image_v5/none_18_CAM_B0.jpg) |

**生成的 reasoning:**

> I should hold my current speed and bear left: <PLAN>KEEP,LEFT</PLAN>

### 19. `KEEP,LEFT`　v0=4.8 m/s　nav=LEFT　灯=无记录
<sub>2021.07.16.16.08.35_veh-35_03711_04709 · token `345539b303525835`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/none_19_CAM_F0.jpg) | ![](image_v5/none_19_CAM_L0.jpg) | ![](image_v5/none_19_CAM_R0.jpg) | ![](image_v5/none_19_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 15 meters ahead in my lane is stopped. I should hold my current speed and bear left: <PLAN>KEEP,LEFT</PLAN>

### 20. `ACCELERATE,STRAIGHT`　v0=0.0 m/s　nav=RIGHT　灯=无记录
<sub>2021.08.31.13.27.52_veh-40_01615_01687 · token `154d4bca95735b49`</sub>

```
q1/q2 最近 3 个对象（共 5）
    Vehicle    front right   (  12.7,  -8.1) m    0.0 m/s
    Pedestrian front right   (  15.7,  -7.7) m    0.0 m/s
    Vehicle    front right   (  17.9, -13.4) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/none_20_CAM_F0.jpg) | ![](image_v5/none_20_CAM_L0.jpg) | ![](image_v5/none_20_CAM_R0.jpg) | ![](image_v5/none_20_CAM_B0.jpg) |

**生成的 reasoning:**

> I should speed up: <PLAN>ACCELERATE,STRAIGHT</PLAN>

### 21. `ACCELERATE,STRAIGHT`　v0=0.0 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.05.12.22.00.38_veh-35_01008_01518 · token `2e05623cb858533a`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/none_21_CAM_F0.jpg) | ![](image_v5/none_21_CAM_L0.jpg) | ![](image_v5/none_21_CAM_R0.jpg) | ![](image_v5/none_21_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 9 meters ahead in my lane is moving slowly. I should speed up: <PLAN>ACCELERATE,STRAIGHT</PLAN>

### 22. `KEEP,LEFT`　v0=4.3 m/s　nav=LEFT　灯=无记录
<sub>2021.06.14.18.33.41_veh-35_02339_02447 · token `5662fc8a0b95525f`</sub>

```
q1/q2 最近 3 个对象（共 11）
    Vehicle    front left    (   9.2,   9.8) m    0.0 m/s
    Vehicle    front left    (  11.8,   7.3) m    0.0 m/s
    Vehicle    front left    (  14.5,  13.3) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/none_22_CAM_F0.jpg) | ![](image_v5/none_22_CAM_L0.jpg) | ![](image_v5/none_22_CAM_R0.jpg) | ![](image_v5/none_22_CAM_B0.jpg) |

**生成的 reasoning:**

> A vehicle 26 meters on my left will move into my path. I should hold my current speed and bear left: <PLAN>KEEP,LEFT</PLAN>

### 23. `ACCELERATE,STRAIGHT`　v0=1.6 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.07.09.02.42.50_veh-35_00038_02629 · token `24021978a7f753b4`</sub>

```
q1/q2 最近 3 个对象（共 25）
    Vehicle    front center  (   6.2,  -3.3) m    1.4 m/s
    Vehicle    side right    (   2.3,  -6.7) m    1.2 m/s
    Vehicle    rear right    (  -7.2,  -6.9) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/none_23_CAM_F0.jpg) | ![](image_v5/none_23_CAM_L0.jpg) | ![](image_v5/none_23_CAM_R0.jpg) | ![](image_v5/none_23_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 6 meters ahead in my lane is moving slowly. I should speed up: <PLAN>ACCELERATE,STRAIGHT</PLAN>

### 24. `KEEP,STRAIGHT`　v0=11.2 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.08.17.13.15.12_veh-45_02025_02103 · token `9a7ee98b68785ab1`</sub>

```
q1/q2 最近 3 个对象（共 1）
    Vehicle    rear right    (  -9.9,  -3.6) m    0.4 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/none_24_CAM_F0.jpg) | ![](image_v5/none_24_CAM_L0.jpg) | ![](image_v5/none_24_CAM_R0.jpg) | ![](image_v5/none_24_CAM_B0.jpg) |

**生成的 reasoning:**

> I should hold my current speed: <PLAN>KEEP,STRAIGHT</PLAN>

### 25. `KEEP,STRAIGHT`　v0=7.7 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.06.23.16.54.19_veh-35_00016_00755 · token `6b87c6c041785f5d`</sub>

```
q1/q2 最近 3 个对象（共 4）
    Vehicle    front left    (   7.8,  13.9) m    8.7 m/s
    Vehicle    rear left     ( -16.7,  11.9) m    6.2 m/s
    Vehicle    front center  (  22.7,   1.8) m    7.4 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/none_25_CAM_F0.jpg) | ![](image_v5/none_25_CAM_L0.jpg) | ![](image_v5/none_25_CAM_R0.jpg) | ![](image_v5/none_25_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 22 meters ahead in my lane is moving at 7 m/s. I should hold my current speed: <PLAN>KEEP,STRAIGHT</PLAN>

### 26. `KEEP,STRAIGHT`　v0=4.7 m/s　nav=RIGHT　灯=无记录
<sub>2021.06.23.14.06.20_veh-26_01563_02494 · token `69679d50376f5544`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/none_26_CAM_F0.jpg) | ![](image_v5/none_26_CAM_L0.jpg) | ![](image_v5/none_26_CAM_R0.jpg) | ![](image_v5/none_26_CAM_B0.jpg) |

**生成的 reasoning:**

> I should hold my current speed: <PLAN>KEEP,STRAIGHT</PLAN>

### 27. `ACCELERATE,STRAIGHT`　v0=1.1 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.07.16.01.22.41_veh-14_04315_07102 · token `0840856ebb6b56dd`</sub>

```
q1/q2 最近 3 个对象（共 18）
    Vehicle    rear left     ( -11.7,   8.6) m    5.9 m/s
    Vehicle    front left    (  10.4,  10.9) m    6.1 m/s
    Vehicle    rear left     (  -8.4,  14.6) m    7.3 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/none_27_CAM_F0.jpg) | ![](image_v5/none_27_CAM_L0.jpg) | ![](image_v5/none_27_CAM_R0.jpg) | ![](image_v5/none_27_CAM_B0.jpg) |

**生成的 reasoning:**

> I should speed up: <PLAN>ACCELERATE,STRAIGHT</PLAN>

### 28. `ACCELERATE,STRAIGHT`　v0=2.1 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.08.17.17.17.01_veh-45_01443_01678 · token `1c0fec75713b5afb`</sub>

```
q1/q2 最近 3 个对象（共 4）
    Vehicle    side right    (  -4.8,  -6.8) m    0.0 m/s
    Vehicle    front right   (   6.1,  -6.9) m    0.0 m/s
    Vehicle    front right   (  16.5,  -5.7) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/none_28_CAM_F0.jpg) | ![](image_v5/none_28_CAM_L0.jpg) | ![](image_v5/none_28_CAM_R0.jpg) | ![](image_v5/none_28_CAM_B0.jpg) |

**生成的 reasoning:**

> I should speed up: <PLAN>ACCELERATE,STRAIGHT</PLAN>

### 29. `KEEP,STRAIGHT`　v0=5.2 m/s　nav=LEFT　灯=无记录
<sub>2021.07.09.20.26.06_veh-35_03898_05974 · token `7f130b63caff5a66`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/none_29_CAM_F0.jpg) | ![](image_v5/none_29_CAM_L0.jpg) | ![](image_v5/none_29_CAM_R0.jpg) | ![](image_v5/none_29_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 17 meters ahead in my lane is moving at 7 m/s. I should hold my current speed: <PLAN>KEEP,STRAIGHT</PLAN>

### 30. `ACCELERATE,STRAIGHT`　v0=1.0 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.07.09.15.53.28_veh-38_00184_02293 · token `4f733785b3b35f8a`</sub>

```
q1/q2 最近 3 个对象（共 10）
    Vehicle    side right    (   2.4, -11.0) m    0.0 m/s
    Vehicle    front right   (   5.4, -10.5) m    0.0 m/s
    Vehicle    front right   (   8.6, -10.6) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v5/none_30_CAM_F0.jpg) | ![](image_v5/none_30_CAM_L0.jpg) | ![](image_v5/none_30_CAM_R0.jpg) | ![](image_v5/none_30_CAM_B0.jpg) |

**生成的 reasoning:**

> I should speed up: <PLAN>ACCELERATE,STRAIGHT</PLAN>

---

