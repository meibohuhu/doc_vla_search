# NAVSIM reasoning —— 按类别抽样 v6(**真实训练 token 集**)

日期 2026-09-02(v6，另一批帧) · 数据源 `cot_navtrain.json`(100,222 条)· 生成器 [`build_navsim_cot.py`](../../../Impromptu-VLA/data_qa_generate/data_engine/datasets/navsim/build_navsim_cot.py)

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

### 1. `STOP,STRAIGHT`　v0=1.9 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.06.07.19.43.00_veh-35_02298_02525 · token `2ce71cd2a0565aa9`</sub>

```
q1/q2 最近 3 个对象（共 25）
    Vehicle    side left     (   1.9,   3.4) m    5.3 m/s
    Pedestrian side right    (  -4.6,  -5.3) m    0.1 m/s
    Pedestrian rear right    (  -5.6,  -5.4) m    0.2 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/stay_behind_1_CAM_F0.jpg) | ![](image_v6/stay_behind_1_CAM_L0.jpg) | ![](image_v6/stay_behind_1_CAM_R0.jpg) | ![](image_v6/stay_behind_1_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 10 meters ahead in my lane is moving slowly. Because I need to stay behind it, I should remain stopped: <PLAN>STOP,STRAIGHT</PLAN>

### 2. `DECELERATE,STRAIGHT`　v0=6.9 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.07.16.01.22.41_veh-14_04315_07102 · token `e36a9f4f0e835235`</sub>

```
q1/q2 最近 3 个对象（共 15）
    Pedestrian side right    (  -3.9, -12.5) m    1.2 m/s
    Pedestrian rear right    (  -7.8, -11.0) m    0.9 m/s
    Vehicle    front right   (  11.8,  -7.3) m    3.6 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/stay_behind_2_CAM_F0.jpg) | ![](image_v6/stay_behind_2_CAM_L0.jpg) | ![](image_v6/stay_behind_2_CAM_R0.jpg) | ![](image_v6/stay_behind_2_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 26 meters ahead in my lane is moving slowly. Because I need to stay behind it, I should slow down: <PLAN>DECELERATE,STRAIGHT</PLAN>

### 3. `STOP,STRAIGHT`　v0=3.7 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.06.09.12.51.31_veh-35_00697_00820 · token `aa0e336da58a56e1`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/stay_behind_3_CAM_F0.jpg) | ![](image_v6/stay_behind_3_CAM_L0.jpg) | ![](image_v6/stay_behind_3_CAM_R0.jpg) | ![](image_v6/stay_behind_3_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 7 meters ahead in my lane is moving at 3 m/s. Because I need to stay behind it, I should come to a stop: <PLAN>STOP,STRAIGHT</PLAN>

### 4. `DECELERATE,STRAIGHT`　v0=5.8 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.07.09.23.23.48_veh-26_01319_01432 · token `e6ba75d23b3a548d`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/stay_behind_4_CAM_F0.jpg) | ![](image_v6/stay_behind_4_CAM_L0.jpg) | ![](image_v6/stay_behind_4_CAM_R0.jpg) | ![](image_v6/stay_behind_4_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 8 meters ahead in my lane is stopped. Because I need to stay behind it, I should slow down: <PLAN>DECELERATE,STRAIGHT</PLAN>

### 5. `STOP,STRAIGHT`　v0=1.2 m/s　nav=LEFT　灯=无记录
<sub>2021.06.09.12.39.51_veh-26_04543_05321 · token `51f4423004a75da9`</sub>

```
q1/q2 最近 3 个对象（共 42）
    Vehicle    side left     (  -4.7,   1.9) m    1.1 m/s
    Vehicle    side left     (  -3.9,   5.8) m    0.0 m/s
    Vehicle    front center  (   9.6,   2.8) m    1.7 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/stay_behind_5_CAM_F0.jpg) | ![](image_v6/stay_behind_5_CAM_L0.jpg) | ![](image_v6/stay_behind_5_CAM_R0.jpg) | ![](image_v6/stay_behind_5_CAM_B0.jpg) |

**生成的 reasoning:**

> I see pedestrians 18 meters ahead of me. The vehicle 9 meters ahead in my lane is moving slowly. Because I need to stay behind it, I should remain stopped: <PLAN>STOP,STRAIGHT</PLAN>

### 6. `STOP,STRAIGHT`　v0=1.5 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.06.09.12.39.51_veh-26_00609_01168 · token `30cf5abfca915573`</sub>

```
q1/q2 最近 3 个对象（共 12）
    Vehicle    side left     (  -1.0,   3.2) m    0.0 m/s
    Vehicle    front center  (  11.1,   0.3) m    0.0 m/s
    Vehicle    rear left     (  -5.2,  11.1) m    8.8 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/stay_behind_6_CAM_F0.jpg) | ![](image_v6/stay_behind_6_CAM_L0.jpg) | ![](image_v6/stay_behind_6_CAM_R0.jpg) | ![](image_v6/stay_behind_6_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 11 meters ahead in my lane is stopped. Because I need to stay behind it, I should remain stopped: <PLAN>STOP,STRAIGHT</PLAN>

### 7. `DECELERATE,STRAIGHT`　v0=4.6 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.06.23.18.23.38_veh-26_00663_01217 · token `da0736a637405df3`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/stay_behind_7_CAM_F0.jpg) | ![](image_v6/stay_behind_7_CAM_L0.jpg) | ![](image_v6/stay_behind_7_CAM_R0.jpg) | ![](image_v6/stay_behind_7_CAM_B0.jpg) |

**生成的 reasoning:**

> I see pedestrians 6 meters ahead of me. The vehicle 16 meters ahead in my lane is moving at 5 m/s. Because I need to stay behind it, I should slow down: <PLAN>DECELERATE,STRAIGHT</PLAN>

### 8. `STOP,STRAIGHT`　v0=2.3 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.07.16.16.27.22_veh-26_02282_03814 · token `06983e06743b514a`</sub>

```
q1/q2 最近 3 个对象（共 31）
    Vehicle    side right    (   3.9,  -5.1) m    1.0 m/s
    Vehicle    side left     (   0.6,   6.5) m    0.0 m/s
    Vehicle    rear left     (  -6.1,   6.7) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/stay_behind_8_CAM_F0.jpg) | ![](image_v6/stay_behind_8_CAM_L0.jpg) | ![](image_v6/stay_behind_8_CAM_R0.jpg) | ![](image_v6/stay_behind_8_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 10 meters ahead in my lane is stopped. Because I need to stay behind it, I should come to a stop: <PLAN>STOP,STRAIGHT</PLAN>

### 9. `DECELERATE,STRAIGHT`　v0=10.1 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.07.16.20.45.29_veh-35_02509_02649 · token `5d19d07033bc52d3`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/stay_behind_9_CAM_F0.jpg) | ![](image_v6/stay_behind_9_CAM_L0.jpg) | ![](image_v6/stay_behind_9_CAM_R0.jpg) | ![](image_v6/stay_behind_9_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 20 meters ahead in my lane is moving at 14 m/s. Because I need to stay behind it, I should slow down: <PLAN>DECELERATE,STRAIGHT</PLAN>

### 10. `STOP,STRAIGHT`　v0=1.1 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.07.16.02.53.40_veh-17_00016_01588 · token `a28534e8b7e75235`</sub>

```
q1/q2 最近 3 个对象（共 34）
    Vehicle    front center  (   5.8,   3.7) m    1.2 m/s
    Vehicle    rear left     (  -6.1,   3.5) m    2.3 m/s
    Vehicle    side left     (  -4.6,   6.9) m    2.8 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/stay_behind_10_CAM_F0.jpg) | ![](image_v6/stay_behind_10_CAM_L0.jpg) | ![](image_v6/stay_behind_10_CAM_R0.jpg) | ![](image_v6/stay_behind_10_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 5 meters ahead in my lane is moving slowly. Because I need to stay behind it, I should remain stopped: <PLAN>STOP,STRAIGHT</PLAN>

---

## 场景:`yield_vru` —— 让行 VRU —— 正前走廊 ≤20m 有行人/骑行者　(全量占比 0.5%)

### 1. `STOP,LEFT`　v0=1.0 m/s　nav=LEFT　灯=无记录
<sub>2021.06.09.12.51.31_veh-35_00007_00089 · token `e827758c9a4d5610`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/yield_vru_1_CAM_F0.jpg) | ![](image_v6/yield_vru_1_CAM_L0.jpg) | ![](image_v6/yield_vru_1_CAM_R0.jpg) | ![](image_v6/yield_vru_1_CAM_B0.jpg) |

**生成的 reasoning:**

> I see pedestrians 12 meters ahead of me. Because I need to yield to them, I should remain stopped and bear left: <PLAN>STOP,LEFT</PLAN>

### 2. `DECELERATE,STRAIGHT`　v0=7.2 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.06.09.11.54.15_veh-12_01403_01526 · token `d364a338ff4656e1`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/yield_vru_2_CAM_F0.jpg) | ![](image_v6/yield_vru_2_CAM_L0.jpg) | ![](image_v6/yield_vru_2_CAM_R0.jpg) | ![](image_v6/yield_vru_2_CAM_B0.jpg) |

**生成的 reasoning:**

> I see a pedestrian 14 meters ahead of me. Because I need to yield to them, I should slow down: <PLAN>DECELERATE,STRAIGHT</PLAN>

### 3. `STOP,STRAIGHT`　v0=2.8 m/s　nav=LEFT　灯=无记录
<sub>2021.05.12.23.36.44_veh-35_01735_01957 · token `b34f06a9557b5585`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/yield_vru_3_CAM_F0.jpg) | ![](image_v6/yield_vru_3_CAM_L0.jpg) | ![](image_v6/yield_vru_3_CAM_R0.jpg) | ![](image_v6/yield_vru_3_CAM_B0.jpg) |

**生成的 reasoning:**

> I see pedestrians 15 meters ahead of me. Because I need to yield to them, I should come to a stop: <PLAN>STOP,STRAIGHT</PLAN>

### 4. `STOP,STRAIGHT`　v0=2.7 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.06.09.17.23.18_veh-38_04544_04697 · token `403cd48e61485877`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/yield_vru_4_CAM_F0.jpg) | ![](image_v6/yield_vru_4_CAM_L0.jpg) | ![](image_v6/yield_vru_4_CAM_R0.jpg) | ![](image_v6/yield_vru_4_CAM_B0.jpg) |

**生成的 reasoning:**

> I see pedestrians and a cyclist 12 meters ahead of me. Because I need to yield to them, I should come to a stop: <PLAN>STOP,STRAIGHT</PLAN>

### 5. `STOP,STRAIGHT`　v0=1.6 m/s　nav=LEFT　灯=无记录
<sub>2021.07.09.20.59.12_veh-38_01208_01692 · token `9381506b45605c88`</sub>

```
q1/q2 最近 3 个对象（共 23）
    Vehicle    side right    (   2.5,  -3.3) m    0.0 m/s
    Pedestrian side left     (   1.8,   3.9) m    0.3 m/s
    Pedestrian side left     (   3.9,   4.0) m    1.1 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/yield_vru_5_CAM_F0.jpg) | ![](image_v6/yield_vru_5_CAM_L0.jpg) | ![](image_v6/yield_vru_5_CAM_R0.jpg) | ![](image_v6/yield_vru_5_CAM_B0.jpg) |

**生成的 reasoning:**

> I see pedestrians 10 meters ahead of me. Because I need to yield to them, I should remain stopped: <PLAN>STOP,STRAIGHT</PLAN>

### 6. `STOP,STRAIGHT`　v0=1.6 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.07.16.20.45.29_veh-35_00600_01084 · token `45eead460b09526d`</sub>

```
q1/q2 最近 3 个对象（共 130）
    Vehicle    side right    (   0.7,  -3.3) m    0.0 m/s
    Pedestrian front left    (   5.8,   7.9) m    1.4 m/s
    Pedestrian front center  (  10.1,   1.2) m    1.3 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/yield_vru_6_CAM_F0.jpg) | ![](image_v6/yield_vru_6_CAM_L0.jpg) | ![](image_v6/yield_vru_6_CAM_R0.jpg) | ![](image_v6/yield_vru_6_CAM_B0.jpg) |

**生成的 reasoning:**

> I see pedestrians 9 meters ahead of me. Because I need to yield to them, I should remain stopped: <PLAN>STOP,STRAIGHT</PLAN>

### 7. `STOP,LEFT`　v0=3.0 m/s　nav=LEFT　灯=无记录
<sub>2021.06.09.17.23.18_veh-38_02526_03027 · token `afeac42dbbf75736`</sub>

```
q1/q2 最近 3 个对象（共 13）
    Pedestrian side left     (   1.0,  14.9) m    1.0 m/s
    Pedestrian front center  (  16.2,  -1.4) m    0.0 m/s
    Pedestrian front center  (  16.9,  -0.7) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/yield_vru_7_CAM_F0.jpg) | ![](image_v6/yield_vru_7_CAM_L0.jpg) | ![](image_v6/yield_vru_7_CAM_R0.jpg) | ![](image_v6/yield_vru_7_CAM_B0.jpg) |

**生成的 reasoning:**

> I see pedestrians 16 meters ahead of me. Because I need to yield to them, I should come to a stop and bear left: <PLAN>STOP,LEFT</PLAN>

### 8. `STOP,STRAIGHT`　v0=4.3 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.07.16.16.27.22_veh-26_01536_02260 · token `c8adc24c2cb05259`</sub>

```
q1/q2 最近 3 个对象（共 55）
    Vehicle    side right    (  -3.0, -10.8) m    0.0 m/s
    Pedestrian front right   (   6.9, -10.2) m    0.9 m/s
    Vehicle    rear right    ( -12.0,  -4.7) m    7.5 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/yield_vru_8_CAM_F0.jpg) | ![](image_v6/yield_vru_8_CAM_L0.jpg) | ![](image_v6/yield_vru_8_CAM_R0.jpg) | ![](image_v6/yield_vru_8_CAM_B0.jpg) |

**生成的 reasoning:**

> I see pedestrians 16 meters ahead of me. Because I need to yield to them, I should come to a stop: <PLAN>STOP,STRAIGHT</PLAN>

### 9. `STOP,STRAIGHT`　v0=0.8 m/s　nav=LEFT　灯=无记录
<sub>2021.06.23.21.56.29_veh-35_00220_00936 · token `efca95aef7615995`</sub>

```
q1/q2 最近 3 个对象（共 23）
    Vehicle    side right    (   0.4,  -4.2) m    0.0 m/s
    Vehicle    side left     (  -3.4,   3.8) m    0.6 m/s
    Pedestrian side left     (   4.6,   2.5) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/yield_vru_9_CAM_F0.jpg) | ![](image_v6/yield_vru_9_CAM_L0.jpg) | ![](image_v6/yield_vru_9_CAM_R0.jpg) | ![](image_v6/yield_vru_9_CAM_B0.jpg) |

**生成的 reasoning:**

> I see pedestrians 5 meters ahead of me. Because I need to yield to them, I should remain stopped: <PLAN>STOP,STRAIGHT</PLAN>

### 10. `STOP,LEFT`　v0=2.0 m/s　nav=LEFT　灯=无记录
<sub>2021.06.08.12.00.19_veh-35_05235_05578 · token `f037846a9e2951af`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/yield_vru_10_CAM_F0.jpg) | ![](image_v6/yield_vru_10_CAM_L0.jpg) | ![](image_v6/yield_vru_10_CAM_R0.jpg) | ![](image_v6/yield_vru_10_CAM_B0.jpg) |

**生成的 reasoning:**

> I see pedestrians 9 meters ahead of me. A vehicle 12 meters on my right will move into my path. Because I need to yield to them, I should remain stopped and bear left: <PLAN>STOP,LEFT</PLAN>

---

## 场景:`cross_path` —— 切入 —— 从我前方穿过中线，或并入我车道　(全量占比 1.0%)

### 1. `DECELERATE,STRAIGHT`　v0=5.6 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.10.05.06.57.40_veh-50_01131_01452 · token `47a980aab2ed5a0e`</sub>

```
q1/q2 最近 3 个对象（共 3）
    Cyclist    front right   (  20.0,  -4.6) m    0.0 m/s
    Pedestrian rear right    ( -19.3,  -7.8) m    1.3 m/s
    Pedestrian front left    (  22.4,   4.6) m    1.2 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/cross_path_1_CAM_F0.jpg) | ![](image_v6/cross_path_1_CAM_L0.jpg) | ![](image_v6/cross_path_1_CAM_R0.jpg) | ![](image_v6/cross_path_1_CAM_B0.jpg) |

**生成的 reasoning:**

> Because the pedestrian on my left will move into my path, I should slow down: <PLAN>DECELERATE,STRAIGHT</PLAN>

### 2. `STOP,STRAIGHT`　v0=5.8 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.08.09.18.37.41_veh-28_00053_00548 · token `3a0916b93da7551b`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/cross_path_2_CAM_F0.jpg) | ![](image_v6/cross_path_2_CAM_L0.jpg) | ![](image_v6/cross_path_2_CAM_R0.jpg) | ![](image_v6/cross_path_2_CAM_B0.jpg) |

**生成的 reasoning:**

> Because the pedestrian on my right will move into my path, I should come to a stop: <PLAN>STOP,STRAIGHT</PLAN>

### 3. `STOP,STRAIGHT`　v0=4.6 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.07.16.16.01.30_veh-38_03893_05253 · token `44bde6a7387f5120`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/cross_path_3_CAM_F0.jpg) | ![](image_v6/cross_path_3_CAM_L0.jpg) | ![](image_v6/cross_path_3_CAM_R0.jpg) | ![](image_v6/cross_path_3_CAM_B0.jpg) |

**生成的 reasoning:**

> Because the vehicle on my left will move into my path, I should come to a stop: <PLAN>STOP,STRAIGHT</PLAN>

### 4. `STOP,STRAIGHT`　v0=3.8 m/s　nav=LEFT　灯=无记录
<sub>2021.06.23.15.56.12_veh-16_01308_04289 · token `c404278a162555b2`</sub>

```
q1/q2 最近 3 个对象（共 13）
    Vehicle    side left     (  -0.2,   9.1) m    0.0 m/s
    Pedestrian front left    (  12.7,  13.1) m    0.6 m/s
    Vehicle    front left    (  12.2,  15.1) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/cross_path_4_CAM_F0.jpg) | ![](image_v6/cross_path_4_CAM_L0.jpg) | ![](image_v6/cross_path_4_CAM_R0.jpg) | ![](image_v6/cross_path_4_CAM_B0.jpg) |

**生成的 reasoning:**

> Because the pedestrian on my center will move into my path, I should come to a stop: <PLAN>STOP,STRAIGHT</PLAN>

### 5. `STOP,LEFT`　v0=3.5 m/s　nav=LEFT　灯=无记录
<sub>2021.07.16.16.01.30_veh-38_00356_02486 · token `c2da7bb1211a5cd8`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/cross_path_5_CAM_F0.jpg) | ![](image_v6/cross_path_5_CAM_L0.jpg) | ![](image_v6/cross_path_5_CAM_R0.jpg) | ![](image_v6/cross_path_5_CAM_B0.jpg) |

**生成的 reasoning:**

> Because the pedestrian on my center will move into my path, I should come to a stop and bear left: <PLAN>STOP,LEFT</PLAN>

### 6. `DECELERATE,LEFT`　v0=5.7 m/s　nav=LEFT　灯=RED
<sub>2021.07.16.16.08.35_veh-35_01303_01641 · token `36a41ad5d5a9516b`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/cross_path_6_CAM_F0.jpg) | ![](image_v6/cross_path_6_CAM_L0.jpg) | ![](image_v6/cross_path_6_CAM_R0.jpg) | ![](image_v6/cross_path_6_CAM_B0.jpg) |

**生成的 reasoning:**

> Because the vehicle on my left will move into my path, I should slow down and bear left: <PLAN>DECELERATE,LEFT</PLAN>

### 7. `DECELERATE,LEFT`　v0=4.7 m/s　nav=LEFT　灯=无记录
<sub>2021.07.16.01.22.41_veh-14_04315_07102 · token `141eef70c106569d`</sub>

```
q1/q2 最近 3 个对象（共 12）
    Vehicle    rear left     (  -7.3,   5.5) m    0.0 m/s
    Vehicle    front left    (   9.2,   9.2) m    0.0 m/s
    Vehicle    front center  (  15.2,  -1.6) m    2.6 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/cross_path_7_CAM_F0.jpg) | ![](image_v6/cross_path_7_CAM_L0.jpg) | ![](image_v6/cross_path_7_CAM_R0.jpg) | ![](image_v6/cross_path_7_CAM_B0.jpg) |

**生成的 reasoning:**

> Because the vehicle on my center will move into my path, I should slow down and bear left: <PLAN>DECELERATE,LEFT</PLAN>

### 8. `DECELERATE,STRAIGHT`　v0=6.0 m/s　nav=LEFT　灯=GREEN
<sub>2021.08.17.18.13.38_veh-45_00946_01854 · token `7033e7addf2354e3`</sub>

```
q1/q2 最近 3 个对象（共 16）
    Vehicle    rear right    (  -8.0,  -4.1) m    0.0 m/s
    Vehicle    front left    (   5.9,  10.0) m    9.3 m/s
    Vehicle    rear right    ( -15.6,  -3.8) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/cross_path_8_CAM_F0.jpg) | ![](image_v6/cross_path_8_CAM_L0.jpg) | ![](image_v6/cross_path_8_CAM_R0.jpg) | ![](image_v6/cross_path_8_CAM_B0.jpg) |

**生成的 reasoning:**

> Because the pedestrian on my center will move into my path, I should slow down: <PLAN>DECELERATE,STRAIGHT</PLAN>

### 9. `DECELERATE,STRAIGHT`　v0=6.4 m/s　nav=RIGHT　灯=无记录
<sub>2021.08.17.13.10.50_veh-08_00726_01027 · token `25c492bc486f5b03`</sub>

```
q1/q2 最近 3 个对象（共 5）
    Pedestrian front right   (  26.7,  -6.5) m    1.8 m/s
    Pedestrian front right   (  26.6,  -7.4) m    1.8 m/s
    Pedestrian front right   (  25.9, -11.7) m    0.7 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/cross_path_9_CAM_F0.jpg) | ![](image_v6/cross_path_9_CAM_L0.jpg) | ![](image_v6/cross_path_9_CAM_R0.jpg) | ![](image_v6/cross_path_9_CAM_B0.jpg) |

**生成的 reasoning:**

> Because the pedestrian on my right will move into my path, I should slow down: <PLAN>DECELERATE,STRAIGHT</PLAN>

### 10. `DECELERATE,LEFT`　v0=5.0 m/s　nav=STRAIGHT　灯=RED
<sub>2021.06.09.12.51.31_veh-35_02975_03207 · token `bd21d7e3f5e55bfe`</sub>

```
q1/q2 最近 3 个对象（共 34）
    Vehicle    front left    (   9.9,   5.9) m    0.0 m/s
    Vehicle    rear left     ( -14.5,   2.6) m    5.0 m/s
    Vehicle    front left    (  17.3,   9.6) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/cross_path_10_CAM_F0.jpg) | ![](image_v6/cross_path_10_CAM_L0.jpg) | ![](image_v6/cross_path_10_CAM_R0.jpg) | ![](image_v6/cross_path_10_CAM_B0.jpg) |

**生成的 reasoning:**

> Because the pedestrian on my center will move into my path, I should slow down and bear left: <PLAN>DECELERATE,LEFT</PLAN>

---

## 场景:`stop_sign` —— 停车让行标志 —— 正前 8~40m 有 STOP_SIGN　(全量占比 1.0%)

### 1. `STOP,STRAIGHT`　v0=4.4 m/s　nav=STRAIGHT　灯=无记录　stop_sign=14m
<sub>2021.08.24.15.09.18_veh-45_00216_00862 · token `f35f0073fdcc5d9b`</sub>

```
q1/q2 最近 3 个对象（共 1）
    Vehicle    front left    (  20.8,  15.3) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/stop_sign_1_CAM_F0.jpg) | ![](image_v6/stop_sign_1_CAM_L0.jpg) | ![](image_v6/stop_sign_1_CAM_R0.jpg) | ![](image_v6/stop_sign_1_CAM_B0.jpg) |

**生成的 reasoning:**

> Because there is a stop sign ahead, I should come to a stop: <PLAN>STOP,STRAIGHT</PLAN>

### 2. `DECELERATE,STRAIGHT`　v0=5.1 m/s　nav=STRAIGHT　灯=无记录　stop_sign=26m
<sub>2021.10.05.06.31.40_veh-52_00734_01305 · token `0c7af9b6379d5ef6`</sub>

```
q1/q2 最近 3 个对象（共 2）
    Vehicle    front left    (   8.3,   6.7) m    0.0 m/s
    Pedestrian front left    (  23.3,   6.5) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/stop_sign_2_CAM_F0.jpg) | ![](image_v6/stop_sign_2_CAM_L0.jpg) | ![](image_v6/stop_sign_2_CAM_R0.jpg) | ![](image_v6/stop_sign_2_CAM_B0.jpg) |

**生成的 reasoning:**

> Because there is a stop sign ahead, I should slow down: <PLAN>DECELERATE,STRAIGHT</PLAN>

### 3. `STOP,STRAIGHT`　v0=2.9 m/s　nav=RIGHT　灯=无记录　stop_sign=28m
<sub>2021.07.09.23.23.48_veh-26_02228_04624 · token `d573ff879d86576d`</sub>

```
q1/q2 最近 3 个对象（共 16）
    Vehicle    front left    (  11.0,  12.2) m    5.0 m/s
    Pedestrian rear right    ( -14.6,  -9.2) m    0.0 m/s
    Vehicle    front left    (  17.7,   4.3) m    5.2 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/stop_sign_3_CAM_F0.jpg) | ![](image_v6/stop_sign_3_CAM_L0.jpg) | ![](image_v6/stop_sign_3_CAM_R0.jpg) | ![](image_v6/stop_sign_3_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 39 meters ahead in my lane is stopped. Because there is a stop sign ahead, I should come to a stop: <PLAN>STOP,STRAIGHT</PLAN>

### 4. `STOP,STRAIGHT`　v0=4.4 m/s　nav=LEFT　灯=无记录　stop_sign=13m
<sub>2021.06.14.18.13.35_veh-26_04547_04710 · token `b4e58cd39f745314`</sub>

```
q1/q2 最近 3 个对象（共 20）
    Pedestrian rear left     ( -24.2,   5.7) m    0.0 m/s
    Vehicle    rear left     ( -10.2,  23.2) m    0.0 m/s
    Pedestrian rear left     ( -24.8,   6.1) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/stop_sign_4_CAM_F0.jpg) | ![](image_v6/stop_sign_4_CAM_L0.jpg) | ![](image_v6/stop_sign_4_CAM_R0.jpg) | ![](image_v6/stop_sign_4_CAM_B0.jpg) |

**生成的 reasoning:**

> A vehicle 36 meters in front of me will move into my path. Because there is a stop sign ahead, I should come to a stop: <PLAN>STOP,STRAIGHT</PLAN>

### 5. `STOP,STRAIGHT`　v0=5.0 m/s　nav=STRAIGHT　灯=无记录　stop_sign=21m
<sub>2021.07.09.23.23.48_veh-26_02228_04624 · token `3ed5c8d2a608504f`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/stop_sign_5_CAM_F0.jpg) | ![](image_v6/stop_sign_5_CAM_L0.jpg) | ![](image_v6/stop_sign_5_CAM_R0.jpg) | ![](image_v6/stop_sign_5_CAM_B0.jpg) |

**生成的 reasoning:**

> Because there is a stop sign ahead, I should come to a stop: <PLAN>STOP,STRAIGHT</PLAN>

### 6. `STOP,STRAIGHT`　v0=4.4 m/s　nav=LEFT　灯=无记录　stop_sign=12m
<sub>2021.10.11.08.31.07_veh-50_00282_00680 · token `86e6bc4289fe5e4d`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/stop_sign_6_CAM_F0.jpg) | ![](image_v6/stop_sign_6_CAM_L0.jpg) | ![](image_v6/stop_sign_6_CAM_R0.jpg) | ![](image_v6/stop_sign_6_CAM_B0.jpg) |

**生成的 reasoning:**

> Because there is a stop sign ahead, I should come to a stop: <PLAN>STOP,STRAIGHT</PLAN>

### 7. `STOP,STRAIGHT`　v0=1.4 m/s　nav=RIGHT　灯=无记录　stop_sign=8m
<sub>2021.09.15.13.16.40_veh-28_00366_00631 · token `55839762db225a3f`</sub>

```
q1/q2 最近 3 个对象（共 22）
    Vehicle    rear left     (  -5.2,   0.4) m    1.1 m/s
    Vehicle    side left     (  -2.1,   5.5) m    0.0 m/s
    Vehicle    side left     (   3.1,   5.9) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/stop_sign_7_CAM_F0.jpg) | ![](image_v6/stop_sign_7_CAM_L0.jpg) | ![](image_v6/stop_sign_7_CAM_R0.jpg) | ![](image_v6/stop_sign_7_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 11 meters ahead in my lane is moving slowly. Because there is a stop sign ahead, I should remain stopped: <PLAN>STOP,STRAIGHT</PLAN>

### 8. `STOP,STRAIGHT`　v0=5.9 m/s　nav=STRAIGHT　灯=无记录　stop_sign=17m
<sub>2021.09.15.12.32.43_veh-28_01513_01697 · token `93e2cb298e615f37`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/stop_sign_8_CAM_F0.jpg) | ![](image_v6/stop_sign_8_CAM_L0.jpg) | ![](image_v6/stop_sign_8_CAM_R0.jpg) | ![](image_v6/stop_sign_8_CAM_B0.jpg) |

**生成的 reasoning:**

> A vehicle 35 meters on my right will move into my path. Because there is a stop sign ahead, I should come to a stop: <PLAN>STOP,STRAIGHT</PLAN>

### 9. `DECELERATE,STRAIGHT`　v0=5.5 m/s　nav=STRAIGHT　灯=无记录　stop_sign=23m
<sub>2021.06.09.14.03.17_veh-12_04129_04237 · token `d243f570f1615426`</sub>

```
q1/q2 最近 3 个对象（共 10）
    Vehicle    side right    (  -0.7,  -5.7) m    7.7 m/s
    Vehicle    rear right    ( -11.1, -12.8) m    0.0 m/s
    Pedestrian front right   (  21.1,  -8.3) m    1.1 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/stop_sign_9_CAM_F0.jpg) | ![](image_v6/stop_sign_9_CAM_L0.jpg) | ![](image_v6/stop_sign_9_CAM_R0.jpg) | ![](image_v6/stop_sign_9_CAM_B0.jpg) |

**生成的 reasoning:**

> Because there is a stop sign ahead, I should slow down: <PLAN>DECELERATE,STRAIGHT</PLAN>

### 10. `DECELERATE,STRAIGHT`　v0=3.9 m/s　nav=STRAIGHT　灯=无记录　stop_sign=15m
<sub>2021.08.17.17.17.01_veh-45_00207_00594 · token `454320aecce558cf`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/stop_sign_10_CAM_F0.jpg) | ![](image_v6/stop_sign_10_CAM_L0.jpg) | ![](image_v6/stop_sign_10_CAM_R0.jpg) | ![](image_v6/stop_sign_10_CAM_B0.jpg) |

**生成的 reasoning:**

> Because there is a stop sign ahead, I should slow down: <PLAN>DECELERATE,STRAIGHT</PLAN>

---

## 场景:`gap_big` —— 前方放空 —— 前车在加速且已拉开　(全量占比 0.4%)

### 1. `KEEP,STRAIGHT`　v0=4.1 m/s　nav=RIGHT　灯=无记录
<sub>2021.06.23.20.00.35_veh-35_00130_00949 · token `309d7afd25cc5476`</sub>

```
q1/q2 最近 3 个对象（共 8）
    Pedestrian front left    (  22.6,  19.3) m    0.0 m/s
    Vehicle    front left    (  23.2,  19.2) m    0.0 m/s
    Vehicle    front left    (  19.9,  24.6) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/gap_big_1_CAM_F0.jpg) | ![](image_v6/gap_big_1_CAM_L0.jpg) | ![](image_v6/gap_big_1_CAM_R0.jpg) | ![](image_v6/gap_big_1_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 32 meters ahead in my lane is stopped. Because the gap in front of me is opening up, I should hold my current speed: <PLAN>KEEP,STRAIGHT</PLAN>

### 2. `ACCELERATE,STRAIGHT`　v0=13.6 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.06.09.12.39.51_veh-26_02901_02978 · token `f99b5da240c456cb`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/gap_big_2_CAM_F0.jpg) | ![](image_v6/gap_big_2_CAM_L0.jpg) | ![](image_v6/gap_big_2_CAM_R0.jpg) | ![](image_v6/gap_big_2_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 28 meters ahead in my lane is moving at 17 m/s. Because the gap in front of me is opening up, I should speed up: <PLAN>ACCELERATE,STRAIGHT</PLAN>

### 3. `KEEP,STRAIGHT`　v0=11.0 m/s　nav=STRAIGHT　灯=GREEN
<sub>2021.10.01.13.28.54_veh-28_01098_01337 · token `aa97edfaebde597a`</sub>

```
q1/q2 最近 3 个对象（共 14）
    Vehicle    rear right    (  -9.5,  -2.7) m    0.0 m/s
    Cyclist    rear right    ( -13.1,  -0.1) m   10.9 m/s
    Vehicle    front left    (   6.5,  12.8) m    7.5 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/gap_big_3_CAM_F0.jpg) | ![](image_v6/gap_big_3_CAM_L0.jpg) | ![](image_v6/gap_big_3_CAM_R0.jpg) | ![](image_v6/gap_big_3_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 31 meters ahead in my lane is moving at 7 m/s. Because the gap in front of me is opening up, I should hold my current speed: <PLAN>KEEP,STRAIGHT</PLAN>

### 4. `KEEP,STRAIGHT`　v0=6.0 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.07.16.16.08.35_veh-35_04744_06051 · token `f9c7bc5888e2558c`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/gap_big_4_CAM_F0.jpg) | ![](image_v6/gap_big_4_CAM_L0.jpg) | ![](image_v6/gap_big_4_CAM_R0.jpg) | ![](image_v6/gap_big_4_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 25 meters ahead in my lane is moving at 7 m/s. Because the gap in front of me is opening up, I should hold my current speed: <PLAN>KEEP,STRAIGHT</PLAN>

### 5. `KEEP,STRAIGHT`　v0=4.2 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.06.09.14.58.55_veh-35_00193_01084 · token `b23d1154fef5571d`</sub>

```
q1/q2 最近 3 个对象（共 15）
    Vehicle    side right    (  -5.0, -12.7) m    0.0 m/s
    Pedestrian front right   (  14.5,  -9.8) m    1.2 m/s
    Vehicle    rear right    ( -17.6,  -5.0) m    7.4 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/gap_big_5_CAM_F0.jpg) | ![](image_v6/gap_big_5_CAM_L0.jpg) | ![](image_v6/gap_big_5_CAM_R0.jpg) | ![](image_v6/gap_big_5_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 30 meters ahead in my lane is moving at 2 m/s. Because the gap in front of me is opening up, I should hold my current speed: <PLAN>KEEP,STRAIGHT</PLAN>

### 6. `ACCELERATE,RIGHT`　v0=1.6 m/s　nav=RIGHT　灯=无记录
<sub>2021.05.12.22.28.35_veh-35_01175_02127 · token `88eec01c6dc35578`</sub>

```
q1/q2 最近 3 个对象（共 34）
    Vehicle    side left     (  -3.3,   3.3) m    0.0 m/s
    Pedestrian side right    (   2.0,  -5.0) m    0.0 m/s
    Pedestrian side right    (   3.0,  -5.5) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/gap_big_6_CAM_F0.jpg) | ![](image_v6/gap_big_6_CAM_L0.jpg) | ![](image_v6/gap_big_6_CAM_R0.jpg) | ![](image_v6/gap_big_6_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 35 meters ahead in my lane is moving at 3 m/s. Because the gap in front of me is opening up, I should speed up and bear right: <PLAN>ACCELERATE,RIGHT</PLAN>

### 7. `ACCELERATE,STRAIGHT`　v0=8.7 m/s　nav=LEFT　灯=GREEN
<sub>2021.07.09.01.37.16_veh-26_04815_04878 · token `8f97954707315f2b`</sub>

```
q1/q2 最近 3 个对象（共 26）
    Vehicle    rear left     (  -7.7,   7.8) m    7.4 m/s
    Vehicle    side left     (   1.4,  11.3) m    0.0 m/s
    Vehicle    front left    (   8.6,  12.6) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/gap_big_7_CAM_F0.jpg) | ![](image_v6/gap_big_7_CAM_L0.jpg) | ![](image_v6/gap_big_7_CAM_R0.jpg) | ![](image_v6/gap_big_7_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 27 meters ahead in my lane is moving at 11 m/s. Because the gap in front of me is opening up, I should speed up: <PLAN>ACCELERATE,STRAIGHT</PLAN>

### 8. `KEEP,STRAIGHT`　v0=4.6 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.06.14.14.03.45_veh-38_02007_02072 · token `c1d965217d5c5063`</sub>

```
q1/q2 最近 3 个对象（共 7）
    Vehicle    rear left     ( -19.4,   7.7) m    6.1 m/s
    Vehicle    rear right    ( -26.2, -10.9) m    0.0 m/s
    Vehicle    front center  (  29.7,  -2.3) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/gap_big_8_CAM_F0.jpg) | ![](image_v6/gap_big_8_CAM_L0.jpg) | ![](image_v6/gap_big_8_CAM_R0.jpg) | ![](image_v6/gap_big_8_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 29 meters ahead in my lane is stopped. Because the gap in front of me is opening up, I should hold my current speed: <PLAN>KEEP,STRAIGHT</PLAN>

### 9. `ACCELERATE,STRAIGHT`　v0=8.9 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.06.09.12.06.35_veh-35_00284_00410 · token `28b9c0a6392e57e7`</sub>

```
q1/q2 最近 3 个对象（共 4）
    Vehicle    front left    (   8.7,  11.1) m    1.0 m/s
    Vehicle    rear left     ( -16.5,   0.7) m    5.8 m/s
    Vehicle    rear left     ( -10.3,  19.2) m    1.5 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/gap_big_9_CAM_F0.jpg) | ![](image_v6/gap_big_9_CAM_L0.jpg) | ![](image_v6/gap_big_9_CAM_R0.jpg) | ![](image_v6/gap_big_9_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 31 meters ahead in my lane is moving at 11 m/s. Because the gap in front of me is opening up, I should speed up: <PLAN>ACCELERATE,STRAIGHT</PLAN>

### 10. `ACCELERATE,LEFT`　v0=6.6 m/s　nav=LEFT　灯=GREEN
<sub>2021.07.09.20.59.12_veh-38_02064_03281 · token `6257ec6e397756a1`</sub>

```
q1/q2 最近 3 个对象（共 33）
    Vehicle    rear left     ( -11.9,   3.3) m    6.0 m/s
    Pedestrian side right    (   3.4, -20.8) m    1.2 m/s
    Vehicle    side left     (   4.6,  22.2) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/gap_big_10_CAM_F0.jpg) | ![](image_v6/gap_big_10_CAM_L0.jpg) | ![](image_v6/gap_big_10_CAM_R0.jpg) | ![](image_v6/gap_big_10_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 26 meters ahead in my lane is moving at 8 m/s. Because the gap in front of me is opening up, I should speed up and bear left: <PLAN>ACCELERATE,LEFT</PLAN>

---

## 场景:`none` —— 没有理由从句　(全量占比 67.6%)

### 1. `KEEP,LEFT`　v0=5.9 m/s　nav=LEFT　灯=无记录
<sub>2021.10.06.18.52.07_veh-28_01072_01157 · token `9db818a368fe5b61`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/none_1_CAM_F0.jpg) | ![](image_v6/none_1_CAM_L0.jpg) | ![](image_v6/none_1_CAM_R0.jpg) | ![](image_v6/none_1_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 26 meters ahead in my lane is stopped. I should hold my current speed and bear left: <PLAN>KEEP,LEFT</PLAN>

### 2. `ACCELERATE,RIGHT`　v0=2.5 m/s　nav=RIGHT　灯=GREEN
<sub>2021.07.09.17.06.37_veh-35_02609_05015 · token `de99ac8969415979`</sub>

```
q1/q2 最近 3 个对象（共 55）
    Pedestrian side right    (  -2.7,  -4.0) m    0.0 m/s
    Pedestrian side right    (  -0.1,  -5.6) m    1.6 m/s
    Vehicle    side left     (  -4.6,   3.6) m   10.7 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/none_2_CAM_F0.jpg) | ![](image_v6/none_2_CAM_L0.jpg) | ![](image_v6/none_2_CAM_R0.jpg) | ![](image_v6/none_2_CAM_B0.jpg) |

**生成的 reasoning:**

> I see pedestrians 13 meters ahead of me. I should speed up and bear right: <PLAN>ACCELERATE,RIGHT</PLAN>

### 3. `DECELERATE,RIGHT`　v0=5.3 m/s　nav=RIGHT　灯=无记录
<sub>2021.06.23.14.54.32_veh-16_01187_03336 · token `2703857f11285d68`</sub>

```
q1/q2 最近 3 个对象（共 6）
    Vehicle    side left     (   2.0,  14.9) m   14.4 m/s
    Vehicle    rear right    ( -14.8,  -3.2) m    7.5 m/s
    Vehicle    rear right    ( -30.3,  -5.0) m    7.6 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/none_3_CAM_F0.jpg) | ![](image_v6/none_3_CAM_L0.jpg) | ![](image_v6/none_3_CAM_R0.jpg) | ![](image_v6/none_3_CAM_B0.jpg) |

**生成的 reasoning:**

> I should slow down and bear right: <PLAN>DECELERATE,RIGHT</PLAN>

### 4. `STOP,STRAIGHT`　v0=3.3 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.10.05.07.10.04_veh-52_00596_00663 · token `18386a87fcfb59ff`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/none_4_CAM_F0.jpg) | ![](image_v6/none_4_CAM_L0.jpg) | ![](image_v6/none_4_CAM_R0.jpg) | ![](image_v6/none_4_CAM_B0.jpg) |

**生成的 reasoning:**

> I should come to a stop: <PLAN>STOP,STRAIGHT</PLAN>

### 5. `KEEP,STRAIGHT`　v0=8.4 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.10.05.07.38.12_veh-50_01085_01463 · token `c4d93843c02a56df`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/none_5_CAM_F0.jpg) | ![](image_v6/none_5_CAM_L0.jpg) | ![](image_v6/none_5_CAM_R0.jpg) | ![](image_v6/none_5_CAM_B0.jpg) |

**生成的 reasoning:**

> I should hold my current speed: <PLAN>KEEP,STRAIGHT</PLAN>

### 6. `ACCELERATE,STRAIGHT`　v0=0.3 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.10.06.14.31.13_veh-28_00362_00475 · token `65c1dfe6f66f5427`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/none_6_CAM_F0.jpg) | ![](image_v6/none_6_CAM_L0.jpg) | ![](image_v6/none_6_CAM_R0.jpg) | ![](image_v6/none_6_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 18 meters ahead in my lane is moving at 7 m/s. I should speed up: <PLAN>ACCELERATE,STRAIGHT</PLAN>

### 7. `KEEP,LEFT`　v0=3.9 m/s　nav=LEFT　灯=无记录
<sub>2021.07.16.18.19.22_veh-35_00869_03454 · token `33336a50210c530e`</sub>

```
q1/q2 最近 3 个对象（共 7）
    Vehicle    side right    (  -2.7, -14.4) m    0.0 m/s
    Vehicle    side right    (   2.4, -15.3) m    0.0 m/s
    Vehicle    rear right    ( -25.4,  -0.2) m    5.8 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/none_7_CAM_F0.jpg) | ![](image_v6/none_7_CAM_L0.jpg) | ![](image_v6/none_7_CAM_R0.jpg) | ![](image_v6/none_7_CAM_B0.jpg) |

**生成的 reasoning:**

> I should hold my current speed and bear left: <PLAN>KEEP,LEFT</PLAN>

### 8. `ACCELERATE,RIGHT`　v0=1.4 m/s　nav=RIGHT　灯=无记录
<sub>2021.08.24.14.35.46_veh-45_00549_00693 · token `939e8428fbdc5bcf`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/none_8_CAM_F0.jpg) | ![](image_v6/none_8_CAM_L0.jpg) | ![](image_v6/none_8_CAM_R0.jpg) | ![](image_v6/none_8_CAM_B0.jpg) |

**生成的 reasoning:**

> I should speed up and bear right: <PLAN>ACCELERATE,RIGHT</PLAN>

### 9. `ACCELERATE,LEFT`　v0=4.3 m/s　nav=LEFT　灯=无记录
<sub>2021.10.05.06.31.40_veh-52_01598_02013 · token `5a652839d3295fca`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/none_9_CAM_F0.jpg) | ![](image_v6/none_9_CAM_L0.jpg) | ![](image_v6/none_9_CAM_R0.jpg) | ![](image_v6/none_9_CAM_B0.jpg) |

**生成的 reasoning:**

> I should speed up and bear left: <PLAN>ACCELERATE,LEFT</PLAN>

### 10. `KEEP,RIGHT`　v0=5.1 m/s　nav=RIGHT　灯=GREEN
<sub>2021.10.05.06.57.40_veh-50_00025_00261 · token `c169245e57215ae8`</sub>

```
q1/q2 最近 3 个对象（共 12）
    Vehicle    front right   (   5.7,  -6.3) m    0.0 m/s
    Pedestrian side right    (  -2.4, -12.7) m    1.2 m/s
    Vehicle    front right   (  14.6,  -8.7) m    1.4 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/none_10_CAM_F0.jpg) | ![](image_v6/none_10_CAM_L0.jpg) | ![](image_v6/none_10_CAM_R0.jpg) | ![](image_v6/none_10_CAM_B0.jpg) |

**生成的 reasoning:**

> A pedestrian 31 meters in front of me will move into my path. I should hold my current speed and bear right: <PLAN>KEEP,RIGHT</PLAN>

### 11. `ACCELERATE,STRAIGHT`　v0=0.0 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.07.09.17.06.37_veh-35_02609_05015 · token `698321857e135d10`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/none_11_CAM_F0.jpg) | ![](image_v6/none_11_CAM_L0.jpg) | ![](image_v6/none_11_CAM_R0.jpg) | ![](image_v6/none_11_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 12 meters ahead in my lane is moving slowly. I should speed up: <PLAN>ACCELERATE,STRAIGHT</PLAN>

### 12. `KEEP,STRAIGHT`　v0=3.6 m/s　nav=RIGHT　灯=无记录
<sub>2021.10.11.02.57.41_veh-50_01522_02088 · token `e0102b44f3d45baa`</sub>

```
q1/q2 最近 3 个对象（共 4）
    Vehicle    side left     (  -1.4,  19.1) m    0.0 m/s
    Pedestrian side left     (  -0.1,  19.3) m    0.0 m/s
    Vehicle    rear right    ( -10.6, -18.4) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/none_12_CAM_F0.jpg) | ![](image_v6/none_12_CAM_L0.jpg) | ![](image_v6/none_12_CAM_R0.jpg) | ![](image_v6/none_12_CAM_B0.jpg) |

**生成的 reasoning:**

> I should hold my current speed: <PLAN>KEEP,STRAIGHT</PLAN>

### 13. `KEEP,RIGHT`　v0=2.6 m/s　nav=RIGHT　灯=RED
<sub>2021.07.16.16.27.22_veh-26_01536_02260 · token `790c30ef2b5354ea`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/none_13_CAM_F0.jpg) | ![](image_v6/none_13_CAM_L0.jpg) | ![](image_v6/none_13_CAM_R0.jpg) | ![](image_v6/none_13_CAM_B0.jpg) |

**生成的 reasoning:**

> I see pedestrians 10 meters ahead of me. I should hold my current speed and bear right: <PLAN>KEEP,RIGHT</PLAN>

### 14. `KEEP,STRAIGHT`　v0=6.2 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.08.31.17.42.52_veh-40_00551_00680 · token `d68f1142da63586f`</sub>

```
q1/q2 最近 3 个对象（共 21）
    Vehicle    side right    (  -2.0,  -6.6) m    0.0 m/s
    Vehicle    side left     (   0.9,  11.8) m    0.0 m/s
    Vehicle    front right   (  10.2,  -7.0) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/none_14_CAM_F0.jpg) | ![](image_v6/none_14_CAM_L0.jpg) | ![](image_v6/none_14_CAM_R0.jpg) | ![](image_v6/none_14_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 26 meters ahead in my lane is moving at 6 m/s. I should hold my current speed: <PLAN>KEEP,STRAIGHT</PLAN>

### 15. `ACCELERATE,STRAIGHT`　v0=2.3 m/s　nav=STRAIGHT　灯=GREEN
<sub>2021.07.09.20.59.12_veh-38_00773_01187 · token `8a8216279baa543f`</sub>

```
q1/q2 最近 3 个对象（共 19）
    Vehicle    side right    (   0.9,  -3.1) m    0.9 m/s
    Vehicle    side left     (  -0.0,   3.9) m    4.4 m/s
    Vehicle    rear right    (  -6.4,  -3.2) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/none_15_CAM_F0.jpg) | ![](image_v6/none_15_CAM_L0.jpg) | ![](image_v6/none_15_CAM_R0.jpg) | ![](image_v6/none_15_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 12 meters ahead in my lane is moving at 6 m/s. I should speed up: <PLAN>ACCELERATE,STRAIGHT</PLAN>

### 16. `ACCELERATE,RIGHT`　v0=1.9 m/s　nav=RIGHT　灯=无记录
<sub>2021.08.31.11.47.30_veh-40_01146_01347 · token `d6090ad6588b55d9`</sub>

```
q1/q2 最近 3 个对象（共 13）
    Vehicle    side left     (   1.1,   4.8) m    0.0 m/s
    Pedestrian rear right    (  -5.2,  -5.0) m    0.1 m/s
    Vehicle    rear left     (  -5.4,   4.9) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/none_16_CAM_F0.jpg) | ![](image_v6/none_16_CAM_L0.jpg) | ![](image_v6/none_16_CAM_R0.jpg) | ![](image_v6/none_16_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 15 meters ahead in my lane is moving at 3 m/s. I should speed up and bear right: <PLAN>ACCELERATE,RIGHT</PLAN>

### 17. `KEEP,LEFT`　v0=4.7 m/s　nav=LEFT　灯=无记录
<sub>2021.06.14.16.48.02_veh-12_04783_04967 · token `1995d6c8a79f58e0`</sub>

```
q1/q2 最近 3 个对象（共 8）
    Vehicle    side right    (  -1.2,  -7.5) m    4.2 m/s
    Vehicle    side right    (   4.5, -12.3) m    0.4 m/s
    Vehicle    rear left     ( -11.6,   8.4) m    0.6 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/none_17_CAM_F0.jpg) | ![](image_v6/none_17_CAM_L0.jpg) | ![](image_v6/none_17_CAM_R0.jpg) | ![](image_v6/none_17_CAM_B0.jpg) |

**生成的 reasoning:**

> I should hold my current speed and bear left: <PLAN>KEEP,LEFT</PLAN>

### 18. `ACCELERATE,STRAIGHT`　v0=1.5 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.06.14.14.03.45_veh-38_04499_05170 · token `9f9779313ad85564`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/none_18_CAM_F0.jpg) | ![](image_v6/none_18_CAM_L0.jpg) | ![](image_v6/none_18_CAM_R0.jpg) | ![](image_v6/none_18_CAM_B0.jpg) |

**生成的 reasoning:**

> I see a pedestrian 18 meters ahead of me. The vehicle 13 meters ahead in my lane is moving at 2 m/s. I should speed up: <PLAN>ACCELERATE,STRAIGHT</PLAN>

### 19. `KEEP,STRAIGHT`　v0=10.5 m/s　nav=STRAIGHT　灯=GREEN
<sub>2021.07.09.23.23.48_veh-26_01454_02217 · token `0fd2f05b7c165e51`</sub>

```
q1/q2 最近 3 个对象（共 30）
    Pedestrian side right    (   2.5,  -8.6) m    1.0 m/s
    Pedestrian side right    (   0.6, -10.1) m    1.3 m/s
    Pedestrian side right    (   3.4,  -9.5) m    1.3 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/none_19_CAM_F0.jpg) | ![](image_v6/none_19_CAM_L0.jpg) | ![](image_v6/none_19_CAM_R0.jpg) | ![](image_v6/none_19_CAM_B0.jpg) |

**生成的 reasoning:**

> I should hold my current speed: <PLAN>KEEP,STRAIGHT</PLAN>

### 20. `ACCELERATE,LEFT`　v0=5.0 m/s　nav=LEFT　灯=无记录
<sub>2021.08.24.18.07.48_veh-45_00873_01142 · token `9862524c29ec5b4e`</sub>

```
q1/q2 最近 3 个对象（共 13）
    Vehicle    side left     (   2.9,  15.0) m    0.0 m/s
    Vehicle    rear left     ( -15.0,   6.5) m    6.7 m/s
    Pedestrian front center  (  18.1,   2.1) m    1.4 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/none_20_CAM_F0.jpg) | ![](image_v6/none_20_CAM_L0.jpg) | ![](image_v6/none_20_CAM_R0.jpg) | ![](image_v6/none_20_CAM_B0.jpg) |

**生成的 reasoning:**

> I see a pedestrian 18 meters ahead of me. I should speed up and bear left: <PLAN>ACCELERATE,LEFT</PLAN>

### 21. `KEEP,STRAIGHT`　v0=8.1 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.08.17.16.48.45_veh-43_02070_02652 · token `711cedac1b4f594e`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/none_21_CAM_F0.jpg) | ![](image_v6/none_21_CAM_L0.jpg) | ![](image_v6/none_21_CAM_R0.jpg) | ![](image_v6/none_21_CAM_B0.jpg) |

**生成的 reasoning:**

> I should hold my current speed: <PLAN>KEEP,STRAIGHT</PLAN>

### 22. `KEEP,LEFT`　v0=6.3 m/s　nav=LEFT　灯=无记录
<sub>2021.08.31.14.40.58_veh-40_00016_00084 · token `afe0ef0cd35b57f7`</sub>

```
q1/q2 最近 3 个对象（共 26）
    Vehicle    side right    (   3.8,  -8.7) m    0.0 m/s
    Vehicle    side right    (   1.5, -10.0) m    0.0 m/s
    Vehicle    front right   (   6.4,  -8.0) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/none_22_CAM_F0.jpg) | ![](image_v6/none_22_CAM_L0.jpg) | ![](image_v6/none_22_CAM_R0.jpg) | ![](image_v6/none_22_CAM_B0.jpg) |

**生成的 reasoning:**

> A vehicle 23 meters in front of me will move into my path. I should hold my current speed and bear left: <PLAN>KEEP,LEFT</PLAN>

### 23. `KEEP,STRAIGHT`　v0=9.1 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.09.15.15.02.19_veh-39_01107_01666 · token `422e82ae15ff56a0`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/none_23_CAM_F0.jpg) | ![](image_v6/none_23_CAM_L0.jpg) | ![](image_v6/none_23_CAM_R0.jpg) | ![](image_v6/none_23_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 10 meters ahead in my lane is stopped. I should hold my current speed: <PLAN>KEEP,STRAIGHT</PLAN>

### 24. `ACCELERATE,LEFT`　v0=6.1 m/s　nav=LEFT　灯=RED
<sub>2021.06.09.18.23.43_veh-35_03609_03793 · token `b393873cd3e95ecf`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/none_24_CAM_F0.jpg) | ![](image_v6/none_24_CAM_L0.jpg) | ![](image_v6/none_24_CAM_R0.jpg) | ![](image_v6/none_24_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 12 meters ahead in my lane is moving at 5 m/s. I should speed up and bear left: <PLAN>ACCELERATE,LEFT</PLAN>

### 25. `ACCELERATE,STRAIGHT`　v0=7.4 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.09.14.18.43.41_veh-45_02296_02477 · token `acea6047685c5388`</sub>

```
q1/q2 最近 3 个对象（共 2）
    Vehicle    rear left     ( -19.5,   8.1) m    2.7 m/s
    Vehicle    front left    (  19.9,   7.5) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/none_25_CAM_F0.jpg) | ![](image_v6/none_25_CAM_L0.jpg) | ![](image_v6/none_25_CAM_R0.jpg) | ![](image_v6/none_25_CAM_B0.jpg) |

**生成的 reasoning:**

> I should speed up: <PLAN>ACCELERATE,STRAIGHT</PLAN>

### 26. `ACCELERATE,STRAIGHT`　v0=8.4 m/s　nav=STRAIGHT　灯=GREEN
<sub>2021.06.09.18.23.43_veh-35_01232_01405 · token `8bee9023fbb8550c`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/none_26_CAM_F0.jpg) | ![](image_v6/none_26_CAM_L0.jpg) | ![](image_v6/none_26_CAM_R0.jpg) | ![](image_v6/none_26_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 9 meters ahead in my lane is moving at 8 m/s. I should speed up: <PLAN>ACCELERATE,STRAIGHT</PLAN>

### 27. `KEEP,STRAIGHT`　v0=6.1 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.08.31.13.27.52_veh-40_00869_01319 · token `84e6b3e1380256a4`</sub>

```
q1/q2 最近 3 个对象（共 5）
    Pedestrian side right    (   2.2,  -9.7) m    0.0 m/s
    Vehicle    front center  (  20.4,   1.4) m    8.1 m/s
    Cyclist    rear left     ( -32.6,   8.2) m    6.1 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/none_27_CAM_F0.jpg) | ![](image_v6/none_27_CAM_L0.jpg) | ![](image_v6/none_27_CAM_R0.jpg) | ![](image_v6/none_27_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 20 meters ahead in my lane is moving at 8 m/s. I should hold my current speed: <PLAN>KEEP,STRAIGHT</PLAN>

### 28. `KEEP,STRAIGHT`　v0=10.3 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.08.17.18.54.02_veh-45_02202_02416 · token `2e4e674b30e45fb8`</sub>

```
q1/q2 最近 3 个对象（共 2）
    Vehicle    rear right    ( -24.9,  -2.5) m   10.5 m/s
    Vehicle    rear right    ( -38.0,  -5.2) m   10.1 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/none_28_CAM_F0.jpg) | ![](image_v6/none_28_CAM_L0.jpg) | ![](image_v6/none_28_CAM_R0.jpg) | ![](image_v6/none_28_CAM_B0.jpg) |

**生成的 reasoning:**

> I should hold my current speed: <PLAN>KEEP,STRAIGHT</PLAN>

### 29. `KEEP,STRAIGHT`　v0=9.9 m/s　nav=STRAIGHT　灯=无记录
<sub>2021.09.15.14.27.22_veh-39_01491_01763 · token `7af4d8afa7325033`</sub>

```
q1/q2 最近 3 个对象（共 0）
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/none_29_CAM_F0.jpg) | ![](image_v6/none_29_CAM_L0.jpg) | ![](image_v6/none_29_CAM_R0.jpg) | ![](image_v6/none_29_CAM_B0.jpg) |

**生成的 reasoning:**

> The cyclist 29 meters ahead in my lane is moving at 8 m/s. I should hold my current speed: <PLAN>KEEP,STRAIGHT</PLAN>

### 30. `ACCELERATE,LEFT`　v0=6.8 m/s　nav=LEFT　灯=RED
<sub>2021.06.09.11.54.15_veh-12_01705_01845 · token `472ee2754def56fe`</sub>

```
q1/q2 最近 3 个对象（共 21）
    Vehicle    rear left     ( -12.1,   5.2) m    6.0 m/s
    Vehicle    front center  (  18.6,   1.7) m    9.3 m/s
    Vehicle    front left    (  12.0,  15.7) m    0.0 m/s
```

| 前视 CAM_F0 | 左前 CAM_L0 | 右前 CAM_R0 | 后视 CAM_B0 |
|---|---|---|---|
| ![](image_v6/none_30_CAM_F0.jpg) | ![](image_v6/none_30_CAM_L0.jpg) | ![](image_v6/none_30_CAM_R0.jpg) | ![](image_v6/none_30_CAM_B0.jpg) |

**生成的 reasoning:**

> The vehicle 18 meters ahead in my lane is moving at 9 m/s. I should speed up and bear left: <PLAN>ACCELERATE,LEFT</PLAN>

---

