# 166k vs 103k(navtrain):数据分布与重合分析

日期 2026-09-05 · 相关 [nuplan_103k_to_166k.md](nuplan_103k_to_166k.md)(两者来源)· 数据都在 `/data/autovla_data/nuplan/`

> 结论先行:**166k 不是"navtrain 加量版"** —— 两者只重合 15%,且 166k 多出来的帧几乎全是
> STOP/匀速这类被动帧,真正有信息的机动帧(ACCEL/DECEL)**绝对数量反而比 navtrain 还少**。
> 直接用 166k 会掉分(实测)。正确用法是 **mix:navtrain + 从 166k 按 navtrain 分布采样**。
>
> **[2026-09-27 更新]** §5 推荐的 mix(按 CoT 标签分布采样 53k)实测**也没涨**(no-CoT 79.90 vs 103k 80.63)。
> 根因:navtrain/navtest 是 NAVSIM 同一套筛选的产物(几乎无真停帧、无 unknown 导航、转弯 ~34%),
> 而 **CoT 标签的 STOP ≠ 真停**,按 CoT 标签对齐分布对不上这套筛选。改用 GT 运动学过滤后的新集见 **§6**。

---

## 1. 规模与重合

| | 数量 | scene_filter |
|---|---|---|
| **navtrain_cot**(官方 103k,train) | 101,288(+val 2,000) | `navtrain.yaml`:fi=1,has_route,**手工 token 白名单**,1192 log |
| **166k**(trainval_cot_166k 全量) | 166,282 | 默认 filter:**fi=4**,不限 log,**无白名单**,1310 log |

**token 重合(实测):**

| | 数量 | 占比 |
|---|---|---|
| 交集(两者都有) | **25,430** | 占 166k 的 15%、占 navtrain 的 25% |
| navtrain 独有(166k 没有) | **75,858** | —— |
| 166k 独有(navtrain 没有) | **140,852** | —— |

**⇒ 谁都不是谁的子集。** navtrain 有 75,858 个 token 是 166k 完全没有的。
根因:`frame_interval` 1 vs 4 → 取的"中心帧"网格不同,大部分帧对不上;加上 navtrain 是精选白名单、
166k 是机械全铺 → 只有 15% 重合。

---

## 2. 速度决策分布(核心差异)

从 `cot_output` 的 `<PLAN>SPEED,PATH</PLAN>` 解析,抽样实测:

| 速度决策 | navtrain 103k | 166k 原始 |
|---|---:|---:|
| **STOP**(停住) | 19.4% | **44.1%** |
| KEEP(匀速) | 34.3% | 32.8% |
| **ACCELERATE** | **27.4%** | 13.2% |
| **DECELERATE** | **17.9%** | 8.5% |
| EMPTY(无 CoT) | 1.0% | 1.3% |

**166k 有一半是 STOP(等灯/停车/堵车/排队),机动帧(ACCEL/DECEL)只占 ~22%;navtrain 只有 19% STOP、机动帧占 ~45%。**

### ⚠️ 换成绝对条数 —— 关键洞察

| | navtrain 103k | 166k 原始 | 谁多 |
|---|---:|---:|---|
| ACCELERATE | ~27,800 | ~21,900 | **navtrain 更多** |
| DECELERATE | ~18,100 | ~14,100 | **navtrain 更多** |
| STOP | ~19,600 | ~73,000 | 166k(废帧) |
| KEEP | ~34,700 | ~54,000 | 166k |

**166k 比 navtrain 多出来的 6 万帧,几乎全是 STOP + KEEP 被动帧;真正有信息的机动帧,166k 的绝对数量还比 navtrain 少。**
navtrain 的官方 curation 专门富集了机动帧;166k 的 fi=4 全铺捞到的绝大多数是被动帧。

---

## 3. 数据格式完全同构(mix 安全)

navtrain 与 166k **逐字段一致**(同一条 `nocot_sample_generation` 管线,只是选帧不同):

| 字段 | 两边 |
|---|---|
| 字段集合(schema) | **完全相同,零差异** |
| `gt_trajectory` | 10×3(10 pose × xyh) |
| `his_trajectory` | 4×3 |
| 各路 camera_paths | 每路 4 帧,绝对路径 |
| 图像 | 1920×1080 RGB |
| velocity/acceleration | list[2];dataset_name=`nuplan` |

跨决策类型(STOP/KEEP/ACCEL/DECEL)结构也一致。**⇒ 混训不会有格式隐患,`SFTDataset` 一视同仁。**

---

## 4. 实测后果(同口径 1000 seed=42,greedy,CoT SFT)

| ckpt | PDMS | EP |
|---|---:|---:|
| 103k CoT SFT **ep2** | **79.48** | 74.31 |
| 166k CoT SFT **ep2**(同 epoch,样本见得更多) | **74.65** | 70.68 |
| 103k CoT SFT ep4 | 80.19 | 75.79 |

**同 epoch 下 166k 低 ~4.8 分,EP(推进)掉最狠** —— STOP 帧把模型训消极了。不是欠训(166k ep2 见过的样本更多)。

---

## 5. 怎么用 166k(结论)

- ❌ **原样用 166k**:被 STOP 稀释,掉分。
- 🟡 **只 STOP 下采样**(`trainval_cot_166k_stopds`,113,561,STOP→18%):修好 STOP 轴,但机动帧补不回来
  (KEEP 偏高 49%、ACCEL/DECEL 仍低于 navtrain)—— 大概率仍打不过 navtrain。
- ✅ **mix(推荐)**:navtrain 101,288 + 从 166k-only 按 navtrain 分布采样 **53,171**(瓶颈=ACCEL,只有 14.6k)
  = **154,459**,全程 navtrain 健康分布,且机动帧比 navtrain 单用**更多**(ACCEL 27.8k+14.6k=42.4k)。
  这是唯一有希望超过 navtrain 的配置。val 用 `navtrain_cot_val`(2,000,已去泄漏)。
- ✅ **保底**:直接用官方 **navtrain 103k**(已验证 no-CoT 80.63 / CoT ep4 80.45),省心。

### 已就绪的派生集(全 symlink,不动 navtrain 原始实体)
- `trainval_cot_166k_clean`(165,786)= 166k 去 val 泄漏
- `trainval_cot_166k_stopds`(113,561)= STOP→18%
- (待建)`trainval_mix166k_add`(53,171)= 供 mix 的净新增采样

---

## 6. [2026-09-27] 复盘:为什么 166k(和 mix)都不涨 —— 按 navtest 口径重看

口径:所有判据都用 **GT 轨迹运动学 + `instruction` 字段**,不用 CoT 标签;各集同一套规则。
"真停" = 5s 位移 < 1m;"起点静止" = v0 < 0.3 m/s;"CV-easy" = 匀速外推 5s 误差 < 1m。

### 6.1 navtrain ≈ navtest,166k 不是

| | navtrain | navtest | 166k | mix_add 53k(旧) |
|---|---:|---:|---:|---:|
| 真停 | 1.5% | 1.7% | **32.6%** | **15.5%** |
| 起点静止 | 6.4% | 5.4% | **36.3%** | 19.3% |
| unknown 导航 | 0% | 0% | **6.4%** | **6.7%** |
| 转弯指令 | 37% | 34% | **18%** | **13%** |
| CV-easy | 8% | 10% | **40%** | 23% |
| 5s 位移中位 | 20m | 22m | **14m** | 29m |

(8k 抽样,seed=0。)navtrain/navtest 明显是同一套筛选的产物;166k 的 fi=4 全铺没有经过这套筛选。

- **静止帧语义相反**:navtest 里起点静止的帧 **99.6% 在 GT 里起步了**(NAVSIM 只留"停着且马上要走"的帧);
  166k 的真停帧 **95% 起点就是静止**(停车/排队/等灯)且一直不动 → 教模型"停着就继续停"。
- **导航信号被稀释**:166k 6.4% 帧 instruction=`unknown`(无 has_route 过滤),真在转弯的帧里 12% 是 unknown;转弯帧只有一半。
- **CoT 标签 STOP ≠ 真停**:navtrain 按 CoT 标签 STOP 19.4%,按 GT 只有 1.5%。§5 的 mix 按 CoT 标签配分布,
  所以 mix_add 里仍有 15.5% 真停(navtest 的 ~10 倍)、转弯只有 13%。

### 6.2 分数掉在哪(navtest 全量 12,125,按场景拆)

| 场景 | 占比 | 103k CoT ep4 | 166k CoT ep3 | Δ | 对总分贡献 | mix − 103k(no-CoT) |
|---|---:|---:|---:|---:|---:|---:|
| 起步帧(v0<0.3) | 5.5% | 93.9 | 85.8 | **−8.1** | −0.4 | −1.2 |
| 转弯指令 | 31.5% | 74.3 | 71.4 | −2.9 | −0.9 | +0.1 |
| 直行在动 | 63.0% | 82.4 | 79.0 | −3.3 | **−2.1** | −1.1 |
| 全部 | | 80.45 | 77.00 | −3.45 | | −0.73 |

起步帧单帧掉最多(EP 88.2→69.4,模型不肯起步),但总分主要掉在直行和转弯:EP、DAC 全面下降,
符合"40% 简单帧 + 位移中位只有 14m → 学到整体偏慢偏保守"。
(注:166k 用 ep3、103k 用 ep4,不同 epoch;§4 同 epoch 1k 子集的对比方向一致。)

### 6.3 新建的净新增集(GT 运动学过滤,全 symlink)

脚本:`scripts/0926/build_166k_navfilter.py`(可复现,配平采样 random_state=0)。

| 漏斗 | 帧数 |
|---|---:|
| 166k 全量 | 166,282 |
| 去 navtrain 重合 (25,430) 和 navtrain_val 泄漏 (496) | 140,356 |
| F1 去真停 | 86,868 |
| F2 去 unknown 导航 | 80,934 |
| F3 去 CV-easy → **`trainval_166k_navfilt`** | **69,065** |
| 转弯配平(转弯全留、直行下采样) → **`trainval_166k_navfilt_bal`** | **42,473** |
| 转弯上调到 40%(同上,直行下采样更多) → **`trainval_166k_navfilt_turn40`** | **19,322** |

三个集与 navtrain / navtrain_val / navtest **token 交集均为 0**;`_bal`、`_turn40` 都 ⊂ navfilt(转弯帧相同,直行各自采样,两者交集 15,514)。
链到 `trainval_cot_166k(_val)`,带 `cot_output`,CoT / no-CoT 都能直接用。token 列表在 `/data/autovla_data/nuplan/<集名>_tokens.txt`。

混合后分布(navtest 为 8k 抽样参照):

| | 帧数 | 真停 | 起点静止 | CV-easy | 转弯指令 | unknown | 5s 位移中位 | v0 中位 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| navtest(参照) | — | 1.7% | 5.4% | 9.7% | 34.0% | 0% | 22.3m | 5.0 |
| navtrain 103k | 101,288 | 1.5% | 6.5% | 8.9% | 36.6% | 0% | 20.0m | 4.4 |
| navtrain + navfilt | **170,353**(+68%) | 0.9% | 7.3% | 5.3% | 28.3% | 0% | 23.3m | 5.1 |
| navtrain + navfilt_bal | **143,761**(+42%) | 1.1% | 7.3% | 6.3% | **33.6%** | 0% | 21.6m | 4.7 |
| navtrain + navfilt_turn40 | **120,610**(+19%) | 1.3% | 7.3% | 7.5% | **40.0%** | 0% | 20.1m | 4.4 |

- **推荐先跑 `navtrain + navfilt_bal`**(+42k):各维度都贴近 navtest。
- `navtrain + navfilt_turn40`(+19k):故意多给转弯(转弯场景 PDMS 最低,§6.2),量少一半。
  launcher:`logs/0926_nvidia/run_nocot_sft_navtrain_navfiltturn40_vit_unfreeze_adamw8bit_4gpu.sh`。
- `navtrain + navfilt`(+69k)量更大,但转弯被稀释到 28%,可作为对照。
- 新增帧本身偏快(v0 中位 ~6–7 m/s),混合后被 navtrain 拉回 navtest 水平。
