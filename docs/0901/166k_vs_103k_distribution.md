# 166k vs 103k(navtrain):数据分布与重合分析

日期 2026-09-05 · 相关 [nuplan_103k_to_166k.md](nuplan_103k_to_166k.md)(两者来源)· 数据都在 `/data/autovla_data/nuplan/`

> 结论先行:**166k 不是"navtrain 加量版"** —— 两者只重合 15%,且 166k 多出来的帧几乎全是
> STOP/匀速这类被动帧,真正有信息的机动帧(ACCEL/DECEL)**绝对数量反而比 navtrain 还少**。
> 直接用 166k 会掉分(实测)。正确用法是 **mix:navtrain + 从 166k 按 navtrain 分布采样**。

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
