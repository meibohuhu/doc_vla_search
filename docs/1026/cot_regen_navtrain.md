# D0：重新生成 navtrain 103k 的 CoT

2026-10-27。计划：[cf_sft_autovla_plan.md](cf_sft_autovla_plan.md) §3 D0。

## 为什么要重新生成

103k 的 CoT（`navtrain_cot`）和它的生成器 `build_navsim_cot.py` 都只在 brev 上，brev 不再用了。HF 上只有 166k 版本（`PhoenixHu/doc_vla_nuplan_166k_cot`），它和 navtrain 只重合 25,609 个 token。所以照 166k 的输出反推规则，重写了生成器。

## 代码

在 Impromptu-VLA 仓库（`/home/mh2803/vla/Impromptu-VLA`，原版也在这个位置）：

- `data_qa_generate/data_engine/datasets/navsim/build_navsim_cot.py`：读 `navsim_logs` 和地图，输出 `{token: cot}`。规则写在文件头。
- `data_qa_generate/data_engine/datasets/navsim/merge_cot_into_samples.py`：灌进样本 JSON 的 `cot_output`，相机路径改成相对 `sensor_blobs/`。

`<PLAN>` 用 `doc_vla_search` 的 `plan_from_track`（和训练侧同一个函数），所以需要 `DOC_VLA_REPO` 指向本仓库（默认 `/home/mh2803/vla/doc_vla_search`）。

```bash
conda activate autovla_codeclean
export PYTHONPATH=/home/mh2803/vla/doc_vla_search/navsim NUPLAN_MAPS_ROOT=/scratch/mh2803/vla/nuplan/maps NUPLAN_MAP_VERSION=nuplan-maps-v1.0
cd /home/mh2803/vla/Impromptu-VLA/data_qa_generate/data_engine/datasets/navsim
python build_navsim_cot.py --json_dir /scratch/mh2803/vla/nuplan/navtrain_nocot --out cot_navtrain.json --workers 68
python merge_cot_into_samples.py --src /scratch/mh2803/vla/nuplan/navtrain_nocot --cot cot_navtrain.json \
    --dst /scratch/mh2803/vla/nuplan/navtrain_cot --sensor_root /scratch/mh2803/vla/nuplan/sensor_blobs
```

navtrain 生成 47 秒、val 4 秒（68 进程）。

## 输出

| | 样本数 | 有 CoT | 空（弃权）|
|---|---|---|---|
| `/scratch/mh2803/vla/nuplan/navtrain_cot` | 101,288 | **100,222**（98.9%）| 1,066 |
| `/scratch/mh2803/vla/nuplan/navtrain_cot_val` | 2,000 | 1,977（98.8%）| 23 |

- 有 CoT 的条数 100,222 和老文档里 `cot_navtrain.json` 的条数**完全一样**，说明弃权规则复现对了。
- `<PLAN>` 和 `gt_trajectory` 算出来的全部一致；token 一一对应。
- 中间结果 `{token: cot}` 在 `/scratch/mh2803/vla/nuplan/cf_navtrain_v1/cot/`。
- 🔴 **相机路径现在是相对路径**（`trainval/<log>/CAM_F0/xxx.jpg`）。训练 config 的 `sensor_data_path` 要设成各机器的 `sensor_blobs` 目录，不能再是 `null`。

子模式分布（navtrain，不含弃权）：

| | 老 103k（`logs/0901_nvidia/navsim_reasoning_bytype_v6.md`）| 新 |
|---|---|---|
| none | 67.6% | 67.5% |
| stay_behind（跟车）| 29.5% | 29.5% |
| stop_sign | 1.0% | 1.6% |
| cross_path（切入）| 1.0% | 0.6% |
| yield_vru（让行）| 0.5% | 0.5% |
| gap_big（前方放空）| 0.4% | 0.3% |

停车标志比老版多，是因为改成停车线多边形和路径线相交以后，斜的大停车线也能判到；抽查过，新多出来的都对。切入比老版少，是因为老版里停着不动的车不再算。

## 怎么反推的、和 166k 差多少

把 166k 的句子拆成槽（VRU / 前车 / 切入 / 理由 / 决策），从 `navsim_logs` 取同一帧的他车和地图，逐槽找规则。

和 166k 一致的部分：

| 部分 | 规则 | 一致率（60 个 log，8,621 帧）|
|---|---|---|
| 弃权 | t=0 往前 3 帧、往后 10 帧中间缺帧（时间戳跳 1 s）| 269 / 270 |
| `<PLAN>` 和动词 | `plan_from_track` + `teacher.speed_verb` | 99.99% |
| 前车句 | 正前 5–40 m、横向 < 4 m、同向（车 < 90°，行人 < 60°）里 x 最小的；距离 int(x)；< 0.3 m/s stopped，< 2 m/s moving slowly | 98.3% |
| VRU 句 | 同一条走廊、20 m 内；前车是 VRU 时不再写前车句 | 99.5% |
| 跟车 / 让行 | 只在 STOP / DECELERATE 给理由：有前车 → 跟车，只有 VRU → 让行 | |

反推不出来、重新定义的三类（用户 10-27 选的方案 B）：

| 类 | 老版的问题 | 新定义 |
|---|---|---|
| 切入 | 不少"will move into my path"的车停着不动、5 s 内没动过；也不是按朝向延长线算的 | t=0 在自车路径带外（> 2 m），3 s 内真的开进路径带（< 1.5 m），进入点在自车前面、前车后面，他车速度 ≥ 0.5 m/s |
| 停车标志 | 前车挡着时也写停车标志 | STOP_SIGN 停车线多边形和自车路径线相交（不看中心点）、前方 40 m 内，而且前车不在它前面 |
| 前方放空 | 有的车距其实在缩小 | 前车 ≥ 25 m、2.5 s 内提速 ≥ 1 m/s、2 s 后车距确实变大 |

自车路径带 = GT 路径沿 GT 末点航向直线延长到 45 m，两侧 1.5 / 2.0 m（延长方向不能用最后一段位移，快停下时那是噪声）。理由优先级：停车标志 > 让行（无前车时）> 切入 > 跟车。切入的人已经在 VRU 句里时不再单独写切入句。

结果：独立的 100 个 log（13,400 帧）整句一致 90.2%；和 navtrain 重合的 25,609 个 token 整句一致 89.5%、理由一致 98.1%、`<PLAN>` 一致 99.98%。

## 例子

带图的示例（每个子模式 1–2 张，加新老不一致的几组）见 [cot_regen_examples/](cot_regen_examples/README.md)。

```
none        I should hold my current speed: <PLAN>KEEP,STRAIGHT</PLAN>
none        The vehicle 6 meters ahead in my lane is moving at 2 m/s. I should speed up: <PLAN>ACCELERATE,STRAIGHT</PLAN>
跟车        The vehicle 10 meters ahead in my lane is stopped. Because I need to stay behind it, I should remain stopped: <PLAN>STOP,STRAIGHT</PLAN>
让行        I see a pedestrian 11 meters ahead of me. Because I need to yield to them, I should come to a stop: <PLAN>STOP,STRAIGHT</PLAN>
切入        Because the vehicle on my left will move into my path, I should remain stopped: <PLAN>STOP,STRAIGHT</PLAN>
切入(观察)  A vehicle 10 meters on my left will move into my path. I should speed up: <PLAN>ACCELERATE,STRAIGHT</PLAN>
停车标志    A vehicle 25 meters on my right will move into my path. Because there is a stop sign ahead, I should come to a stop: <PLAN>STOP,STRAIGHT</PLAN>
放空        The vehicle 25 meters ahead in my lane is moving at 9 m/s. Because the gap in front of me is opening up, I should hold my current speed: <PLAN>KEEP,STRAIGHT</PLAN>
```

## 照搬下来的老毛病

- "in my lane" 的走廊是横向 < 4 m，旁边车道停着的车也会被写成在我车道里。跟车占 29.5%，改它等于改大部分数据，所以没动。
- "on my center" 语法不对，照 166k 保留。
- 红灯理由老版就砍掉了（灯态不可靠），这里也没有。
