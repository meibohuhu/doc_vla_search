# AutoVLA 环境移植到 aarch64 (GH200) —— 推导记录

日期 2026-09-03 · 机器 `tigris.rc.rit.edu` (RIT RC) · 产出:
[environment_aarch64.yml](../../environment_aarch64.yml) ·
[requirements-aarch64.txt](../../requirements-aarch64.txt) ·
[scripts/install_aarch64.sh](../../scripts/install_aarch64.sh)

> 结论先说:**只有 torch 栈真的动了**(`torch 2.5.1+cu124` / `torchvision 0.20.1`,
> `triton` 删除)。其余 pin 全部保持与 `requirements.txt` 一致 —— 有 4 个改从
> conda-forge 装,但**版本号一模一样**。

---

## 0. 为什么需要这份东西

`conda` 在本机一调用就报错:

```
/home/mh2803/miniconda3/bin/conda: line 6: syntax error near unexpected token `sys.argv'
```

不是 conda 坏了,是**架构不对**:

```
$ uname -m                                          → aarch64
$ file ~/miniconda3/bin/python3.13                  → ELF 64-bit LSB executable, x86-64
$ ~/miniconda3/bin/python -V                        → cannot execute binary file: Exec format error
```

`~/miniconda3` 是 x86_64 装的(装它的那台不是这台)。shebang 指向的 python 跑不了,
bash 回退成"当 shell 脚本执行",于是 `import sys` 被当成 ImageMagick 的 `import` ——
那句 `unable to open X server` 就是这么来的。

**整个集群都是 aarch64**,没有 x86 节点可退:

```
$ sinfo -o "%20N %10G"
gh-b-001     gpu:gh200:...          # Grace-Hopper
gg-a-[001-036]  (null)
gh-a-[001-...]  gpu:gh200:...
$ module avail                      # → linux-rhel9-neoverse_v2, cuda/12.0.0, cuda/12.3.0
```

→ 另装 **Miniforge3-Linux-aarch64** 到 `~/miniforge3`(旧的 `~/miniconda3` 没动)。
注意用户 shell 里的 `conda` 函数还指向坏掉的那个,脚本里要 `unset -f conda`,
光改 `PATH` 不管用。

---

## 1. torch 栈 —— 唯一真正的版本改动

### PyPI 上的 aarch64 torch 是 CPU-only

`requirements.txt` 钉 `torch==2.4.0`。PyPI 确实有 cp39 的 aarch64 wheel,但:

| version | cp39 aarch64 wheel | 大小 |
|---|---|---|
| 2.4.0 | `manylinux2014_aarch64` | **90 MB** |
| 2.5.1 | `manylinux2014_aarch64` | 92 MB |
| 2.8.0 | `manylinux_2_28_aarch64` | 102 MB |

90–100MB = **CPU-only**(CUDA 版是 GB 级)。GH200 要的 CUDA aarch64 wheel 在
`download.pytorch.org/whl/cu124` 的 `linux_aarch64` (sbsa) 分支。

### cu124 上 cp39 能拿到什么

```
torch-2.4.1-cp39-cp39-linux_aarch64.whl
torch-2.5.0-cp39-cp39-linux_aarch64.whl
torch-2.5.1-cp39-cp39-linux_aarch64.whl      ← 选它
torchvision-0.20.1-cp39-cp39-linux_aarch64.whl
```

- **选 2.5.1 而不是更接近 pin 的 2.4.1**:2.4.1 的配套是 torchvision 0.19.x,
  而 cu124 上 torchvision **没有 cp39 aarch64 的 0.19.x**(只有 0.16/0.17,然后跳到 0.20)。
  `torch 2.5.1 + torchvision 0.20.1` 才是自洽的一对。
- cu126 / cu128 的 aarch64 只到 cp310 起,cp39 用不上。CUDA 12.4 支持 sm_90(Hopper),够用。

### ⚠️ 必须用 `--index-url`,不能用 `--extra-index-url`

PyPI 的 CPU-only wheel 和 cu124 的 CUDA wheel **版本号完全相同**(都叫 `2.5.1`,
cu124 的 aarch64 wheel 没有 `+cu124` local version)。`--extra-index-url` 不保证
优先级 → pip 可能静默装上 CPU-only 版,之后 `torch.cuda.is_available()` 是 `False`,
排查起来很费时间。

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1 torchvision==0.20.1
python -c "import torch; assert torch.version.cuda"   # 装完立刻断言
```

### triton 删掉

cp39 + aarch64 在**任何** index 都没有 triton(cu124 上 triton 只有 cp312+,
版本 3.7.1/3.8.0)。影响:`torch.compile` / Inductor 不可用;eager + SDPA 正常。
本仓库没有任何地方调 `torch.compile`。

---

## 2. 4 个包改从 conda-forge 装（版本不变）

这 4 个在 PyPI 的 **cp39/aarch64 只有 sdist**,pip 装会现场编译 GDAL / Rust / OpenCV
(慢且极易失败)。而 conda-forge 的 `linux-aarch64` 恰好有**同版本**的预编译包:

| requirements.txt | conda-forge linux-aarch64 py39 | PyPI cp39/aarch64 情况 |
|---|---|---|
| `opencv-python==4.9.0.80` | `py-opencv=4.9.0` ✅ | 只到 4.5.4.60 |
| `rasterio==1.3.11` | `rasterio=1.3.11` ✅ | **任何版本都没有** |
| `fiona==1.10.1` | `fiona=1.10.1` ✅ | **任何版本都没有** |
| `safetensors==0.5.3` | `safetensors=0.5.3` ✅ | 只到 0.4.5 |
| `numpy=1.23.4` (environment.yml) | `numpy=1.23.4` ✅ | — |

pin 组合是可解的 —— `py-opencv 4.9.0` 要 `numpy >=1.22.4,<2.0a0`,`rasterio 1.3.11`
要 `numpy >=1.19,<3`,`1.23.4` 都满足。

其余 pin(`pyproj` `shapely` `pyogrio` `rtree` `casadi` `ray` `numexpr` `scikit-learn`
`tokenizers` `av` `guppy3` …)PyPI 都有 cp39 aarch64 wheel,原样 pip 装。

---

## 3. install.sh 里本平台装不上的 3 个

都**不影响 166k nocot SFT 主链路**:

| 包 | 情况 | 影响面 |
|---|---|---|
| `flash-attn==2.7.4.post1` | 无 aarch64 wheel,需 nvcc 源码编译(得在 GPU 节点上跑,~1–2h) | 见下 |
| `waymo-open-dataset-tf-2-12-0==1.6.7` | 仅 x86_64 | 只影响 waymo 预处理 |
| `autoawq==0.2.8` | 仅 x86_64 | 只影响 AWQ 量化推理 |

**flash-attn 的实际暴露面比看起来小**:训练侧
[models/autovla.py](../../models/autovla.py) 的 `load_vlm()` **没有**指定
`attn_implementation`,transformers 默认走 sdpa → SFT/RFT 训练不受影响。
全仓库只有一处写死 `flash_attention_2`:
[dataset_utils/preprocessing/cot_annotation_model.py](../../dataset_utils/preprocessing/cot_annotation_model.py)
(CoT 标注模型)。而 166k 走的是 `nocot_sample_generation --fast`,不碰它。
真要跑 CoT 标注,把那处改成 `"sdpa"` 即可。

---

## 3.5 两个必须知道的坑（不修就是坏环境）

### (a) pip 会把 CUDA torch 换成 CPU-only 版

`requirements-aarch64.txt` 里 **必须留着** `torch==2.5.1` / `torchvision==0.20.1` 两行 pin。

一开始为了"别让 pip 从 PyPI 重装 torch"把这两行删了 —— 结果 `accelerate==1.5.2`
要求 `torch>=2.0.0`,没有 pin 约束,resolver 直接从 PyPI 升到
`torch-2.8.0-cp39-cp39-manylinux_2_28_aarch64.whl` (102MB,**CPU-only**),
把前面用 cu124 装好的 CUDA 版**静默覆盖**掉了。

留着 pin 反而安全:已装的就是 2.5.1,`==2.5.1` 已满足 → pip 跳过不下载;
同时 `accelerate` 的 `torch>=2.0.0` 也被满足,resolver 没有升级空间。

顺带:`requirements.txt` 钉 `sympy==1.13.3`,但 **torch 2.5.1 硬性要求
`sympy==1.13.1`**(2.4.0 对 sympy 没有上下界)。不改会 `ResolutionImpossible`。

装完务必断言一次:

```bash
python -c "import torch; assert torch.version.cuda, 'CPU-only wheel!'; print(torch.version.cuda)"
```

### (b) `import torch` 之后 cv2 / gdal / geos 全部 ImportError

现象:

```
ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.30' not found
  (required by .../libopencv_gapi.so.409)
```

而**单独** `import cv2` 是好的,`import torch` 之后才坏 —— 纯粹的**加载顺序**问题:

- RHEL9 系统 `/usr/lib64/libstdc++.so.6.0.29` 最高 `GLIBCXX_3.4.29`
- conda-forge 的 opencv/gdal/geos 需要 `3.4.30`~`3.4.32`
- env 里的 `libstdc++.so.6.0.36` 提供到 `3.4.36`,**够用**,而且
  `libopencv_gapi.so.409` / `libgdal.so.35` / `libgeos_c.so.1` 的
  RPATH 都是 `$ORIGIN/.`(就是 env 的 `lib/`),**指得也对**
- 但 `import torch` 会**先**把系统那份载进进程(torch 的 .so 没有指向 env lib 的
  RPATH,libstdc++ 走 ldconfig 默认路径解析到 `/usr/lib64`)。之后同 SONAME
  `libstdc++.so.6` 的库直接复用**已加载**的旧版 → 报错。

修法:activate 钩子把 `$CONDA_PREFIX/lib` 放到 `LD_LIBRARY_PATH` 最前面
(`step_ldfix`,已写进 `$CONDA_PREFIX/etc/conda/activate.d/zz-libstdcxx.sh`)。
之后 `conda activate auto_vla` 自动生效。不会遮蔽 torch 自带的 nvidia 库
——那些在 `site-packages/nvidia/*/lib`,文件名不同。

> 附带一个**无害**的噪音:GDAL 导入时会打
> `ERROR 1: libpoppler.so.140: cannot open shared object file`。
> 那是 GDAL 的 PDF 驱动插件缺 poppler,nuPlan 用不到这个驱动,不影响功能。
> 想消掉就 `conda install -c conda-forge poppler`。

## 4. 踩过的坑

- **别在 `conda create` 中途 SIGTERM**。第一次因为 120s 超时被打断,pkgs 缓存里
  留下半解压的 python 包,之后重试一直报
  `CondaVerificationError: ... appears to be corrupted`。得
  `rm -rf pkgs/python-3.9.23-*` + `conda clean --all` 才能恢复(而 `clean --all`
  在 Ceph 上要跑十几分钟)。
- **`/home` 和 `/shared` 挂同一个 Ceph MDS**。同时跑 166k sensor 解压
  (几百万个小 jpg)时,conda 往 `/home` 写小文件会被饿死 —— `conda create` 卡在
  `ceph_mdsc_wait_request` 9 分钟一个文件都没写出来。一暂停下载,
  site-packages 立刻从 10 涨到 159。**装环境和下数据别同时跑。**
- **判断 env 好没好别用 `[ -x bin/python ]`**。python 二进制会在 stdlib 写完**之前**
  就出现,那时跑它报
  `init_fs_encoding: failed to get the Python codec of the filesystem encoding`。
  要等 `conda` 进程自己退出。
