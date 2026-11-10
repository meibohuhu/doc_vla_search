# tools/navsim_v2 的公共环境变量，由同目录的 .sh 脚本 source。
# 每一项都可以在命令行前面用同名变量覆盖，例如换机器：
#   V2_ROOT=/path/navsim_v2 V2_MAPS=/path/maps PY=python CKPT=... BASE=... bash tools/navsim_v2/run.sh
#
# 注意：这里故意不用全局的 OPENSCENE_DATA_ROOT / NAVSIM_EXP_ROOT 做默认值 —— 那两个在 v1 的脚本里指向
# nuplan 目录，v2 的数据和输出要分开放。

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
HERE=$REPO/tools/navsim_v2

V2_ROOT=${V2_ROOT:-/scratch/mh2803/vla/navsim_v2}          # 下面三个目录的父目录
V2_DEVKIT=${V2_DEVKIT:-$V2_ROOT/navsim-devkit}             # NAVSIM v2 官方代码（setup_data.sh clone）
V2_DATA=${V2_DATA:-$V2_ROOT/data}                          # = OPENSCENE_DATA_ROOT
V2_EXP=${V2_EXP:-$V2_ROOT/exp}                             # metric cache、推理结果、打分 csv
V2_MAPS=${V2_MAPS:-/scratch/mh2803/vla/nuplan/maps}        # nuPlan 地图，v1 / v2 通用
V2_COMMIT=${V2_COMMIT:-0a380a9}                            # 2025-10-27 的 main（v2.2 + 之后的 bugfix）

export OPENSCENE_DATA_ROOT=$V2_DATA
export NAVSIM_EXP_ROOT=$V2_EXP
export NAVSIM_DEVKIT_ROOT=$V2_DEVKIT
export NUPLAN_MAPS_ROOT=$V2_MAPS
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export TOKENIZERS_PARALLELISM=false

PY=${PY:-/home/mh2803/miniconda3/envs/autovla_codeclean/bin/python}
MODEL_DIR=${MODEL_DIR:-/data/mh2803/navsafe/navsafe_data/model_zoo/autovla_l0r0_cot_turn40_freeze}
CKPT=${CKPT:-$MODEL_DIR/epoch=4-val_action_loss=2.1827.ckpt}
CONFIG=${CONFIG:-$MODEL_DIR/lightning_logs/version_0/hparams.yaml}   # 训练时的完整 config（L0R0、CoT）
BASE=${BASE:-/data/mh2803/navsafe/navsafe_data/model_zoo/_base/qwen2.5-vl-3b-instruct}

# 两套 navsim 不能同时在 path 上：
#   v2 步骤（读场景、写 submission、打分）→ PYTHONPATH 只放 v2 devkit
#   推理步骤 → 不改 PYTHONPATH，用环境里装的仓库版 navsim（v1.1）和 models/
v2py() { PYTHONPATH=$V2_DEVKIT "$PY" "$@"; }
