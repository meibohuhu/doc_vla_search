#!/bin/bash
# 把 cf SFT 训练要的数据从 gcis 传到 tigris（docs/1026/cf_sft_code_changes.md「上 cluster 之前」）。
#   bash scripts/1028/rsync_cf_data_to_tigris.sh                 # 传
#   DRY=1 bash scripts/1028/rsync_cf_data_to_tigris.sh           # 只列出要传什么，不传
#   DEST=/其他/目录 bash scripts/1028/rsync_cf_data_to_tigris.sh
#
# 传四样（约 830 MB，101k + 2k 个小 JSON）：
#   navtrain_cot/        101,288 个 JSON，带重新生成的 CoT（docs/1026/cot_regen_navtrain.md）
#   navtrain_cot_val/      2,000 个 JSON
#   cf_navtrain_v1/cf_navtrain_v1_{train,val}.npz + stats.json   cf 索引（D6）
# 不传：图片（sensor_blobs，tigris 上要已经有 navtrain 的）、基座权重（tigris 上从 HF 下）、代码（走 git）。
# JSON 里的相机路径是相对 sensor_blobs/ 的，tigris 上不用改 JSON，只在 config 里设 sensor_data_path。
set -eu
HOST="${HOST:-mh2803@tigris.rc.rit.edu}"
DEST="${DEST:-/shared/rc/llm-gen-agent/mhu/auto_vla/nuplan}"   # tigris 上的 nuplan 根目录（sensor_blobs 也在这下面）
SRC=/scratch/mh2803/vla/nuplan
OPT=(-a -z --partial --info=stats1,progress2)
[ "${DRY:-0}" = 1 ] && OPT+=(--dry-run)

echo "from : $SRC"
echo "to   : $HOST:$DEST"

# 只输一次密码：第一次 ssh 建一条主连接（ControlMaster），后面的 rsync / ssh 都复用它，结束时关掉
mkdir -p ~/.ssh && chmod 700 ~/.ssh
SSH="ssh -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-%r@%h:%p -o ControlPersist=30m"
trap '$SSH -O exit "$HOST" 2>/dev/null' EXIT
echo "== 连接 tigris（这里输一次密码）=="
$SSH "$HOST" "mkdir -p '$DEST/cf_navtrain_v1'"

# 目录末尾不带 / → 在 DEST 下建同名目录
rsync -e "$SSH" "${OPT[@]}" "$SRC/navtrain_cot" "$SRC/navtrain_cot_val" "$HOST:$DEST/"
rsync -e "$SSH" "${OPT[@]}" "$SRC/cf_navtrain_v1/cf_navtrain_v1_train.npz" "$SRC/cf_navtrain_v1/cf_navtrain_v1_val.npz" \
      "$SRC/cf_navtrain_v1/stats.json" "$HOST:$DEST/cf_navtrain_v1/"

[ "${DRY:-0}" = 1 ] && exit 0
# 对账：文件数和 npz 校验和
echo "== 对账 =="
$SSH "$HOST" "cd '$DEST' && echo navtrain_cot \$(find navtrain_cot -maxdepth 1 -name '*.json' | wc -l) '(应为 101288)' \
  && echo navtrain_cot_val \$(find navtrain_cot_val -maxdepth 1 -name '*.json' | wc -l) '(应为 2000)' \
  && md5sum cf_navtrain_v1/*.npz"
echo "本机："
md5sum "$SRC"/cf_navtrain_v1/cf_navtrain_v1_*.npz
