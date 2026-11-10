#!/bin/bash
# 把 CoLT 测评要的数据从 gcis 传到 tigris（docs/1104/colt_eval.md）。
#   bash scripts/1104/rsync_colt_to_tigris.sh                    # 传
#   DRY=1 bash scripts/1104/rsync_colt_to_tigris.sh              # 只列出要传什么，不传
#   SPLITS=vfull bash scripts/1104/rsync_colt_to_tigris.sh       # 只传 vfull（默认 vfull + vclean）
#   DEST=/其他/目录 bash scripts/1104/rsync_colt_to_tigris.sh
#
# 传两样：
#   data/{vfull,vclean}/   CoLT-Drive 数据，每个 split 1768 个样本目录 + manifest.json，约 2.4 GB / split
#   vlm_labels/vfull/qwen3vl-32b/final_single_v28_single_f0_01aabc_w1280.jsonl   v28 的 VLM 标签（32B 单次调用，约 470 KB）
#     目录结构和本机一致，--vlm-labels 可以写成 …/vlm_labels/{split}/qwen3vl-32b/…（vclean 还没有标签）
# 不传：ckpt 和 Qwen2.5-VL-3B 底座（tigris 上要已经有）、代码（走 git）。
set -eu
HOST="${HOST:-mh2803@tigris.rc.rit.edu}"
DEST="${DEST:-/shared/rc/llm-gen-agent/mhu/colt-drive}"
SPLITS="${SPLITS:-vfull vclean}"
SRC_DATA=/scratch/mh2803/vla/CoLT-Drive/data
LABEL_REL=vfull/qwen3vl-32b/final_single_v28_single_f0_01aabc_w1280.jsonl
SRC_LABEL=/scratch/mh2803/vla/colt/vlm_probe/results/$LABEL_REL
OPT=(-a -z --partial --info=stats1,progress2)
[ "${DRY:-0}" = 1 ] && OPT+=(--dry-run)

echo "from : $SRC_DATA ($SPLITS)  +  $SRC_LABEL"
echo "to   : $HOST:$DEST"

# 只输一次密码：第一次 ssh 建一条主连接（ControlMaster），后面的 rsync / ssh 都复用它，结束时关掉
mkdir -p ~/.ssh && chmod 700 ~/.ssh
SSH="ssh -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-%r@%h:%p -o ControlPersist=30m"
trap '$SSH -O exit "$HOST" 2>/dev/null' EXIT
echo "== 连接 tigris（这里输一次密码）=="
$SSH "$HOST" "mkdir -p '$DEST/data' '$DEST/vlm_labels/$(dirname "$LABEL_REL")'"

# 目录末尾不带 / → 在 DEST/data 下建同名目录（vfull / vclean）
for s in $SPLITS; do
    rsync -e "$SSH" "${OPT[@]}" "$SRC_DATA/$s" "$HOST:$DEST/data/"
done
rsync -e "$SSH" "${OPT[@]}" "$SRC_LABEL" "$HOST:$DEST/vlm_labels/$LABEL_REL"

[ "${DRY:-0}" = 1 ] && exit 0
# 对账：每个 split 的样本目录数、图片数，标签文件的 md5
echo "== 对账 =="
for s in $SPLITS; do
    $SSH "$HOST" "cd '$DEST/data/$s' && echo $s 样本目录 \$(find . -mindepth 1 -maxdepth 1 -type d | wc -l) \
      图片 \$(find . -name front_camera.jpg | wc -l) gt \$(find . -name gt.json | wc -l) '(都应为 1768)' \
      && ls manifest.json"
done
$SSH "$HOST" "md5sum '$DEST/vlm_labels/$LABEL_REL'"
echo "本机："
md5sum "$SRC_LABEL"

cat <<EOF

== tigris 上跑（仓库根目录下，先 --every 50 冒烟，总计应为 接 VLM 72.2 / GT 指令 94.4）==
PY=python LOGDIR=<日志目录> GPUS="0 1" bash scripts/1104/run_colt_autovla.sh \\
    --data $DEST/data \\
    --vlm-labels "$DEST/vlm_labels/{split}/qwen3vl-32b/final_single_v28_single_f0_01aabc_w1280.jsonl" \\
    --ckpt <…/epoch=4-loss=0.6051.ckpt> --base /shared/rc/llm-gen-agent/mhu/auto_vla/Qwen2.5-VL-3B-Instruct \\
    --out $DEST/autovla
EOF
