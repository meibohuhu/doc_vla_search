#!/bin/bash
# 把本机 nuplan 的 train + test 数据整份传到 sporc（约 229 GB）。
#   bash scripts/1103/rsync_nuplan_to_sporc.sh              # 传（建议在 tmux 里跑，断了重跑会续传）
#   DRY=1   bash scripts/1103/rsync_nuplan_to_sporc.sh      # 只列出要传什么，不传
#   ONLY3=1 bash scripts/1103/rsync_nuplan_to_sporc.sh      # 图片只传训练用的 CAM_F0/L1/R1（trainval 155→约 100 GB）
#   DEST=/其他/目录 bash scripts/1103/rsync_nuplan_to_sporc.sh
#
# 传的东西（DEST 下的目录结构和本机 /scratch/mh2803/vla/nuplan 一样）：
#   sensor_blobs/{trainval,test}                       174 GB  图片
#   navsim_logs/{trainval,test}                         15 GB
#   navtrain_metric_cache, navtest_metric_cache         37 GB
#   navtrain_cot, navtrain_cot_val                     0.4 GB  相机路径是相对的
#   navtrain_nocot, navtrain_nocot_val, navtest_nocot  0.6 GB  相机路径是绝对的
#   cf_navtrain_v1                                     0.9 GB
#   maps                                               1.4 GB  train/test 共用
# 不传：navsim_logs/mini、166k 那两份、eval 结果、exp。
#
# 原样传，不改路径。到 sporc 上要改的绝对路径（都是 /scratch/mh2803/vla/nuplan 开头）：
#   navtrain_nocot / navtrain_nocot_val / navtest_nocot 的 JSON 里的相机路径
#   navtrain_metric_cache/metadata/*.csv、navtest_metric_cache/metadata/*.csv 里的 pkl 路径
set -eu
HOST="${HOST:-jw4905@sporcsubmit.rc.rit.edu}"
DEST="${DEST:-/home/jw4905/nuplan_data}"
SRC=/scratch/mh2803/vla/nuplan
DIRS=(sensor_blobs/trainval sensor_blobs/test
      navsim_logs/trainval navsim_logs/test
      navtrain_metric_cache navtest_metric_cache
      navtrain_cot navtrain_cot_val navtrain_nocot navtrain_nocot_val navtest_nocot
      cf_navtrain_v1 maps)

# -R + "$SRC/./xxx"：在 DEST 下保留 xxx 这段相对路径；图片和 pkl 本身压过，不开 -z
OPT=(-a -R --partial --info=stats1,progress2)
[ "${ONLY3:-0}" = 1 ] && OPT+=(--exclude=CAM_L0/ --exclude=CAM_R0/)
[ "${DRY:-0}" = 1 ] && OPT+=(--dry-run)

echo "from : $SRC"
echo "to   : $HOST:$DEST"

# 只输一次密码：第一次 ssh 建一条主连接（ControlMaster），后面的 rsync / ssh 都复用它，结束时关掉
mkdir -p ~/.ssh && chmod 700 ~/.ssh
SSH="ssh -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-%r@%h:%p -o ControlPersist=30m -o ServerAliveInterval=60"
trap '$SSH -O exit "$HOST" 2>/dev/null' EXIT
echo "== 连接 sporc（这里输一次密码）=="
$SSH "$HOST" "mkdir -p '$DEST' && echo '-- 远端空间 --' && df -h '$DEST' | tail -1 && (quota -s 2>/dev/null || true)"

if [ "${DRY:-0}" != 1 ]; then
  read -r -p "要传约 229 GB（ONLY3=1 约 175 GB），上面的空间/配额够吗？继续 [y/N] " ok
  [ "$ok" = y ] || [ "$ok" = Y ] || exit 1
fi

ARGS=()
for d in "${DIRS[@]}"; do ARGS+=("$SRC/./$d"); done
rsync -e "$SSH" "${OPT[@]}" "${ARGS[@]}" "$HOST:$DEST/"

[ "${DRY:-0}" = 1 ] && exit 0

# 对账：每个目录的文件数两边比
echo "== 对账（文件数：本机 / 远端）=="
CNT='for d in '"${DIRS[*]}"'; do echo "$d $(find "$d" -type f'
[ "${ONLY3:-0}" = 1 ] && CNT+=' -not -path "*/CAM_L0/*" -not -path "*/CAM_R0/*"'
CNT+=' | wc -l)"; done'
paste <(cd "$SRC" && bash -c "$CNT") <($SSH "$HOST" "cd '$DEST' && $CNT") | awk '{printf "%-24s %9s %9s %s\n",$1,$2,$4,($2==$4?"ok":"<-- 不一致，重跑脚本续传")}'
