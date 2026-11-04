#!/bin/bash
# ============================================================================
# 把训练用的 CoT 数据从 tigris /shared 拷到 sporc 上 jw4905 的目录。
#   在 tigris 登录节点上跑（数据在这边），一次 rsync 只认证一次：
#     bash scripts/1103/rsync_cot_to_sporc.sh
#   DRY=1 先只列要传的东西，不真传。中断了直接重跑，rsync 会跳过已经传完的文件。
#
# 拷过去的东西（共约 145 MB，1.2 万个小 json）:
#   （navtrain_cot / navtrain_cot_val 不传，sporc 那边已经有）
#   navtest_cot/        12,146 json  103M   train（只有加 navtest 的跑用）
#   cot_navtest.json                  1.5M   navtest 的中间结果（训练不读，留个底）
#   cf_navtrain_v1/       2 个 npz + stats.json  41M   cf 索引
#
# ⚠️ 不包括图片。json 里的相机路径是相对路径（trainval/... 和 test/...），
#    sporc 上训练 config 的 sensor_data_path 要指向那边的 sensor_blobs
#    （下面要有 trainval/ 和 test/ 两个子目录）。
# ============================================================================
set -eu

SRC=/shared/rc/llm-gen-agent/mhu/auto_vla/nuplan
DST_HOST=jw4905@sporcsubmit.rc.rit.edu
DST_DIR="${DST_DIR:-/home/jw4905/nuplan_data}"   # 换目录: DST_DIR=/xxx bash 本脚本

ITEMS=(navtest_cot cot_navtest.json cf_navtrain_v1)

# 本地先检查一遍，免得传了半截才发现少东西
for it in "${ITEMS[@]}"; do
  [ -e "$SRC/$it" ] || { echo "ERROR: 找不到 $SRC/$it"; exit 1; }
done

# -a       保留时间戳（重跑时靠它跳过已传的文件）
# --no-perms --no-group --chmod  对面是别人的 home，不带过去我们的组和 ACL
# -R + /./ 在对面保留 nuplan_data/<目录名>/ 这一层结构
# -z       json 压缩率很高，值得开
DRYFLAG=(); [ "${DRY:-0}" = "1" ] && DRYFLAG=(--dry-run)
rsync -aRz --no-perms --no-group --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
  --info=progress2,stats1 --partial "${DRYFLAG[@]}" \
  --rsync-path="mkdir -p $DST_DIR && rsync" \
  "${ITEMS[@]/#/$SRC/./}" \
  "$DST_HOST:$DST_DIR/"

[ "${DRY:-0}" = "1" ] && exit 0

# 对一下文件数
echo "=== 对面的文件数（应该是 12146 / 3）==="
ssh "$DST_HOST" "cd $DST_DIR && for d in navtest_cot cf_navtrain_v1; do printf '%-18s %s\n' \$d \$(find \$d -maxdepth 1 -type f | wc -l); done; ls -la cot_navtest.json; du -sh ."
