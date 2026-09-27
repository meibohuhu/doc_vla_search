#!/bin/bash
# ============================================================================
# 把【评测相关数据】归到单独 folder /data/autovla_data/eval/，旧路径留 symlink，
# 这样现有 ~8 个引用旧路径的脚本/config 全部不受影响。
#
# 同一文件系统内 mv = 秒级 rename，可逆（symlink 也可删了 mv 回去）。
# 默认 DRY-RUN 只打印；确认无误后 APPLY=1 执行。
#
#   bash scripts/0901/reorg_eval_data.sh          # dry-run，只看要动什么
#   APPLY=1 bash scripts/0901/reorg_eval_data.sh  # 真执行
# ============================================================================
set -uo pipefail
DATA=/data/autovla_data
EVAL="$DATA/eval"
APPLY="${APPLY:-0}"

# (源绝对路径, 目标绝对路径) —— 评测相关数据
MAP=(
  "$DATA/nuplan/navtest_nocot|$EVAL/nuplan/navtest_nocot"
  "$DATA/nuplan/navtest_metric_cache|$EVAL/nuplan/navtest_metric_cache"
  "$DATA/nuplan/sensor_blobs/test|$EVAL/nuplan/sensor_blobs/test"
  "$DATA/nuplan/navsim_logs/test|$EVAL/nuplan/navsim_logs/test"
  "$DATA/nuscenes/nusc_eval_seg|$EVAL/nuscenes/nusc_eval_seg"
  "$DATA/nuplan/sft_eval|$EVAL/nuplan/sft_eval"
  "$DATA/nuplan/sft_eval_bf16full|$EVAL/nuplan/sft_eval_bf16full"
  "$DATA/nuplan/sft_eval_loradry|$EVAL/nuplan/sft_eval_loradry"
  "$DATA/nuplan/pdms_shards|$EVAL/nuplan/pdms_shards"
  "$DATA/nuplan/exp|$EVAL/nuplan/exp"
  "$DATA/nuplan/viz_tiers|$EVAL/nuplan/viz_tiers"
)

echo "=== reorg eval data  (APPLY=$APPLY) ==="
for pair in "${MAP[@]}"; do
  src="${pair%%|*}"; dst="${pair##*|}"
  if [ -L "$src" ]; then echo "  · 已是 symlink，跳过: $src"; continue; fi
  if [ ! -e "$src" ]; then echo "  · 源不存在，跳过: $src"; continue; fi
  sz=$(du -sh "$src" 2>/dev/null | cut -f1)
  if [ "$APPLY" = "1" ]; then
    mkdir -p "$(dirname "$dst")"
    mv "$src" "$dst"
    ln -s "$dst" "$src"
    echo "  ✓ [$sz] $src  →  $dst  (+symlink)"
  else
    echo "  [dry] [$sz] $src  →  $dst  (+symlink 回指)"
  fi
done

echo
if [ "$APPLY" = "1" ]; then
  echo "完成。eval 数据现集中在: $EVAL"
  echo "旧路径均为 symlink，现有脚本/config 无需改动。"
  du -sh "$EVAL" 2>/dev/null
else
  echo "以上为预览。确认后执行: APPLY=1 bash scripts/0901/reorg_eval_data.sh"
fi
