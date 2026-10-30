#!/bin/bash
# ============================================================================
# 【计算节点上跑】一个节点 = 1 张 GH200，在这张卡上并行起若干个 navtest 分片的 PDMS 进程。
# 由 scripts/1001/eval_pdms_6ckpt_2node_6proc.sh 通过 srun 调用，不要直接跑。
#
#   _pdms_node_worker.sh <CKPT> <WORK> <SPLIT_TAG> <NSHARD> <shard> [<shard> ...]
#
# 环境（conda / PYTHONPATH / OPENSCENE_DATA_ROOT 等）由调用方 export，srun --export=ALL 传进来。
# CUDA_VISIBLE_DEVICES 由 Slurm 按 --gres 设好，这里不动。
# ============================================================================
set -u
CKPT="$1"; WORK="$2"; SPLIT_TAG="$3"; NSHARD="$4"; shift 4

cd "$REPO" || exit 1
echo "[$(hostname -s)] GPU=${CUDA_VISIBLE_DEVICES:-?}  shards: $*  ckpt: $(basename "$CKPT")"

pids=()
for s in "$@"; do
  d="$WORK/shard$s"; mkdir -p "$d"
  NAVSIM_EXP_ROOT="$d" \
  python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_cot.py" \
      train_test_split="${SPLIT_TAG}_shard${s}of${NSHARD}" \
      agent=autovla_agent \
      agent.config_path="$EVAL_CONFIG" \
      agent.checkpoint_path="'$CKPT'" \
      agent.sensor_data_path="$SENSOR_DATA_PATH" \
      agent.lora_conf.use_lora=false \
      metric_cache_path="$CACHE_PATH" \
      +json_data_path="$JSON_DATA_PATH" \
      experiment_name="pdms_s${s}" \
      > "$d/run.log" 2>&1 &
  pids+=($!)
  sleep 3            # 错开模型加载，避免同时读 ckpt 把 /shared 打满
done

rc=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { echo "[$(hostname -s)] ⚠️ shard 进程 ${pids[$i]} 非 0 退出，看 $WORK/shard*/run.log"; rc=1; }
done
echo "[$(hostname -s)] done rc=$rc"
exit $rc
