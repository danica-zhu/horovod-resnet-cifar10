#!/usr/bin/env bash
set -euo pipefail

# (可选) 如果用的是 conda 环境，取消注释两行
# source ~/.bashrc
# conda activate myconda

mkdir -p logs

EPOCHS=5
BS=1024

# 单机常用的稳定性设置（可留可删）
# export NCCL_DEBUG=WARN
export CUDA_LAUNCH_BLOCKING=0
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=0   # 若仍卡再改为 1 试试


for NP in 1 2 4; do
  echo "=== Running with ${NP} GPU(s) ==="
  horovodrun -np ${NP} -H localhost:${NP} \
    python train_hvd.py \
      --epochs ${EPOCHS} \
      --batch-size ${BS} \
    | tee logs/exp_${NP}g.log
done