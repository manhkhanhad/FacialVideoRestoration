#!/usr/bin/env bash

CONFIG="/home/ldtuan/VideoRestoration/BasicVSR_PlusPlus/configs/sterr_gan.py"
CHECKPOINT="/home/ldtuan/VideoRestoration/BasicVSR_PlusPlus/work_dirs/BasicVSR_GFPGAN_facial_w_component/latest.pth"
GPUS=1
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# BASICSR_JIT=True python -m torch.distributed.launch \
#     --nnodes=$NNODES \
#     --node_rank=$NODE_RANK \
#     --master_addr=$MASTER_ADDR \
#     --nproc_per_node=$GPUS \
#     --master_port=$PORT \
#     $(dirname "$0")/test.py \
#     $CONFIG \
#     $CHECKPOINT \
#     --launcher pytorch \
#     --save-path /home/ldtuan/VideoRestoration/BasicVSR_PlusPlus/work_dirs/BasicVSR_GFPGAN/test_result \
#     ${@:4}

