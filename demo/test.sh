#!/usr/bin/env bash

CONFIG="/home/ldtuan/VideoRestoration/BasicVSR_PlusPlus/configs/basicvsr_reds4.py"
CHECKPOINT="/home/ldtuan/VideoRestoration/BasicVSR_PlusPlus/work_dirs/BasicVSR_GFPGAN/latest.pth"
GPUS=1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
BASICSR_JIT=True python \
    $(dirname "$0")/restoration_video_demo.py \
    $CONFIG \
    $CHECKPOINT \
    /home/ldtuan/VideoRestoration/dataset/official_degradation/degradation/degraded_images/-B1Z9vrjpgg_0014_S620_E731_L1152_T168_R1264_B424 \
    /home/ldtuan/VideoRestoration/BasicVSR_PlusPlus/work_dirs/BasicVSR_GFPGAN/test_result_single \
    --start-idx 0 \
    --filename-tmpl '{:03d}.png'
    ${@:4}
    
