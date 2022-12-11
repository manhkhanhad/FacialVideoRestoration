: ${OUTPUT_FOLDER="/home/ldtuan/Test/results/"}
: ${VAL_VISUAL_FOLDER="/home/ldtuan/VideoRestoration/BasicVSR_PlusPlus/work_dirs/BasicVSR_GFPGAN/val_visuals"}
: ${OFFICIAL_DEGRADATION_FOLDER="/home/ldtuan/VideoRestoration/dataset/official_degradation/"}
: ${GFP_GAN_MODEL_PATH="/home/ldtuan/VideoRestoration/GFPGAN/experiments/finetune2_GFPGAN_TalkingHead/models/net_g_420000.pth"}
: ${VID="-cNe_z2qsGQ_0000_S1010_E1114_L893_T112_R1213_B432"}

# get e2e video
mkdir -p ${OUTPUT_FOLDER}/images/${VID}_e2e_facial
CONFIG="/home/ldtuan/VideoRestoration/BasicVSR_PlusPlus/configs/sterr_gan.py"
CHECKPOINT="/home/ldtuan/VideoRestoration/BasicVSR_PlusPlus/work_dirs/BasicVSR_GFPGAN_facial_w_component/latest.pth"
GPUS=1

PYTHONPATH="/home/ldtuan/VideoRestoration/BasicVSR_PlusPlus/":$PYTHONPATH \
BASICSR_JIT=True CUDA_VISIBLE_DEVICES=1 python \
    /home/ldtuan/VideoRestoration/BasicVSR_PlusPlus/demo/restoration_video_demo.py \
    $CONFIG \
    $CHECKPOINT \
    ${OFFICIAL_DEGRADATION_FOLDER}/degradation/degraded_images/${VID} \
    ${OUTPUT_FOLDER}/images/${VID}_e2e_facial \
    --start-idx 0 \
    --filename-tmpl '{:03d}.png' \
    --max-seq-len 30
    ${@:4}
ffmpeg -y -framerate 24 -pattern_type glob -i "${OUTPUT_FOLDER}/images/${VID}_e2e_facial/*.png" -c:v libx264 -pix_fmt yuv420p ${OUTPUT_FOLDER}/video/sterr_gan/${VID}_e2e_facial.mp4