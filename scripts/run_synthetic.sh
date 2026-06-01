#!/bin/bash

synthetic_scenes=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")

# Scheduler Settings. Milestones and Values
BATCHSIZE_VALUES="8,8"
BATCH_BREAKS="50"
SAMPLE_BREAKS="50"
SAMPLE_VALS="32,32"
CG_BREAKS="50"
CG_VALS="3,8"

LOSS_FN="mse" # "mse" or "mse_approx_ssim"
SSIM_WEIGHT=0.0
ITERATIONS=200
if [ "$LOSS_FN" = "mse_approx_ssim" ]; then
    SSIM_WEIGHT=0.2
    ITERATIONS=300
fi


for scene in "${synthetic_scenes[@]}"; do
    SOURCE_PATH="datasets/nerf_synthetic/${scene}"
    MODEL_PATH="output/${scene}"
    mkdir -p $MODEL_PATH
    python train.py \
            --auto_lr --max_lr 0.2 \
            --regularizer=0.1 --source_path $SOURCE_PATH --model_path $MODEL_PATH \
            --eval --iterations $ITERATIONS --sh_degree=0 --log_freq=10 --log_image_freq=100 \
            --sampling_distribution uniform --camera_sampler clusterdir \
            --cgiter_values $CG_VALS --cgiter_breakpoints $CG_BREAKS \
            --batchsize_values $BATCHSIZE_VALUES --batchsize_breakpoints $BATCH_BREAKS \
            --samplesize_values $SAMPLE_VALS --samplesize_breakpoints $SAMPLE_BREAKS \
            --loss_fn $LOSS_FN --ssim_weight $SSIM_WEIGHT
done 