#!/bin/bash

source $path_to_your_env

JOB_NAME="prune_mae_base"

cd $Your_workspace

## adjust batch size for bigger models
MASTER_ADDR=localhost MASTER_PORT=29500 python example_mae.py \
                    --batch_size 512 \
                    --data_path /datasets/imagenet/ \
                    --nb_classes 1000 \
                    --output_dir ${JOB_NAME} \
                    --ckpt_path ckpts/mae_finetuned_vit_base.pth \
                    --model vit_base_patch16 \
                    --benchmark_throughput \
                    --benchmark_gflops
