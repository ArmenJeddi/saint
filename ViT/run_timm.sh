#!/bin/bash

source $path_to_your_env

JOB_NAME="prune_vit_base_timm"

cd $Your_workspace

## adjust batch size for bigger models
MASTER_ADDR=localhost MASTER_PORT=29500 python example_timm.py \
                    --batch_size 512 \
                    --data_path /datasets/imagenet/ \
                    --nb_classes 1000 \
                    --output_dir ${JOB_NAME} \
                    --model vit_base_patch16_224 \
                    --benchmark_throughput \
                    --benchmark_gflops
