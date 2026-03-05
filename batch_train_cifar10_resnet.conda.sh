#!/bin/bash

source ./slurm/functional.sh

JOB_NAME="spgnn-cifar10-resnet"
PARTITION_NAME=${PARTITION_NAME:-ouyqcobre_el9}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
NODES=${NODES:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
RUNTIME_ENV_LINES="export WANDB_MODE=online"

SCRIPT=$(cat <<'INNER'
train.py --dataset cifar10 \
        --model resnet \
        --train_backend transformers \
        --image_size 64 \
        --batch_size 128 \
        --epochs 50 \
        --lr 1e-3 \
        --scheduler cosine \
        --mixed_precision bf16 \
        --num_workers 8 \
        --use_wandb 1 \
        --wandb_project superpixel-gnn-imgcls \
        --run_name cifar10_resnet_tf
INNER
)

job_id=$(submit_slurm_job)
print_job_helpers "$job_id" "$SBATCH_OUTPUT" "$JOB_NAME"
