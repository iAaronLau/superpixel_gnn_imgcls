#!/bin/bash

source ./slurm/functional.sh

JOB_NAME="spgnn-imagewoof-gat"
PARTITION_NAME=${PARTITION_NAME:-ouyqcobre_el9}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
NODES=${NODES:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-16}
EXTRA_SBATCH_LINES="#SBATCH --time=72:00:00"
RUNTIME_ENV_LINES="export WANDB_MODE=online"

SCRIPT=$(cat <<'INNER'
train.py --dataset imagewoof \
        --model gat \
        --train_backend transformers \
        --n_segments 100 \
        --use_xy 1 \
        --image_size 224 \
        --batch_size 32 \
        --epochs 40 \
        --lr 3e-4 \
        --scheduler cosine \
        --mixed_precision bf16 \
        --num_workers 8 \
        --use_cache 1 \
        --cache_dir graph_cache \
        --eval_strategy steps \
        --eval_steps 200 \
        --save_strategy steps \
        --save_steps 200 \
        --use_wandb 1 \
        --wandb_project superpixel-gnn-imgcls \
        --run_name imagewoof_gat_tf
INNER
)

job_id=$(submit_slurm_job)
print_job_helpers "$job_id" "$SBATCH_OUTPUT" "$JOB_NAME"
