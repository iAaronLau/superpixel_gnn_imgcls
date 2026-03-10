#!/bin/bash

source ./slurm/functional.sh

JOB_NAME="spgnn-cifar10-gcn"
PARTITION_NAME=${PARTITION_NAME:-ouyqcobre_el9,el9_gpu_test}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
NODES=${NODES:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-12}
RUNTIME_ENV_LINES="export WANDB_MODE=online"

SCRIPT=$(cat <<'EOF'
train.py --dataset cifar10 
        --model gcn 
        --train_backend transformers 
        --n_segments 200 
        --use_xy 1 
        --image_size 64 
        --batch_size 128 
        --epochs 30 
        --lr 1e-3 
        --weight_decay 1e-4 
        --seed 42 
        --scheduler cosine 
        --mixed_precision bf16 
        --num_workers 8 
        --use_cache 1 
        --cache_dir graph_cache 
        --use_wandb 1 
        --wandb_project superpixel-gnn-imgcls 
        --run_name cifar10_gcn_seg200_xy1_final
EOF
)

job_id=$(submit_slurm_job)
print_job_helpers "$job_id" "$SBATCH_OUTPUT" "$JOB_NAME"
