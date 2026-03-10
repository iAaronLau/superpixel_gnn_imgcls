#!/bin/bash

source ./slurm/functional.sh

JOB_NAME="spgnn-imagenette2-resnet"
PARTITION_NAME=${PARTITION_NAME:-ouyqcobre_el9,el9_gpu_test}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
NODES=${NODES:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
RUNTIME_ENV_LINES="export WANDB_MODE=online"

SCRIPT=$(cat <<'EOF'
train.py --dataset imagenette2 
        --model resnet 
        --train_backend transformers 
        --resnet_name resnet34 
        --image_size 224 
        --batch_size 128 
        --epochs 30 
        --lr 3e-4 
        --weight_decay 1e-4 
        --seed 42 
        --scheduler cosine 
        --mixed_precision bf16 
        --num_workers 8 
        --use_wandb 1 
        --wandb_project superpixel-gnn-imgcls 
        --run_name imagenette2_resnet34_final
EOF
)

job_id=$(submit_slurm_job)
print_job_helpers "$job_id" "$SBATCH_OUTPUT" "$JOB_NAME"
