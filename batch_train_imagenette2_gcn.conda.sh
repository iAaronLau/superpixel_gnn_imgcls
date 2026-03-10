#!/bin/bash

source ./slurm/functional.sh

JOB_NAME="spgnn-imagenette2-gcn"
PARTITION_NAME=${PARTITION_NAME:-ouyqcobre_el9,el9_gpu_test}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
NODES=${NODES:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-16}
RUNTIME_ENV_LINES="export WANDB_MODE=online"

SCRIPT=$(cat <<'EOF'
train.py --dataset imagenette2 
        --model gcn 
        --train_backend transformers 
        --n_segments 100 
        --use_xy 0 
        --hidden_dim 256 
        --gnn_layers 4 
        --dropout 0.1 
        --pooling meanmax 
        --slic_compactness 20 
        --image_size 224 
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
        --graph_cache_version v3 
        --graph_node_drop_prob 0.05 
        --graph_edge_drop_prob 0.1 
        --graph_feature_mask_prob 0.05 
        --graph_feature_noise_std 0.02 
        --graph_edge_noise_std 0.01 
        --use_wandb 1 
        --wandb_project superpixel-gnn-imgcls 
        --run_name imagenette2_gcn_seg100_xy0_final
EOF
)

job_id=$(submit_slurm_job)
print_job_helpers "$job_id" "$SBATCH_OUTPUT" "$JOB_NAME"
