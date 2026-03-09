# Superpixel GNN Image Classification

Runnable baseline for image classification with:

- CNN baseline (`resnet`)
- Superpixel graph models (`gcn`, `gat`, `graph_transformer`)
- Datasets from HuggingFace/local image folders (`cifar10`, `imagewoof`, `imagenette2`)
- Two training backends: `accelerate` and `transformers.Trainer`
- Default training backend: `transformers.Trainer` (HF ecosystem first)
- Optional logging with `wandb`
- W&B classification metrics: AUC / PR-AUC / F1 / Precision / Recall / MCC / Kappa
- W&B plots: confusion matrix, ROC curve, PR curve, per-class metric table
- Checkpoint save/load/resume (`--resume`)
- Optional push to Hugging Face Hub
- SLURM submission scripts

## Install

```bash
pip install -r requirements.txt
```

## Quick Run

### CIFAR-10 + GCN (`transformers` backend)

```bash
python train.py \
  --dataset cifar10 \
  --model gcn \
  --n_segments 100 \
  --use_xy 1 \
  --image_size 64 \
  --batch_size 64 \
  --epochs 50 \
  --lr 1e-3 \
  --seed 42 \
  --use_cache 1 \
  --cache_dir graph_cache \
  --use_wandb 1
```

### CIFAR-10 + ResNet (`transformers` backend)

```bash
python train.py \
  --dataset cifar10 \
  --model resnet \
  --image_size 64 \
  --batch_size 128 \
  --epochs 50 \
  --lr 1e-3 \
  --scheduler cosine \
  --seed 42 \
  --use_wandb 1
```

### Resume Training

```bash
python train.py \
  --dataset cifar10 \
  --model resnet \
  --run_name cifar10_resnet_tf \
  --resume auto
```

`--resume` supports:

- an explicit checkpoint path
- `latest` / `last` (resolve `outputs/<run_name>/latest` or last checkpoint)
- `auto` (use `transformers` checkpoint discovery)

### Push To HF Hub

```bash
python train.py \
  --dataset cifar10 \
  --model resnet \
  --push_to_hub 1 \
  --hub_model_id your-org/superpixel-gnn-cifar10 \
  --hub_private_repo 1 \
  --hub_strategy every_save
```

## Output

- Run config: `outputs/<run_name>/config.json`
- Best checkpoint (accelerate): `outputs/<run_name>/best.pt`
- Best checkpoint (transformers): `outputs/<run_name>/checkpoint-*` and symlink `outputs/<run_name>/latest`
- Result table: `outputs/results.csv`

## SLURM

Prebuilt scripts:

- `batch_train_cifar10_resnet.conda.sh`
- `batch_train_cifar10_gcn.conda.sh`
- `batch_train_cifar10_gat.conda.sh`
- `batch_train_imagewoof_gat.conda.sh`

Common helper:

- `slurm/functional.sh`

Usage:

```bash
bash batch_train_cifar10_resnet.conda.sh
```

Optional envs before submission:

- `CONDA_SH`, `CONDA_ENV_NAME`
- `PARTITION_NAME`, `GPUS_PER_NODE`, `NODES`, `CPUS_PER_TASK`
- `WANDB_MODE` (`online`/`offline`)

## Batch Experiments / Hyperparameter Sweep

Use the built-in sweep launcher to run the requested experiment matrix with fixed seed/split and automatic logging:

```bash
python run_experiments.py \
  --seed 42 \
  --train_backend transformers \
  --lr_grid 3e-4,1e-3 \
  --batch_size_grid 16,32,64 \
  --weight_decay_grid 1e-4
```

Useful controls:

- `--dry_run`: print planned commands only
- `--max_runs N`: cap total launched runs
- `--retries K`: retry failed runs up to K times, then skip
