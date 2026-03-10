#!/bin/bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$PROJECT_ROOT"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  AUTO_CONDA_ACTIVATE_SUBMIT=${AUTO_CONDA_ACTIVATE_SUBMIT:-0}
else
  AUTO_CONDA_ACTIVATE_SUBMIT=${AUTO_CONDA_ACTIVATE_SUBMIT:-1}
fi
if [[ "$AUTO_CONDA_ACTIVATE_SUBMIT" == "1" ]] && [[ -f "./slurm/functional.sh" ]]; then
  # shellcheck disable=SC1091
  source ./slurm/functional.sh
  if declare -F setup_conda_env >/dev/null 2>&1; then
    setup_conda_env
  fi
fi

PYTHON_BIN=${PYTHON_BIN:-$(command -v python)}
if [[ -z "${PYTHON_BIN:-}" ]]; then
  echo "[ERROR] python executable not found"
  exit 2
fi

SWEEP_NAME=${SWEEP_NAME:-model_hparam_sweep_$(date +%Y%m%d_%H%M%S)}
SWEEP_ROOT=${SWEEP_ROOT:-outputs/sweeps/$SWEEP_NAME}
MANIFEST_PATH="$SWEEP_ROOT/manifest.jsonl"
PLANNED_CSV="$SWEEP_ROOT/planned_tasks.csv"

DATASETS=${DATASETS:-imagenette2,imagewoof}
MODELS=${MODELS:-resnet,gcn,gat}
RESNET_VARIANTS=${RESNET_VARIANTS:-resnet18,resnet34}
GRAPH_N_SEGMENTS_GRID=${GRAPH_N_SEGMENTS_GRID:-50,100,200}
GRAPH_USE_XY_GRID=${GRAPH_USE_XY_GRID:-0,1}
IMAGEWOOF_GRAPH_SEGMENTS=${IMAGEWOOF_GRAPH_SEGMENTS:-100}
LR_GRID=${LR_GRID:-3e-4,1e-3}
BATCH_SIZE_GRID=${BATCH_SIZE_GRID:-128}
WEIGHT_DECAY_GRID=${WEIGHT_DECAY_GRID:-1e-4}

SEED=${SEED:-42}
CIFAR_EPOCHS=${CIFAR_EPOCHS:-30}
IMAGEWOOF_EPOCHS=${IMAGEWOOF_EPOCHS:-30}
TRAIN_BACKEND=${TRAIN_BACKEND:-transformers}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
NUM_WORKERS=${NUM_WORKERS:-8}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
SCHEDULER=${SCHEDULER:-cosine}
EVAL_STRATEGY=${EVAL_STRATEGY:-epoch}
SAVE_STRATEGY=${SAVE_STRATEGY:-auto}
CHECKPOINTS_TOTAL_LIMIT=${CHECKPOINTS_TOTAL_LIMIT:-3}
USE_CACHE=${USE_CACHE:-1}
CACHE_DIR=${CACHE_DIR:-graph_cache}
USE_WANDB=${USE_WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-superpixel-gnn-imgcls}
WANDB_MODE=${WANDB_MODE:-online}
MAX_GRAPH_BATCH_CIFAR=${MAX_GRAPH_BATCH_CIFAR:-129}
MAX_GRAPH_BATCH_IMAGEWOOF=${MAX_GRAPH_BATCH_IMAGEWOOF:-129}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}
SKIP_EXISTING=${SKIP_EXISTING:-0}
MAX_RUNS=${MAX_RUNS:--1}
EXTRA_TRAIN_ARGS=${EXTRA_TRAIN_ARGS:-}
ALLOW_TF32=${ALLOW_TF32:-1}
CUDNN_BENCHMARK=${CUDNN_BENCHMARK:-1}
CHANNELS_LAST=${CHANNELS_LAST:-1}
TORCH_COMPILE=${TORCH_COMPILE:-0}
TORCH_COMPILE_MODE=${TORCH_COMPILE_MODE:-reduce-overhead}

mkdir -p "$SWEEP_ROOT"

BUILD_CMD=(
  "$PYTHON_BIN" slurm/build_sweep_manifest.py
  --project_root "$PROJECT_ROOT"
  --python "$PYTHON_BIN"
  --sweep_name "$SWEEP_NAME"
  --sweep_root "$SWEEP_ROOT"
  --datasets "$DATASETS"
  --models "$MODELS"
  --resnet_variants "$RESNET_VARIANTS"
  --graph_n_segments_grid "$GRAPH_N_SEGMENTS_GRID"
  --graph_use_xy_grid "$GRAPH_USE_XY_GRID"
  --imagewoof_graph_segments "$IMAGEWOOF_GRAPH_SEGMENTS"
  --cifar_epochs "$CIFAR_EPOCHS"
  --imagewoof_epochs "$IMAGEWOOF_EPOCHS"
  --lr_grid "$LR_GRID"
  --batch_size_grid "$BATCH_SIZE_GRID"
  --weight_decay_grid "$WEIGHT_DECAY_GRID"
  --max_graph_batch_cifar "$MAX_GRAPH_BATCH_CIFAR"
  --max_graph_batch_imagewoof "$MAX_GRAPH_BATCH_IMAGEWOOF"
  --train_backend "$TRAIN_BACKEND"
  --seed "$SEED"
  --mixed_precision "$MIXED_PRECISION"
  --num_workers "$NUM_WORKERS"
  --gradient_accumulation_steps "$GRAD_ACCUM_STEPS"
  --scheduler "$SCHEDULER"
  --eval_strategy "$EVAL_STRATEGY"
  --save_strategy "$SAVE_STRATEGY"
  --checkpoints_total_limit "$CHECKPOINTS_TOTAL_LIMIT"
  --use_wandb "$USE_WANDB"
  --wandb_project "$WANDB_PROJECT"
  --wandb_mode "$WANDB_MODE"
  --use_cache "$USE_CACHE"
  --cache_dir "$CACHE_DIR"
  --skip_completed "$SKIP_COMPLETED"
  --skip_existing "$SKIP_EXISTING"
  --max_runs "$MAX_RUNS"
)

PERF_TRAIN_ARGS="--allow_tf32 $ALLOW_TF32 --cudnn_benchmark $CUDNN_BENCHMARK --channels_last $CHANNELS_LAST --torch_compile $TORCH_COMPILE --torch_compile_mode $TORCH_COMPILE_MODE"
if [[ -n "$EXTRA_TRAIN_ARGS" ]]; then
  PERF_TRAIN_ARGS="$PERF_TRAIN_ARGS $EXTRA_TRAIN_ARGS"
fi
BUILD_CMD+=(--extra_train_args "$PERF_TRAIN_ARGS")
if [[ -n "${EXTRA_MANIFEST_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_MANIFEST_ARR=($EXTRA_MANIFEST_ARGS)
  BUILD_CMD+=("${EXTRA_MANIFEST_ARR[@]}")
fi

echo "[Submit] building sweep manifest..."
"${BUILD_CMD[@]}"

if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "[ERROR] manifest not found: $MANIFEST_PATH"
  exit 2
fi

TOTAL_TASKS=$(wc -l < "$MANIFEST_PATH")
if [[ "$TOTAL_TASKS" -le 0 ]]; then
  echo "[Submit] no tasks to submit"
  exit 0
fi

EFFECTIVE_TASKS=$TOTAL_TASKS
if [[ "${MAX_TASKS:-0}" -gt 0 ]] && [[ "$MAX_TASKS" -lt "$EFFECTIVE_TASKS" ]]; then
  EFFECTIVE_TASKS=$MAX_TASKS
fi

MAX_PARALLEL=${MAX_PARALLEL:-8}
ARRAY_SPEC="0-$((EFFECTIVE_TASKS - 1))%$MAX_PARALLEL"

PARTITION_NAME=${PARTITION_NAME:-ouyqcobre_el9,el9_gpu_test}
NODES=${NODES:-1}
NTASKS_PER_NODE=${NTASKS_PER_NODE:-${NTASKS:-1}}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
NODE_GPU_COUNT=${NODE_GPU_COUNT:-2}
GPU_PER_TASK=${GPU_PER_TASK:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-${GPUS_PER_TASK:-$GPU_PER_TASK}}
TIME_LIMIT=${TIME_LIMIT:-24:00:00}
SBATCH_NICE=${SBATCH_NICE:-100}
RETRIES=${RETRIES:-1}
RETRY_SLEEP_SEC=${RETRY_SLEEP_SEC:-15}
SKIP_IF_RECORDED=${SKIP_IF_RECORDED:-1}
IDLE_TIMEOUT_SEC=${IDLE_TIMEOUT_SEC:-5400}
AUTO_CONDA_ACTIVATE_TASK=${AUTO_CONDA_ACTIVATE_TASK:-0}

if [[ "$NTASKS_PER_NODE" -ne 1 ]]; then
  echo "[WARN] current script assumes one task per array job; forcing NTASKS_PER_NODE=1 (got $NTASKS_PER_NODE)"
  NTASKS_PER_NODE=1
fi

if [[ "$GPU_PER_TASK" -lt 0 ]]; then
  echo "[ERROR] GPU_PER_TASK must be >= 0, got $GPU_PER_TASK"
  exit 2
fi

if [[ "$NODE_GPU_COUNT" -gt 0 ]] && [[ "$GPU_PER_TASK" -gt "$NODE_GPU_COUNT" ]]; then
  echo "[ERROR] GPU_PER_TASK=$GPU_PER_TASK exceeds NODE_GPU_COUNT=$NODE_GPU_COUNT"
  exit 2
fi

if [[ "$GPU_PER_TASK" -eq 0 ]]; then
  GPUS_PER_NODE=0
else
  GPUS_PER_NODE=$GPU_PER_TASK
fi

SLURM_LOG_DIR=${SLURM_LOG_DIR:-$SWEEP_ROOT/slurm_logs}
SBATCH_OUTPUT=${SBATCH_OUTPUT:-$SLURM_LOG_DIR/%x-%A_%a.out}
# Merge stderr into stdout file: keep one log file per task.
SBATCH_ERROR=${SBATCH_OUTPUT}
mkdir -p "$SLURM_LOG_DIR"

SBATCH_CMD=(
  sbatch
  --job-name "${JOB_NAME:-spgnn-sweep}"
  --partition "$PARTITION_NAME"
  --nodes "$NODES"
  --ntasks-per-node "$NTASKS_PER_NODE"
  --cpus-per-task "$CPUS_PER_TASK"
  --time "$TIME_LIMIT"
  --nice="$SBATCH_NICE"
  --array "$ARRAY_SPEC"
  --output "$SBATCH_OUTPUT"
  --error "$SBATCH_ERROR"
  --export "ALL,PROJECT_ROOT=$PROJECT_ROOT,MANIFEST=$MANIFEST_PATH,SWEEP_ROOT=$SWEEP_ROOT,PYTHON_BIN=$PYTHON_BIN,RETRIES=$RETRIES,RETRY_SLEEP_SEC=$RETRY_SLEEP_SEC,SKIP_IF_RECORDED=$SKIP_IF_RECORDED,IDLE_TIMEOUT_SEC=$IDLE_TIMEOUT_SEC,AUTO_CONDA_ACTIVATE=$AUTO_CONDA_ACTIVATE_TASK"
)

if [[ "${GPUS_PER_NODE:-0}" -gt 0 ]]; then
  SBATCH_CMD+=(--gpus-per-node "$GPUS_PER_NODE")
fi

if [[ -n "${MEMORY:-}" ]] && [[ "$MEMORY" != "0" ]]; then
  SBATCH_CMD+=(--mem "$MEMORY")
fi
if [[ -n "${ACCOUNT:-}" ]]; then
  SBATCH_CMD+=(--account "$ACCOUNT")
fi
if [[ -n "${QOS:-}" ]]; then
  SBATCH_CMD+=(--qos "$QOS")
fi
if [[ -n "${CONSTRAINT:-}" ]]; then
  SBATCH_CMD+=(--constraint "$CONSTRAINT")
fi
if [[ -n "${EXTRA_SBATCH_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_SBATCH_ARR=($EXTRA_SBATCH_ARGS)
  SBATCH_CMD+=("${EXTRA_SBATCH_ARR[@]}")
fi

SBATCH_CMD+=("$PROJECT_ROOT/slurm/sweep_array_job.sh")

echo "[Submit] sweep_name=$SWEEP_NAME"
echo "[Submit] sweep_root=$SWEEP_ROOT"
echo "[Submit] planned_csv=$PLANNED_CSV"
echo "[Submit] manifest=$MANIFEST_PATH"
echo "[Submit] tasks=$EFFECTIVE_TASKS/$TOTAL_TASKS array=$ARRAY_SPEC"
echo "[Submit] resources partition=$PARTITION_NAME node_gpu_count=$NODE_GPU_COUNT gpu_per_task=$GPU_PER_TASK ntasks_per_node=$NTASKS_PER_NODE"
echo "[Submit] scheduling nice=$SBATCH_NICE"
echo "[Submit] batch_grid=$BATCH_SIZE_GRID"
echo "[Submit] resume skip_completed=$SKIP_COMPLETED skip_existing_started=$SKIP_EXISTING"
echo "[Submit] watchdog idle_timeout_sec=$IDLE_TIMEOUT_SEC retries=$RETRIES"
echo "[Submit] perf allow_tf32=$ALLOW_TF32 cudnn_benchmark=$CUDNN_BENCHMARK channels_last=$CHANNELS_LAST torch_compile=$TORCH_COMPILE mode=$TORCH_COMPILE_MODE"
echo "[Submit] metrics_file=$SWEEP_ROOT/task_metrics.csv"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[DryRun] sbatch command:"
  printf ' %q' "${SBATCH_CMD[@]}"
  echo
  exit 0
fi

SBATCH_OUT=$("${SBATCH_CMD[@]}")
JOB_ID=$(awk '{print $4}' <<<"$SBATCH_OUT")

echo "$SBATCH_OUT"
echo "[Submit] job_id=$JOB_ID"
echo "squeue -j $JOB_ID"
echo "sacct -j $JOB_ID --format=JobID,JobName,State,Elapsed,MaxRSS"
echo "tail -f $SWEEP_ROOT/task_metrics.csv"
