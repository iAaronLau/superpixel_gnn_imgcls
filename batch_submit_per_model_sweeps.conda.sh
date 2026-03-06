#!/bin/bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$PROJECT_ROOT"

MODELS_CSV=${MODELS:-resnet,gcn,gat}
SWEEP_PREFIX=${SWEEP_PREFIX:-model_hparam_sweep}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}

IFS=',' read -r -a MODEL_ARR <<< "$MODELS_CSV"

if [[ ${#MODEL_ARR[@]} -eq 0 ]]; then
  echo "[ERROR] MODELS is empty"
  exit 2
fi

for raw_model in "${MODEL_ARR[@]}"; do
  model=$(echo "$raw_model" | xargs)
  if [[ -z "$model" ]]; then
    continue
  fi

  sweep_name="${SWEEP_PREFIX}_${model}_${TIMESTAMP}"
  echo "[SubmitPerModel] model=$model sweep_name=$sweep_name"

  set +e
  MODELS="$model" SWEEP_NAME="$sweep_name" "$PROJECT_ROOT/batch_submit_hparam_sweep.conda.sh"
  rc=$?
  set -e

  if [[ $rc -ne 0 ]]; then
    echo "[SubmitPerModel] failed model=$model rc=$rc"
    if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
      exit $rc
    fi
  fi
done
