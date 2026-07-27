#!/bin/bash
# Run AlgoEvolve on a single competition task.
# Usage:
#   Grading server disabled by default:
#     bash run_single_task.sh <EXP_ID> [TASK_ROOT]
#   Grading server enabled:
#     USE_GRADING_SERVER=True bash run_single_task.sh <EXP_ID> <DATASET_DIR> [SERVER_ID]
set -x

EXP_ID=${1:?Usage: bash run_single_task.sh <EXP_ID> [TASK_ROOT|DATASET_DIR] [SERVER_ID]}
TASK_ROOT_OR_DATASET_DIR=${2:-}
SERVER_ID=${3:-111}
USE_GRADING_SERVER=${USE_GRADING_SERVER:-False}

# Proxy (uncomment & fill in if behind a corporate firewall)
# export http_proxy=http://YOUR_PROXY:PORT
# export https_proxy=http://YOUR_PROXY:PORT

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [ -n "${TASK_ROOT_OR_DATASET_DIR}" ]; then
  if [ -f "${TASK_ROOT_OR_DATASET_DIR}/description.md" ]; then
    TASK_ROOT="${TASK_ROOT_OR_DATASET_DIR}"
    dataset_dir="$(cd "${TASK_ROOT}/../.." && pwd)"
  else
    dataset_dir="${TASK_ROOT_OR_DATASET_DIR}"
    TASK_ROOT="${dataset_dir}/${EXP_ID}/prepared/public"
  fi
else
  TASK_ROOT="./data/${EXP_ID}/prepared/public"
  dataset_dir=""
fi

if [ "${USE_GRADING_SERVER}" = "True" ] || [ "${USE_GRADING_SERVER}" = "true" ]; then
  if [ -z "${dataset_dir}" ]; then
    echo "Error: dataset_dir is required when USE_GRADING_SERVER=True"
    exit 1
  fi

  export DATASET_DIR="${dataset_dir}"
  bash "$ROOT/launch_server.sh" "${SERVER_ID}"

  BASE_PORT=5005
  GRADING_SERVER_PORT=$((BASE_PORT + SERVER_ID))
  export GRADING_SERVER_PORT

  echo "Waiting for grading server on port ${GRADING_SERVER_PORT} ..."
  MAX_WAIT=30
  WAITED=0
  while [ $WAITED -lt $MAX_WAIT ]; do
      if curl -s "http://127.0.0.1:${GRADING_SERVER_PORT}/health" > /dev/null 2>&1; then
          echo "Grading server ready (port ${GRADING_SERVER_PORT})."
          break
      fi
      sleep 1
      WAITED=$((WAITED + 1))
  done
  if [ $WAITED -ge $MAX_WAIT ]; then
      echo "Warning: grading server may not be ready yet, proceeding anyway ..."
  fi
else
  echo "Grading server disabled (USE_GRADING_SERVER=${USE_GRADING_SERVER})."
fi

MEMORY_INDEX=0
start_cpu=0
CPUS_PER_TASK=21
TIME_LIMIT_SECS=43200

export MEMORY_INDEX
format_time() {
  local t=$1
  echo "$((t/3600))hrs $(((t%3600)/60))mins $((t%60))secs"
}
export TIME_LIMIT=$(format_time $TIME_LIMIT_SECS)
export STEP_LIMIT=500

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
CLOSEST_EXP_NAME="${TIMESTAMP}_${EXP_ID}"

# HuggingFace cache (optional, point to a shared directory)
# export HF_ENDPOINT=https://huggingface.co
# export HF_DATASETS_CACHE=/path/to/hf_cache
# export HF_MODELS_CACHE=/path/to/hf_cache
# export HUGGINGFACE_HUB_CACHE=/path/to/hf_cache
# export TRANSFORMERS_CACHE=/path/to/hf_cache

CUDA_VISIBLE_DEVICES=$MEMORY_INDEX timeout --foreground --signal=TERM --kill-after=10s "${TIME_LIMIT_SECS}s" python run.py \
  exp_id="${EXP_ID}" \
  dataset_dir="${dataset_dir}" \
  data_dir="${TASK_ROOT}" \
  desc_file="${TASK_ROOT}/description.md" \
  exp_name="${EXP_ID}" \
  use_grading_server="${USE_GRADING_SERVER}" \
  start_cpu_id="${start_cpu}" \
  cpu_number="${CPUS_PER_TASK}"

RUN_EXIT=$?

if [ $RUN_EXIT -eq 124 ]; then
  echo "Timed out after $TIME_LIMIT"
elif [ $RUN_EXIT -eq 130 ]; then
  echo "Interrupted."
  exit 130
elif [ $RUN_EXIT -ne 0 ]; then
  echo "Run failed with exit code: $RUN_EXIT"
fi

echo "Running submission fusion ..."
python utils/submission_fusion_utils.py \
  --task_id "${EXP_ID}" \
  --exp_name "${CLOSEST_EXP_NAME}"
