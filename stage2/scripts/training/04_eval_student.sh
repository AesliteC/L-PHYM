#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

resolve_student
print_config
echo "[INFO] Evaluating student: ${STUDENT_RUN}/${STUDENT_CHECKPOINT}"

run_isaaclab -p scripts/rsl_rl/eval.py \
    --num_envs "${NUM_ENVS_EVAL}" \
    --reference_motion_path "${REFERENCE_MOTION_PATH}" \
    --robot "${ROBOT}" \
    --student_player \
    --student_path "${STUDENT_RUN}" \
    --student_checkpoint "${STUDENT_CHECKPOINT}" \
    --headless \
    "$@"
