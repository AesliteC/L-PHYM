#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

resolve_teacher
print_config
echo "[INFO] Evaluating teacher: ${TEACHER_RUN}/${TEACHER_CHECKPOINT}"

run_isaaclab -p scripts/rsl_rl/eval.py \
    --num_envs "${NUM_ENVS_EVAL}" \
    --reference_motion_path "${REFERENCE_MOTION_PATH}" \
    --robot "${ROBOT}" \
    --teacher_policy.resume_path "${TEACHER_RUN}" \
    --teacher_policy.checkpoint "${TEACHER_CHECKPOINT}" \
    --headless \
    "$@"
