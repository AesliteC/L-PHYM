#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

resolve_teacher
resolve_student
print_config
echo "[INFO] Resuming student: ${STUDENT_RUN}/${STUDENT_CHECKPOINT}"
echo "[INFO] Using teacher: ${TEACHER_RUN}/${TEACHER_CHECKPOINT}"

run_isaaclab -p scripts/rsl_rl/train_student_policy.py \
    --num_envs "${NUM_ENVS_TRAIN}" \
    --reference_motion_path "${REFERENCE_MOTION_PATH}" \
    --robot "${ROBOT}" \
    --teacher_policy.resume_path "${TEACHER_RUN}" \
    --teacher_policy.checkpoint "${TEACHER_CHECKPOINT}" \
    --student_policy.resume_path "${STUDENT_RUN}" \
    --student_policy.checkpoint "${STUDENT_CHECKPOINT}" \
    --headless \
    "$@"
