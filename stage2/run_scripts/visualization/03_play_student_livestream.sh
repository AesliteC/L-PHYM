#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

resolve_student
print_config
echo "[INFO] Playing student with livestream: ${STUDENT_RUN}/${STUDENT_CHECKPOINT}"
echo "[INFO] Use NUM_ENVS_PLAY=1 for clean shots; pass --randomize if desired."

run_isaaclab -p scripts/rsl_rl/play.py \
    --num_envs "${NUM_ENVS_PLAY}" \
    --reference_motion_path "${REFERENCE_MOTION_PATH}" \
    --robot "${ROBOT}" \
    --student_player \
    --student_path "${STUDENT_RUN}" \
    --student_checkpoint "${STUDENT_CHECKPOINT}" \
    --livestream "${LIVESTREAM:-2}" \
    "$@"
