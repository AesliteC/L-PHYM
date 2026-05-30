#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

resolve_teacher
require_file "${ISAACLAB_SH}" "Set ISAACLAB_PATH=/path/to/IsaacLab."
print_config
echo "[INFO] Exporting teacher ONNX via play.py: ${TEACHER_RUN}/${TEACHER_CHECKPOINT}"
echo "[INFO] Expected output: ${TEACHER_RUN}/exported/policy.onnx"

timeout "${EXPORT_TIMEOUT:-60s}" \
    "${ISAACLAB_SH}" -p scripts/rsl_rl/play.py \
        --num_envs 1 \
        --reference_motion_path "${REFERENCE_MOTION_PATH}" \
        --robot "${ROBOT}" \
        --teacher_policy.resume_path "${TEACHER_RUN}" \
        --teacher_policy.checkpoint "${TEACHER_CHECKPOINT}" \
        --headless \
        "$@" || true

require_file "${TEACHER_RUN}/exported/policy.onnx" "ONNX export is triggered when play.py loads the teacher policy."
echo "[INFO] Exported: ${TEACHER_RUN}/exported/policy.onnx"
