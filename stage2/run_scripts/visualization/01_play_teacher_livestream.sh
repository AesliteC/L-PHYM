#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

resolve_teacher
print_config
echo "[INFO] Playing teacher with livestream: ${TEACHER_RUN}/${TEACHER_CHECKPOINT}"
echo "[INFO] Use NUM_ENVS_PLAY=1 for clean shots; pass --randomize if desired."

LIVESTREAM_MODE="${LIVESTREAM:-2}"
LIVESTREAM_KIT_ARGS="${LIVESTREAM_KIT_ARGS:-}"
if [[ -n "${LIVESTREAM_HTTP_PORT:-}" ]]; then
    LIVESTREAM_KIT_ARGS+=" --/exts/omni.services.transport.server.http/port=${LIVESTREAM_HTTP_PORT}"
fi
if [[ -n "${LIVESTREAM_SIGNAL_PORT:-}" ]]; then
    LIVESTREAM_KIT_ARGS+=" --/app/livestream/port=${LIVESTREAM_SIGNAL_PORT}"
    LIVESTREAM_KIT_ARGS+=" --/exts/omni.kit.livestream.app/primaryStream/signalPort=${LIVESTREAM_SIGNAL_PORT}"
fi
if [[ -n "${LIVESTREAM_STREAM_PORT:-}" ]]; then
    LIVESTREAM_KIT_ARGS+=" --/app/livestream/publicEndpointPort=${LIVESTREAM_STREAM_PORT}"
    LIVESTREAM_KIT_ARGS+=" --/exts/omni.kit.livestream.app/primaryStream/streamPort=${LIVESTREAM_STREAM_PORT}"
fi
if [[ -n "${LIVESTREAM_PUBLIC_IP:-}" ]]; then
    LIVESTREAM_KIT_ARGS+=" --/app/livestream/publicEndpointAddress=${LIVESTREAM_PUBLIC_IP}"
    LIVESTREAM_KIT_ARGS+=" --/exts/omni.kit.livestream.app/primaryStream/publicIp=${LIVESTREAM_PUBLIC_IP}"
fi
EXTRA_ARGS=()
if [[ -n "${LIVESTREAM_KIT_ARGS// }" ]]; then
    EXTRA_ARGS+=(--kit_args "${LIVESTREAM_KIT_ARGS}")
fi

run_isaaclab -p scripts/rsl_rl/play.py \
    --num_envs "${NUM_ENVS_PLAY}" \
    --reference_motion_path "${REFERENCE_MOTION_PATH}" \
    --robot "${ROBOT}" \
    --teacher_policy.resume_path "${TEACHER_RUN}" \
    --teacher_policy.checkpoint "${TEACHER_CHECKPOINT}" \
    --livestream "${LIVESTREAM_MODE}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
