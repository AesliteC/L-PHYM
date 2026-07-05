#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

resolve_teacher
print_config

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
MOTION_LABEL="${MOTION_LABEL:-$(basename "${REFERENCE_MOTION_PATH}" .pkl)}"
VIDEO_FOLDER="${VIDEO_FOLDER:-videos/${TIMESTAMP}_${MOTION_LABEL}}"
VIDEO_LENGTH="${VIDEO_LENGTH:-0}"
VIDEO_FPS="${VIDEO_FPS:-50}"
VIDEO_WIDTH="${VIDEO_WIDTH:-7680}"
VIDEO_HEIGHT="${VIDEO_HEIGHT:-4320}"
FOLLOW_ROBOT_CAMERA="${FOLLOW_ROBOT_CAMERA:-1}"
if [[ "${FOLLOW_ROBOT_CAMERA}" == "1" ]]; then
    CAMERA_EYE="${CAMERA_EYE:-3.0 -4.0 1.2}"
    CAMERA_TARGET="${CAMERA_TARGET:-0.0 0.0 -0.1}"
else
    CAMERA_EYE="${CAMERA_EYE:-3.0 -4.0 2.2}"
    CAMERA_TARGET="${CAMERA_TARGET:-0.0 0.0 0.9}"
fi
ANTIALIASING_MODE="${ANTIALIASING_MODE:-TAA}"
STOP_ON_DONE="${STOP_ON_DONE:-1}"
RECORDING_SCENE="${RECORDING_SCENE:-1}"
HIDE_REF_MARKERS="${HIDE_REF_MARKERS:-1}"
ROBOT_COLOR="${ROBOT_COLOR:-0.95 0.68 0.48}"
ROBOT_METALLIC="${ROBOT_METALLIC:-0.03}"
ROBOT_ROUGHNESS="${ROBOT_ROUGHNESS:-0.45}"
RENDER_SAMPLES_PER_PIXEL="${RENDER_SAMPLES_PER_PIXEL:-2}"
MOTION_START_IDX="${MOTION_START_IDX:-0}"
MOTION_COUNT="${MOTION_COUNT:-10}"

echo "[INFO] Recording teacher MP4: ${TEACHER_RUN}/${TEACHER_CHECKPOINT}"
echo "[INFO] Video folder: ${VIDEO_FOLDER}"
read -r -a CAMERA_EYE_ARGS <<< "${CAMERA_EYE}"
read -r -a CAMERA_TARGET_ARGS <<< "${CAMERA_TARGET}"
EXTRA_ARGS=()
if [[ "${STOP_ON_DONE}" == "0" ]]; then
    EXTRA_ARGS+=(--no_stop_on_done)
fi
if [[ "${RECORDING_SCENE}" == "1" ]]; then
    EXTRA_ARGS+=(--recording_scene)
fi
if [[ "${HIDE_REF_MARKERS}" == "1" ]]; then
    EXTRA_ARGS+=(--hide_ref_markers)
fi
if [[ "${ROBOT_COLOR}" != "none" ]]; then
    read -r -a ROBOT_COLOR_ARGS <<< "${ROBOT_COLOR}"
    EXTRA_ARGS+=(
        --robot_color "${ROBOT_COLOR_ARGS[@]}"
        --robot_metallic "${ROBOT_METALLIC}"
        --robot_roughness "${ROBOT_ROUGHNESS}"
    )
fi
if [[ -n "${MOTION_COUNT}" ]]; then
    EXTRA_ARGS+=(--motion_count "${MOTION_COUNT}")
fi
if [[ "${FOLLOW_ROBOT_CAMERA}" == "1" ]]; then
    EXTRA_ARGS+=(--follow_robot_camera)
fi

run_isaaclab -p scripts/rsl_rl/record_teacher_mp4.py \
    --num_envs 1 \
    --reference_motion_path "${REFERENCE_MOTION_PATH}" \
    --robot "${ROBOT}" \
    --teacher_policy.resume_path "${TEACHER_RUN}" \
    --teacher_policy.checkpoint "${TEACHER_CHECKPOINT}" \
    --video_folder "${VIDEO_FOLDER}" \
    --video_length "${VIDEO_LENGTH}" \
    --fps "${VIDEO_FPS}" \
    --width "${VIDEO_WIDTH}" \
    --height "${VIDEO_HEIGHT}" \
    --motion_start_idx "${MOTION_START_IDX}" \
    --camera_eye "${CAMERA_EYE_ARGS[@]}" \
    --camera_target "${CAMERA_TARGET_ARGS[@]}" \
    --antialiasing_mode "${ANTIALIASING_MODE}" \
    --render_samples_per_pixel "${RENDER_SAMPLES_PER_PIXEL}" \
    --headless \
    --enable_cameras \
    "${EXTRA_ARGS[@]}" \
    "$@"
