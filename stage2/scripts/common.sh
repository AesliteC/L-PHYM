#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export ISAACLAB_PATH="${ISAACLAB_PATH:-/data/xiyuanyang/IsaacLab4HOVER}"
ISAACLAB_SH="${ISAACLAB_SH:-${ISAACLAB_PATH}/isaaclab.sh}"

ROBOT="${ROBOT:-h1}"
MOTION_DIR="${MOTION_DIR:-neural_wbc/data/data/motions}"
MOTION="${MOTION:-long_2}"
REFERENCE_MOTION_PATH="${REFERENCE_MOTION_PATH:-}"

if [[ $# -gt 0 && "${1}" != -* ]]; then
    MOTION="$1"
    shift
fi

TEACHER_ROOT="${TEACHER_ROOT:-logs/teacher}"
STUDENT_ROOT="${STUDENT_ROOT:-logs/student}"

NUM_ENVS_TRAIN="${NUM_ENVS_TRAIN:-1024}"
NUM_ENVS_EVAL="${NUM_ENVS_EVAL:-10}"
NUM_ENVS_PLAY="${NUM_ENVS_PLAY:-1}"

TEACHER_RUN="${TEACHER_RUN:-}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-}"
STUDENT_RUN="${STUDENT_RUN:-}"
STUDENT_CHECKPOINT="${STUDENT_CHECKPOINT:-}"

cd "${REPO_ROOT}"

require_file() {
    local path="$1"
    local hint="${2:-}"
    if [[ ! -e "${path}" ]]; then
        echo "[ERROR] Missing: ${path}" >&2
        if [[ -n "${hint}" ]]; then
            echo "        ${hint}" >&2
        fi
        exit 1
    fi
}

latest_run_dir() {
    local root="$1"
    find "${root}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1
}

latest_checkpoint() {
    local run_dir="$1"
    find "${run_dir}" -maxdepth 1 -type f -name 'model_*.pt' 2>/dev/null \
        | sed -n 's#^\(.*model_\([0-9][0-9]*\)\.pt\)$#\2 \1#p' \
        | sort -n \
        | tail -n 1 \
        | cut -d' ' -f2-
}

resolve_motion() {
    if [[ -z "${REFERENCE_MOTION_PATH}" ]]; then
        if [[ "${MOTION}" == /* || "${MOTION}" == */* ]]; then
            REFERENCE_MOTION_PATH="${MOTION}"
        elif [[ "${MOTION}" == *.pkl ]]; then
            REFERENCE_MOTION_PATH="${MOTION_DIR}/${MOTION}"
        else
            REFERENCE_MOTION_PATH="${MOTION_DIR}/${MOTION}.pkl"
        fi
    fi
    require_file "${REFERENCE_MOTION_PATH}" "Set REFERENCE_MOTION_PATH=/path/to/file.pkl or MOTION=name_without_pkl."
}

resolve_teacher() {
    if [[ -z "${TEACHER_RUN}" ]]; then
        TEACHER_RUN="$(latest_run_dir "${TEACHER_ROOT}")"
    fi
    if [[ -z "${TEACHER_RUN}" ]]; then
        echo "[ERROR] Could not find a teacher run under ${TEACHER_ROOT}." >&2
        echo "        Set TEACHER_RUN=/path/to/run and TEACHER_CHECKPOINT=model_x.pt." >&2
        exit 1
    fi
    require_file "${TEACHER_RUN}/config.json" "Teacher run should contain config.json."

    if [[ -z "${TEACHER_CHECKPOINT}" ]]; then
        local checkpoint_path
        checkpoint_path="$(latest_checkpoint "${TEACHER_RUN}")"
        if [[ -z "${checkpoint_path}" ]]; then
            echo "[ERROR] Could not find model_*.pt under ${TEACHER_RUN}." >&2
            exit 1
        fi
        TEACHER_CHECKPOINT="$(basename "${checkpoint_path}")"
    fi
    require_file "${TEACHER_RUN}/${TEACHER_CHECKPOINT}" "Set TEACHER_CHECKPOINT=model_x.pt."
}

resolve_student() {
    if [[ -z "${STUDENT_RUN}" ]]; then
        STUDENT_RUN="$(latest_run_dir "${STUDENT_ROOT}")"
    fi
    if [[ -z "${STUDENT_RUN}" ]]; then
        echo "[ERROR] Could not find a student run under ${STUDENT_ROOT}." >&2
        echo "        Set STUDENT_RUN=/path/to/run and STUDENT_CHECKPOINT=model_x.pt." >&2
        exit 1
    fi
    require_file "${STUDENT_RUN}/config.json" "Student run should contain config.json."

    if [[ -z "${STUDENT_CHECKPOINT}" ]]; then
        local checkpoint_path
        checkpoint_path="$(latest_checkpoint "${STUDENT_RUN}")"
        if [[ -z "${checkpoint_path}" ]]; then
            echo "[ERROR] Could not find model_*.pt under ${STUDENT_RUN}." >&2
            exit 1
        fi
        STUDENT_CHECKPOINT="$(basename "${checkpoint_path}")"
    fi
    require_file "${STUDENT_RUN}/${STUDENT_CHECKPOINT}" "Set STUDENT_CHECKPOINT=model_x.pt."
}

print_config() {
    echo "[INFO] REPO_ROOT=${REPO_ROOT}"
    echo "[INFO] ISAACLAB_PATH=${ISAACLAB_PATH}"
    echo "[INFO] MOTION=${MOTION}"
    echo "[INFO] REFERENCE_MOTION_PATH=${REFERENCE_MOTION_PATH}"
}

run_isaaclab() {
    require_file "${ISAACLAB_SH}" "Set ISAACLAB_PATH=/path/to/IsaacLab."
    "${ISAACLAB_SH}" "$@"
}

resolve_motion
