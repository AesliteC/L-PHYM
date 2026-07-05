#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

HUMAN2HUMANOID_DIR="third_party/human2humanoid"
AMASS_DIR="$HUMAN2HUMANOID_DIR/data/AMASS/AMASS_Complete"
SMPL_DIR="$HUMAN2HUMANOID_DIR/data/smpl"
SMPL_MODEL_DIR="$SMPL_DIR/SMPL_python_v.1.1.0/smpl/models"
AMASS_FILTERED_DIR="$HUMAN2HUMANOID_DIR/data/AMASS/AMASS_Filtered"
KEEP_FILTERED=0
REFRESH_FILTERED=0

# Function to check and extract AMASS dataset
check_amass() {
    if [ -d "$AMASS_DIR" ]; then
        if find "$AMASS_DIR" -type f -name "*.npz" | grep -q . ; then
            echo "AMASS dataset is already extracted and ready."
        elif compgen -G "$AMASS_DIR/*.tar.bz2" > /dev/null || compgen -G "$AMASS_DIR/*.zip" > /dev/null; then
            echo "Extracting compressed files..."
            # Extract .tar.bz2 files
            find "$AMASS_DIR" -name "*.tar.bz2" -exec tar -xvjf {} -C "$AMASS_DIR" \;
            # Extract .zip files
            find "$AMASS_DIR" -name "*.zip" -exec unzip -o {} -d "$AMASS_DIR" \;
        else
            echo "Please download the AMASS dataset in the 'SMPL + H G' format from https://amass.is.tue.mpg.de/index.html and place it under $AMASS_DIR"
            exit 1
        fi
    else
        echo "$AMASS_DIR folder does not exist. Please create it and download the AMASS dataset in the 'SMPL + H G' format from https://amass.is.tue.mpg.de/index.html there."
        exit 1
    fi
}

check_files_exist() {
    for file in "$@"; do
        if [ ! -f "$file" ]; then
            return 1
        fi
    done
    return 0
}

# Function to check and extract SMPL files
check_smpl() {
    FEMALE_MODEL="$SMPL_MODEL_DIR/basicmodel_f_lbs_10_207_0_v1.1.0.pkl"
    MALE_MODEL="$SMPL_MODEL_DIR/basicmodel_m_lbs_10_207_0_v1.1.0.pkl"
    NEUTRAL_MODEL="$SMPL_MODEL_DIR/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"
    MODELS=($FEMALE_MODEL $MALE_MODEL $NEUTRAL_MODEL)

    RENAMED_FEMALE_MODEL="$SMPL_DIR/SMPL_FEMALE.pkl"
    RENAMED_MALE_MODEL="$SMPL_DIR/SMPL_MALE.pkl"
    RENAMED_NEUTRAL_MODEL="$SMPL_DIR/SMPL_NEUTRAL.pkl"
    RENAMED_MODELS=($RENAMED_FEMALE_MODEL $RENAMED_MALE_MODEL $RENAMED_NEUTRAL_MODEL)

    if [ -d "$SMPL_DIR" ]; then
        if check_files_exist "${RENAMED_MODELS[@]}"; then
            echo "SMPL files are already available."
        else
            if [ -f "$SMPL_DIR/SMPL_python_v.1.1.0.zip" ]; then
                echo "Extracting SMPL_python_v.1.1.0.zip..."
                # Create model directory if it doesn't exist
                mkdir -p "$SMPL_MODEL_DIR"
                # -o to overwrite any existing files
                unzip -o "$SMPL_DIR/SMPL_python_v.1.1.0.zip" -d "$SMPL_DIR"

                if check_files_exist "${MODELS[@]}"; then
                    echo "Renaming SMPL model files..."
                    cp "$FEMALE_MODEL" "$RENAMED_FEMALE_MODEL"
                    cp "$MALE_MODEL" "$RENAMED_MALE_MODEL"
                    cp "$NEUTRAL_MODEL" "$RENAMED_NEUTRAL_MODEL"
                else
                    echo "Error: Required SMPL model files not found after extraction"
                    exit 1
                fi
            else
                echo "Please download the SMPL files from https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.1.0.zip and place it under $SMPL_DIR"
                exit 1
            fi
        fi
    else
        echo "$SMPL_DIR folder does not exist. Please create it and download the SMPL files from https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.1.0.zip there."
        exit 1
    fi
}

apply_patch() {
    local target_file=$1
    local patch_file=$2

    echo "Applying $patch_file to $target_file"
    if patch --dry-run --quiet -f "$target_file" < "$patch_file"; then
        echo "Dry run succeeded. Applying the patch..."
        patch "$target_file" < "$patch_file"
        echo "Patch applied successfully."
    fi
}

# Function to prepare filtered motion files
prepare_filtered_motions() {
    local yaml_file=$1
    if [ -f "$yaml_file" ]; then
        # Check if filtered directory exists and has files
        if [ -d "$AMASS_FILTERED_DIR" ] && [ "$(ls -A $AMASS_FILTERED_DIR)" ]; then
            echo "Filtered motion files already exist in $AMASS_FILTERED_DIR:"
            find "$AMASS_FILTERED_DIR" \( -type f -o -type l \) -name "*.npz" -printf "  %P -> %l\n" | sort
            if [ "$REFRESH_FILTERED" -eq 1 ]; then
                echo "Refreshing filtered motion files from $yaml_file..."
                echo "Removing existing filtered motion files..."
                rm -rf "$AMASS_FILTERED_DIR"/*
            elif [ "$KEEP_FILTERED" -eq 1 ]; then
                echo "Keeping existing filtered motion files. The --motions-file list will not be applied."
                return
            else
                read -p "Keep these existing files? If yes, the --motions-file list will not be applied. [y/N] " response
                if [[ ! $response =~ ^[Yy]$ ]]; then
                    echo "Removing existing filtered motion files..."
                    rm -rf "$AMASS_FILTERED_DIR"/*
                else
                    echo "Keeping existing filtered motion files. The --motions-file list will not be applied."
                    return
                fi
            fi
        fi

        echo "Preparing filtered motion files from $yaml_file..."
        echo "Filtered motions will be saved to $AMASS_FILTERED_DIR"
        mkdir -p "$AMASS_FILTERED_DIR"

        # Read yaml file and create symbolic links
        while IFS= read -r line; do
            # Extract the pattern between quotes
            if [[ $line =~ \"(.*)\" ]]; then
                pattern="${BASH_REMATCH[1]}"
                # Check if pattern contains a glob
                if [[ $pattern == *"*"* ]]; then
                    # Use find with -path for glob patterns
                    while IFS= read -r source_file; do
                        relative_path="${source_file#$AMASS_DIR/}"
                        target_dir="$AMASS_FILTERED_DIR/$(dirname "$relative_path")"
                        mkdir -p "$target_dir"
                        if [ -f "$source_file" ]; then
                            ln -s "$(realpath "$source_file")" "$target_dir/$(basename "$source_file")"
                            echo "Linked: $relative_path"
                        else
                            echo "Warning: File not found: $source_file"
                        fi
                    done < <(find "$AMASS_DIR" -path "$AMASS_DIR/$pattern")
                else
                    # Handle exact file paths
                    source_file="$AMASS_DIR/$pattern"
                    relative_path="${pattern}"
                    target_dir="$AMASS_FILTERED_DIR/$(dirname "$relative_path")"
                    mkdir -p "$target_dir"
                    if [ -f "$source_file" ]; then
                        ln -s "$(realpath "$source_file")" "$target_dir/$(basename "$source_file")"
                        echo "Linked: $relative_path"
                    else
                        echo "Warning: File not found: $source_file"
                    fi
                fi
            fi
        done < <(grep -o '"[^"]*"' "$yaml_file")
    fi
}

retarget() {
    # Install required packages to run the retargeting scripts.
    echo "Installing Python requirements..."
    # chumpy's setup.py imports pip during build, which fails with build isolation.
    # Install it first without build isolation to work around this.
    # pip install chumpy --no-build-isolation 2>/dev/null || true
    # pip install -r requirements.txt || return 1
    # export PYTHONPATH="$PWD/phc:${PYTHONPATH:-}"

    # Fit the SMPL shape that matches the H1 kinematics
    echo "Running grad_fit_h1_shape.py..."
    python scripts/data_process/grad_fit_h1_shape.py || return 1

    # Determine which AMASS directory to use
    local amass_dir="data/AMASS/AMASS_Complete"
    local filtered_amass_dir="data/AMASS/AMASS_Filtered"
    if [ -d "$filtered_amass_dir" ] && [ "$(ls -A $filtered_amass_dir)" ]; then
        amass_dir="$filtered_amass_dir"
        echo "Using filtered motion files from $filtered_amass_dir"
    fi

    # Retarget the AMASS dataset based on the corresponding keypoints between fitted SMLP shape and H1
    echo "Running grad_fit_h1.py on $amass_dir..."
    python scripts/data_process/grad_fit_h1.py --amass_root "$amass_dir" || return 1

    return 0
}

# Function to print usage information
print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Retarget AMASS dataset to H1 robot."
    echo ""
    echo "Options:"
    echo "  --motions-file FILE    Specify a YAML file containing a list of motion files to process"
    echo "                         If not provided, all AMASS motion files will be processed"
    echo "  --refresh-filtered     Remove existing filtered motion links before applying --motions-file"
    echo "  --keep-filtered        Reuse existing filtered motion links and ignore --motions-file filtering"
    echo ""
    echo "Example:"
    echo "  $0 --motions-file motions.yaml --refresh-filtered"
}

# Parse command line arguments
YAML_FILE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --motions-file)
            YAML_FILE="$2"
            shift 2
            ;;
        --refresh-filtered)
            REFRESH_FILTERED=1
            shift
            ;;
        --keep-filtered)
            KEEP_FILTERED=1
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Check AMASS dataset
check_amass

# Check SMPL files
check_smpl

# Prepare filtered motion files if yaml file is provided
if [ -n "$YAML_FILE" ]; then
    prepare_filtered_motions "$YAML_FILE"
fi

# Apply patches
# apply_patch $HUMAN2HUMANOID_DIR/requirements.txt third_party/requirements.patch || exit 1
apply_patch $HUMAN2HUMANOID_DIR/scripts/data_process/grad_fit_h1.py third_party/grad_fit_h1.patch || exit 1

echo "Move to $HUMAN2HUMANOID_DIR"
pushd $HUMAN2HUMANOID_DIR

retarget
if [ $? -ne 0 ]; then
    echo "Motion retargeting failed."
else
    echo "Motion retargeting finished. You should find the retargeted dataset at $HUMAN2HUMANOID_DIR/data/h1/amass_all.pkl."
fi

# Change back to the original directory
popd
