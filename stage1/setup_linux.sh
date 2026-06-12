#!/usr/bin/env bash
set -euo pipefail

# Linux translation of setup.cmd. Run this from the MoConVQ directory:
#   cd /home/chenjie/cc/robotics/MoConVQ
#   bash setup_linux.sh

echo "building conda environment"
conda env create -f requirements.yml

echo "activating conda environment"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate moconvq

echo "installing pytorch"
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -y cuda-nvcc=12.4.131 -c nvidia

echo "installing text-to-motion extras"
pip install transformers sentencepiece

echo "building rotation library"
pip install -e ./diff-quaternion/TorchRotation

echo "building VclSimuBackend"
(
  cd ModifyODESrc
  bash ./clear.sh
  pip install -e .
)

echo "building moconvq core"
pip install -e .

echo "done"
