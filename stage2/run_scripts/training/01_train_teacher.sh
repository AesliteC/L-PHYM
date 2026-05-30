#!/bin/bash

export ISAACLAB_PATH="/data/xiyuanyang/IsaacLab4HOVER"

/data/xiyuanyang/IsaacLab4HOVER/isaaclab.sh -p scripts/rsl_rl/train_teacher_policy.py \
    --num_envs 1024 \
    --reference_motion_path neural_wbc/data/data/motions/stable_punch.pkl \
    --headless 

# neural_wbc/data/data/motions
# /data/xiyuanyang/HOVER/third_party/human2humanoid/data/h1/amass_all.pkl