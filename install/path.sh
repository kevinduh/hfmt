# This file is sourced by many scripts in this repo
# It sets some system-specific paths and environments
# Modify this to match your setup

source /cluster/apps/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate hfmt
module load cuda/toolkit/12.8.1-1 cudnn/9.1.1.17_cuda12 nccl/2.21.5-1_cuda12.4

