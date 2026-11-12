# This file is sourced by many scripts in this repo
# It sets some system-specific paths and environments
# Modify this to match your setup

source /cluster/apps/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate hfmt
# No cluster cuDNN module: the pip torch wheel bundles its own (9.10.2 for torch 2.10.0+cu128), and an older
# one on LD_LIBRARY_PATH (cudnn/9.1.1.17_cuda12) makes any cuDNN call fail, e.g. a fp16 convolution
# (Qwen3.5 on a V100). features/15_running_on_v100/data.md, T2a.
module load cuda/toolkit/12.8.1-1 nccl/2.21.5-1_cuda12.4

