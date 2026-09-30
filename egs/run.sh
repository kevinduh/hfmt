#!/bin/sh

# Generic Hydra launcher for HFMT experiments (feature 00_hydra).
# The experiment recipe lives in conf/experiment/<name>.yaml; this script just
# activates the env and submits to Slurm via the submitit launcher.
#
# Run from a LOGIN node; submitit submits the Slurm job for you (do NOT sbatch this):
#   export HFMT_ROOT=$(pwd)
#   bash egs/run.sh mmtc_fr-en_sft1
#
# Override any knob on the CLI (forwarded via "$@"), e.g.:
#   bash egs/run.sh mmtc_fr-en_sft1 model.lora_r=16 train.learning_rate=2e-5
#
# Run locally/in-process instead of submitting (e.g. already on a GPU node):
#   python ${HFMT_ROOT}/hfmt/sft_translation.py +experiment=mmtc_fr-en_sft1

set -e

if [ -z "$1" ]; then
    echo "usage: bash egs/run.sh <experiment> [hydra overrides...]" >&2
    echo "       <experiment> is a file in conf/experiment/ (without .yaml)" >&2
    exit 1
fi

experiment="$1"
shift

source ${HFMT_ROOT}/install/path.sh

python ${HFMT_ROOT}/hfmt/sft_translation.py -m \
    +experiment="${experiment}" \
    hydra/launcher=slurm \
    "$@"
