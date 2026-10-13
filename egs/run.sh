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
#   cd ${HFMT_ROOT} && python -m hfmt.sft_translation +experiment=mmtc_fr-en_sft1
#
# NOTE: we run as a module (python -m hfmt.sft_translation) with HFMT_ROOT on
# PYTHONPATH so that submitit can re-import hfmt.* when it unpickles the job on the
# compute node.

set -e

if [ -z "$1" ]; then
    echo "usage: bash egs/run.sh <experiment> [hydra overrides...]" >&2
    echo "       <experiment> is a file in conf/experiment/ (without .yaml)" >&2
    exit 1
fi

experiment="$1"
shift

source ${HFMT_ROOT}/install/path.sh
export PYTHONPATH="${HFMT_ROOT}:${PYTHONPATH}"
cd "${HFMT_ROOT}"

# Sweep toggle (feature 04): the experiment carries an inline `sweep: {enabled, params}`
# block. sweep_probe.py composes it and, when enabled, prints bare `<axis>=<csv>`
# overrides that the basic sweeper fans out into the job grid (disabled -> nothing ->
# one job). We always launch --multirun (the submitit Slurm launcher only runs in
# multirun), so the grid is what makes a sweep, not the flag.
#
# Forward only the `sweep.enabled` toggle to the probe so it can be flipped from the
# CLI (`sweep.enabled=true`). Everything else stays in "$@" and goes to the launch:
# ad-hoc comma-list axes (e.g. lora.lora_r=8,16, or basemodel=qwen2.5-7b,qwen2.5-1.5b
# to sweep whole component files) are expanded by the sweeper directly. We do NOT
# forward sweep.params.* here -- a comma-list value breaks the probe's single-run,
# type-validated compose; author extra axes in the experiment YAML or pass them as
# bare overrides instead.
probe_overrides=""
for arg in "$@"; do
    case "$arg" in
        sweep.enabled=*) probe_overrides="${probe_overrides} ${arg}" ;;
    esac
done

sweep_axes=$(python -m hfmt.sweep_probe "${experiment}" ${probe_overrides})

python -m hfmt.sft_translation --multirun \
    +experiment="${experiment}" \
    hydra/launcher=slurm \
    ${sweep_axes} \
    "$@"
