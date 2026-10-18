#!/bin/bash
# Launch an SFT experiment (feature 08_hydra_redo). Run from a login node:
#   bash egs/sft.sh egs/mmtc/fr-en/sft_lora.yaml [overrides]        # 1 job
#   bash egs/sft.sh egs/mmtc/fr-en/sft_lora.sweep.yaml [overrides]  # its full grid
#   bash egs/sft.sh egs/mmtc/fr-en/sft_lora.yaml peft.lora.lora_r=8,16  # comma list -> grid
#   bash egs/sft.sh <run folder>/run_config.yaml [overrides]         # re-run a saved job
# hfmt/grid.py expands the grid, checks and saves every job's config in its run folder
# (<experiment folder>/outputs/<run_name>[-<idx>]/run_config.yaml), then this script submits
# one Slurm array of egs/job.sh and returns. Resources come from the config's slurm: block
# (e.g. slurm.gres=gpu:l40s:1). LOCAL=1 runs the jobs here, one after another, instead;
# DRYRUN=1 prints the sbatch command without submitting.
# Machine-specific settings (HFMT_DATA_ROOT) come from <experiment folder>/local_setting.sh
# (untracked; copy local_setting.sh.example). Only needed here: saved configs hold resolved paths.
set -eo pipefail

export HFMT_ROOT=${HFMT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
# The experiment path may be relative to the current dir, relative to the repo root, or absolute.
if [ -f "$1" ]; then exp=$(realpath "$1")
elif [ -n "$1" ] && [ -f "${HFMT_ROOT}/$1" ]; then exp=$(realpath "${HFMT_ROOT}/$1")
else
    echo "usage: bash egs/sft.sh <experiment.yaml> [overrides...]  (not found: '$1')" >&2
    exit 1
fi
shift
# Unset first so a value left in your shell can't point this experiment at another folder's data.
unset HFMT_DATA_ROOT
if [ -f "$(dirname "$exp")/local_setting.sh" ]; then source "$(dirname "$exp")/local_setting.sh"; fi
source "${HFMT_ROOT}/install/path.sh"
export PYTHONPATH="${HFMT_ROOT}:${PYTHONPATH}"
cd "${HFMT_ROOT}"

grid=$(python -m hfmt.grid "$exp" "$@")   # sets JOBS and SBATCH_ARGS
eval "$grid"

if [ -n "${LOCAL}" ]; then
    for i in "${!JOBS[@]}"; do
        echo "job $i: ${JOBS[$i]}" >&2
        SLURM_ARRAY_TASK_ID=$i bash egs/job.sh "${JOBS[@]}" > "${JOBS[$i]}/local.log" 2>&1 \
            || echo "job $i failed (see ${JOBS[$i]}/local.log)" >&2
    done
elif [ -n "${DRYRUN}" ]; then
    echo sbatch "${SBATCH_ARGS[@]}" egs/job.sh "${JOBS[@]}"
else
    sbatch "${SBATCH_ARGS[@]}" egs/job.sh "${JOBS[@]}"
fi
