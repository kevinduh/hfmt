#!/bin/bash
# One inference job (feature 09): decode with the saved config in inference folder number
# $SLURM_ARRAY_TASK_ID of the folders given as arguments. Started by egs/launch.sh inf (sbatch, or LOCAL=1).
source "${HFMT_ROOT}/install/path.sh"
export PYTHONPATH="${HFMT_ROOT}:${PYTHONPATH}"
cd "${HFMT_ROOT}"
shift "${SLURM_ARRAY_TASK_ID:-0}"
exec python -m hfmt.inf_translation --config-dir "$1" --config-name inf_config hydra.run.dir="$1"
