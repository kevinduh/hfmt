#!/bin/bash
# One SFT job (features 08, 09): run the saved config in job folder number $SLURM_ARRAY_TASK_ID of
# the folders given as arguments. Started by egs/launch.sh sft (sbatch array, or LOCAL=1).
source "${HFMT_ROOT}/install/path.sh"
export PYTHONPATH="${HFMT_ROOT}:${PYTHONPATH}"
cd "${HFMT_ROOT}"
shift "${SLURM_ARRAY_TASK_ID:-0}"
exec python -m hfmt.sft_translation --config-dir "$1" --config-name run_config hydra.run.dir="$1"
