#!/bin/bash
# One job of a launch (features 08-10):  bash egs/job.sh <sft|inf|merge> <folder>...
# Started by egs/launch.sh, as an sbatch array or with LOCAL=1. Each array task picks folder number
# $SLURM_ARRAY_TASK_ID of the folders given (0 when run by hand) and runs one Python entry point:
#   sft    hfmt.sft_translation on <folder>/run_config.yaml   (a run folder; writes into it)
#   inf    hfmt.inf_translation on <folder>/inf_config.yaml   (an inference folder; writes into it)
#   merge  hfmt.merge_model <folder>                          (a run folder; writes model_merged/)
source "${HFMT_ROOT}/install/path.sh"
export PYTHONPATH="${HFMT_ROOT}:${PYTHONPATH}"
cd "${HFMT_ROOT}"
kind=$1
shift
shift "${SLURM_ARRAY_TASK_ID:-0}"
case "$kind" in
    sft)   exec python -m hfmt.sft_translation --config-dir "$1" --config-name run_config hydra.run.dir="$1" ;;
    inf)   exec python -m hfmt.inf_translation --config-dir "$1" --config-name inf_config hydra.run.dir="$1" ;;
    merge) exec python -m hfmt.merge_model "$1" ;;
    *)     echo "usage: bash egs/job.sh <sft|inf|merge> <folder>..." >&2; exit 1 ;;
esac
