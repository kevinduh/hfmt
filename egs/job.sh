#!/bin/bash
# One job of a launch (features 08-12):  bash egs/job.sh <sft|sft_resume|inf|merge> <folder>...
# Started by egs/launch.sh, as an sbatch array or with LOCAL=1. Each array task picks folder number
# $SLURM_ARRAY_TASK_ID of the folders given (0 when run by hand) and runs one Python entry point:
#   sft         hfmt.sft_translation on <folder>/run_config.yaml   (a run folder; writes into it)
#   sft_resume  the same, continuing from the folder's last checkpoint (HFMT_SFT_RESUME=1)
#   inf         hfmt.inf_translation on <folder>/inf_config.yaml   (an inference folder; writes into it)
#   merge       hfmt.merge_model <folder>                          (a run folder; writes model_merged/)
# Under Slurm it first waits until its GPU is usable, and a requeued sft task continues from its
# folder's last checkpoint (sft_resume) instead of starting over.
source "${HFMT_ROOT}/install/path.sh"
export PYTHONPATH="${HFMT_ROOT}:${PYTHONPATH}"
cd "${HFMT_ROOT}"
kind=$1
shift
shift "${SLURM_ARRAY_TASK_ID:-0}"

# GPUs in exclusive-process mode (one process per GPU) make tasks that start at the
# same moment race for GPUs, and a loser fails with "CUDA-capable device(s) is/are busy or
# unavailable". Slurm doesn't retry a task that failed on its own, so wait here, before
# anything is written (e.g. a W&B run).
if [ -n "${SLURM_JOB_ID}" ]; then
    for try in 1 2 3 4 5 6; do
        err=$(python -c "import torch; torch.zeros(1, device='cuda')" 2>&1) && break
        if [ "$try" = 6 ]; then echo "$err" >&2; echo "job.sh: GPU still not usable after $try tries" >&2; exit 1; fi
        echo "job.sh: GPU not usable (try $try); retrying in 30 s" >&2
        sleep 30
    done
fi

# A requeued task (node failure, preemption; SLURM_RESTART_COUNT > 0) is started again from the top.
# For training, continue from the last checkpoint instead of overwriting the run from step 0.
if [ "$kind" = sft ] && [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ] && compgen -G "$1/checkpoint-*" > /dev/null; then
    echo "job.sh: requeued (restart ${SLURM_RESTART_COUNT}); resuming from the last checkpoint" >&2
    kind=sft_resume
fi

case "$kind" in
    sft)        exec python -m hfmt.sft_translation --config-dir "$1" --config-name run_config hydra.run.dir="$1" ;;
    sft_resume) HFMT_SFT_RESUME=1 exec python -m hfmt.sft_translation --config-dir "$1" --config-name run_config hydra.run.dir="$1" ;;
    inf)        exec python -m hfmt.inf_translation --config-dir "$1" --config-name inf_config hydra.run.dir="$1" ;;
    merge)      exec python -m hfmt.merge_model "$1" ;;
    *)          echo "usage: bash egs/job.sh <sft|sft_resume|inf|merge> <folder>..." >&2; exit 1 ;;
esac
