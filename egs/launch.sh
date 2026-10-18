#!/bin/bash
# Launch SFT training, inference or an adapter merge from a login node (features 08-10).
#
# Training:   bash egs/launch.sh sft <experiment.yaml> [overrides]
#   bash egs/launch.sh sft egs/mmtc/fr-en/sft_lora.yaml                 # 1 job
#   bash egs/launch.sh sft egs/mmtc/fr-en/sft_lora.sweep.yaml           # its full grid
#   bash egs/launch.sh sft egs/mmtc/fr-en/sft_lora.yaml peft.lora.lora_r=8,16   # comma list -> grid
#   bash egs/launch.sh sft <run folder>/run_config.yaml [overrides]     # re-run a saved job
# hfmt/grid.py expands the grid, checks and saves every job's config in its run folder
# (<experiment folder>/outputs/<run_name>[-<idx>]/run_config.yaml); jobs run egs/job.sh sft.
# Machine-specific settings (HFMT_DATA_ROOT) come from <experiment folder>/local_setting.sh
# (untracked; copy local_setting.sh.example). Only needed here: saved configs hold resolved paths.
#
# Inference:  bash egs/launch.sh inf <run folder>/run_config.yaml [overrides]   (one model per call)
#   decodes the run's trained model/. Or, with an experiment yaml, the downloaded checkpoint as is
#   (zero-shot; the peft option sets the precision, e.g. peft=lora_default = bf16):
#        bash egs/launch.sh inf egs/mmtc/fr-en/sft_lora.yaml peft=lora_default
#   output then goes to <output_root>/<run_name>/base_ckpt_inf/<inf_name>/.
#   e.g. decode.num_beams=5 inf_name=beam5, or other test sets (null = no reference):
#        'data.test.src=[/path/x.fr,/path/y.fr]' 'data.test.trg=[/path/x.en,null]' inf_name=xy
# Output: <run folder>/inf/<inf_name>/ (test<i>.pred.trg per test file, scores.json,
# inf_config.yaml); run_config.yaml is never changed. The job runs egs/job.sh inf. A whole sweep:
#   for rc in egs/mmtc/fr-en/outputs/sft_lora_sweep-*/run_config.yaml; do bash egs/launch.sh inf $rc; done
#
# Merge:      bash egs/launch.sh merge <run folder>/run_config.yaml [slurm.* overrides]
#   folds a lora/qlora run's model/ into its base -> <run folder>/model_merged/ (full bf16 weights,
#   tokenizer, merge_info.yaml), usable as `checkpoint:` in another experiment (a second SFT round)
#   or decoded zero-shot via `launch.sh inf <experiment>.yaml`. The job runs egs/job.sh merge.
#
# Each way one Slurm array is submitted and this returns. Resources come from the config's
# slurm: block (e.g. slurm.gres=gpu:l40s:1). LOCAL=1 runs the jobs here, one after another,
# instead; DRYRUN=1 prints the sbatch command without submitting.
set -eo pipefail

usage="usage: bash egs/launch.sh <sft|inf|merge> <yaml> [overrides...]"
kind=$1
case "$kind" in sft|inf|merge) ;; *) echo "$usage" >&2; exit 1 ;; esac
shift
export HFMT_ROOT=${HFMT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
# The yaml path may be relative to the current dir, relative to the repo root, or absolute.
if [ -f "$1" ]; then yaml=$(realpath "$1")
elif [ -n "$1" ] && [ -f "${HFMT_ROOT}/$1" ]; then yaml=$(realpath "${HFMT_ROOT}/$1")
else
    echo "$usage  (not found: '$1')" >&2
    exit 1
fi
shift
# Machine-specific settings (HFMT_DATA_ROOT) from the yaml's folder, for any kind. An experiment
# folder has local_setting.sh; a run folder doesn't and needs none (its saved paths are resolved).
# Unset first so a value left in your shell can't point this experiment at another folder's data.
unset HFMT_DATA_ROOT
if [ -f "$(dirname "$yaml")/local_setting.sh" ]; then source "$(dirname "$yaml")/local_setting.sh"; fi
if [ "$kind" = sft ]; then grid_args=("$yaml"); else grid_args=("--$kind" "$yaml"); fi   # grid.py --inf / --merge
source "${HFMT_ROOT}/install/path.sh"
export PYTHONPATH="${HFMT_ROOT}:${PYTHONPATH}"
cd "${HFMT_ROOT}"

grid=$(python -m hfmt.grid "${grid_args[@]}" "$@")   # sets JOBS and SBATCH_ARGS
eval "$grid"

# Every job runs egs/job.sh <kind> <all job folders>; each array task picks its own folder.
job=(egs/job.sh "$kind")
if [ -n "${LOCAL}" ]; then
    for i in "${!JOBS[@]}"; do
        echo "job $i: ${JOBS[$i]}" >&2
        log="${JOBS[$i]}/local.log"; [ "$kind" = merge ] && log="${JOBS[$i]}/merge-local.log"
        SLURM_ARRAY_TASK_ID=$i bash "${job[@]}" "${JOBS[@]}" > "$log" 2>&1 \
            || echo "job $i failed (see $log)" >&2
    done
elif [ -n "${DRYRUN}" ]; then
    echo sbatch "${SBATCH_ARGS[@]}" "${job[@]}" "${JOBS[@]}"
else
    sbatch "${SBATCH_ARGS[@]}" "${job[@]}" "${JOBS[@]}"
fi
