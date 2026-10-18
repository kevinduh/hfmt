#!/bin/bash
# Launch SFT training, inference or an adapter merge from a login node (features 08-10).
#
#   bash egs/launch.sh <sft|inf|merge> <yaml> [overrides...]
#
# How it works: hfmt/grid.py checks the yaml + overrides and writes each job's folder (and saved
# config), then prints the job folders and sbatch options; this script submits them as one Slurm
# array of egs/job.sh <kind> and returns. Resources come from the config's slurm: block
# (e.g. slurm.gres=gpu:l40s:1). LOCAL=1 runs the jobs here, one after another, instead;
# DRYRUN=1 prints the sbatch command without submitting.
#
# sft: train, one job per grid point -> <output_root>/<run_name>[-<idx>]/ (output_root defaults
# to <experiment folder>/outputs)
#   bash egs/launch.sh sft egs/mmtc/fr-en/sft_lora.yaml                 # 1 job
#   bash egs/launch.sh sft egs/mmtc/fr-en/sft_lora.sweep.yaml           # its full grid
#   bash egs/launch.sh sft egs/mmtc/fr-en/sft_lora.yaml peft.lora.r=8,16       # comma list -> grid
#   bash egs/launch.sh sft <run folder>/run_config.yaml [overrides]     # re-run a saved job
#
# inf: decode the test sets with one model per call -> test<i>.pred.trg per test file, scores.json
#   bash egs/launch.sh inf <run folder>/run_config.yaml [overrides]     # the run's trained model/
#       -> <run folder>/inf/<inf_name>/ (run_config.yaml is never changed)
#   bash egs/launch.sh inf egs/mmtc/fr-en/sft_lora.yaml peft=lora_default   # the checkpoint as
#       downloaded (zero-shot; the peft option sets the precision: lora_default = bf16)
#       -> <output_root>/<run_name>/base_ckpt_inf/<inf_name>/
#   overrides e.g. decode.num_beams=5 inf_name=beam5, or other test sets (null = no reference):
#       'data.test.src=[/path/x.fr,/path/y.fr]' 'data.test.trg=[/path/x.en,null]' inf_name=xy
#   a whole sweep: for rc in egs/mmtc/fr-en/outputs/sft_lora_sweep-*/run_config.yaml; do
#                      bash egs/launch.sh inf $rc; done
#
# merge: fold a lora/qlora run's model/ into its base -> <run folder>/model_merged/ (full bf16
# weights, tokenizer, merge_info.yaml), usable as `checkpoint:` in another experiment (a second
# SFT round) or decoded zero-shot with `launch.sh inf <experiment>.yaml`
#   bash egs/launch.sh merge <run folder>/run_config.yaml [slurm.* overrides]
#
# Machine-specific settings (HFMT_DATA_ROOT) come from the yaml folder's untracked local_setting.sh
# (copy local_setting.sh.example). Saved configs hold resolved paths, so jobs don't need it.
set -eo pipefail

# 1. Arguments: the kind, then the yaml (relative to the current dir or the repo root, or absolute).
usage="usage: bash egs/launch.sh <sft|inf|merge> <yaml> [overrides...]"
kind=$1
case "$kind" in sft|inf|merge) ;; *) echo "$usage" >&2; exit 1 ;; esac
shift
export HFMT_ROOT=${HFMT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
if [ -f "$1" ]; then yaml=$(realpath "$1")
elif [ -n "$1" ] && [ -f "${HFMT_ROOT}/$1" ]; then yaml=$(realpath "${HFMT_ROOT}/$1")
else
    echo "$usage  (not found: '$1')" >&2
    exit 1
fi
shift

# 2. Machine-specific settings from the yaml's folder. An experiment folder has local_setting.sh;
#    a run folder doesn't and needs none. Unset first so a value left in your shell can't point
#    this experiment at another folder's data.
unset HFMT_DATA_ROOT
if [ -f "$(dirname "$yaml")/local_setting.sh" ]; then source "$(dirname "$yaml")/local_setting.sh"; fi

# 3. Environment (conda env, PYTHONPATH); run from the repo root.
source "${HFMT_ROOT}/install/path.sh"
export PYTHONPATH="${HFMT_ROOT}:${PYTHONPATH}"
cd "${HFMT_ROOT}"

# 4. grid.py checks everything and writes the job folders, then prints bash that sets
#    JOBS=(job folders) and SBATCH_ARGS=(sbatch options); see its docstring. Any problem exits here,
#    before anything is submitted.
if [ "$kind" = sft ]; then grid_args=("$yaml"); else grid_args=("--$kind" "$yaml"); fi
grid=$(python -m hfmt.grid "${grid_args[@]}" "$@")
eval "$grid"

# 5. Run the jobs: egs/job.sh <kind> <all job folders>; job i (array task i) works on JOBS[i].
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
