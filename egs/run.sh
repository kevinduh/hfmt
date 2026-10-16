#!/bin/sh

# Generic Hydra launcher for HFMT experiments (feature 00_hydra; egs/ layout + sweep
# toggle in feature 04; per-dataset/lang directory hierarchy in feature 06). An experiment
# recipe lives at egs/<dataset>/<src>-<trg>/<name>.yaml next to its dataset.yaml template;
# this script activates the env, resolves the inline sweep toggle, and submits to Slurm via
# the submitit launcher.
#
# Run from a LOGIN node; submitit submits the Slurm job for you (do NOT sbatch this):
#   export HFMT_ROOT=$(pwd)
#   bash egs/run.sh mmtc/fr-en/sft1.yaml
#
# The sole positional arg is the experiment's path relative to egs/ (the .yaml file; a
# trailing .yaml is optional). Its directory selects the nested Hydra config group and names
# the dataset.yaml location; its basename is the experiment option (feature 06).
#
# Override any knob on the CLI (forwarded via "$@"), e.g.:
#   bash egs/run.sh mmtc/fr-en/sft1.yaml lora.lora_r=16 train.learning_rate=2e-5
# Turn the experiment's inline sweep on/off, or sweep ad hoc:
#   bash egs/run.sh mmtc/fr-en/sft1.yaml sweep.enabled=true
#   bash egs/run.sh mmtc/fr-en/sft1.yaml lora.lora_r=8,16,32
#
# Run locally/in-process instead of submitting (e.g. already on a GPU node) -- pass the same
# nested selector + the dataset dir yourself:
#   cd ${HFMT_ROOT} && python -m hfmt.sft_translation \
#       +mmtc/fr-en=sft1 data.dataset_dir=${HFMT_ROOT}/egs/mmtc/fr-en
#
# NOTE: we run as a module (python -m hfmt.sft_translation) with HFMT_ROOT on
# PYTHONPATH so that submitit can re-import hfmt.* when it unpickles the job on the
# compute node.

set -e

if [ -z "$1" ]; then
    echo "usage: bash egs/run.sh <experiment.yaml> [hydra overrides...]" >&2
    echo "       <experiment.yaml> is a path under egs/, e.g. mmtc/fr-en/sft1.yaml" >&2
    exit 1
fi

exp_rel="$1"
shift

source ${HFMT_ROOT}/install/path.sh
export PYTHONPATH="${HFMT_ROOT}:${PYTHONPATH}"
cd "${HFMT_ROOT}"

# Resolve the experiment path (relative to egs/). Strip a trailing .yaml for the Hydra option
# name, then split into the nested config group (its directory) and the option (its basename).
# feature 06.
exp_rel_noext=$(printf '%s' "${exp_rel}" | sed 's/\.yaml$//')
exp_group=$(dirname "${exp_rel_noext}")
exp_name=$(basename "${exp_rel_noext}")

if [ "${exp_group}" = "." ] || [ -z "${exp_group}" ]; then
    echo "egs/run.sh: experiment must live under egs/<dataset>/<src>-<trg>/ (got '${exp_rel}')" >&2
    exit 1
fi

exp_file="${HFMT_ROOT}/egs/${exp_rel_noext}.yaml"
if [ ! -f "${exp_file}" ]; then
    echo "egs/run.sh: experiment not found: ${exp_file}" >&2
    echo "       pass a path under egs/, e.g. mmtc/fr-en/sft1.yaml" >&2
    exit 1
fi

# The dataset.yaml template lives in the experiment's own directory; inject that directory and
# the per-experiment output root (feature 06). The experiment names the file itself via
# data.dataset_file (default dataset.yaml); run.sh only supplies the directory.
dataset_dir="${HFMT_ROOT}/egs/${exp_group}"
output_root="egs/${exp_group}/outputs"

# Sweep toggle (feature 04): the experiment carries an inline `sweep: {enabled, params}` block.
# sweep_probe.py composes it and, when enabled, prints bare `<axis>=<csv>` overrides that the
# basic sweeper fans out into the job grid (disabled -> nothing -> one job). We always launch
# --multirun (the submitit Slurm launcher only runs in multirun), so the grid is what makes a
# sweep, not the flag.
#
# Forward only the `sweep.enabled` toggle to the probe so it can be flipped from the CLI
# (`sweep.enabled=true`). Everything else stays in "$@" and goes to the launch: ad-hoc
# comma-list axes (e.g. lora.lora_r=8,16, or basemodel=qwen2.5-7b,qwen2.5-1.5b to sweep whole
# component files) are expanded by the sweeper directly. We do NOT forward sweep.params.* here
# -- a comma-list value breaks the probe's single-run, type-validated compose; author extra
# axes in the experiment YAML or pass them as bare overrides instead.
probe_overrides=""
for arg in "$@"; do
    case "$arg" in
        sweep.enabled=*) probe_overrides="${probe_overrides} ${arg}" ;;
    esac
done

# The probe composes the same experiment (nested selector + injected dataset_dir) and also
# fast-fails here on the login node if the dataset.yaml template is missing, before the submit.
sweep_axes=$(python -m hfmt.sweep_probe "${exp_group}=${exp_name}" "data.dataset_dir=${dataset_dir}" ${probe_overrides})

# The launcher is a selectable component now (feature 04): each experiment SELECTS one in its
# defaults list (`- override /hydra/launcher: slurm` or a preset like slurm_fullft), so we no
# longer force it here. Override on the CLI if needed, e.g. hydra/launcher=local. (An experiment
# that selects none falls back to Hydra's local `basic` launcher.)
python -m hfmt.sft_translation --multirun \
    "+${exp_group}=${exp_name}" \
    data.dataset_dir="${dataset_dir}" \
    output_root="${output_root}" \
    ${sweep_axes} \
    "$@"
