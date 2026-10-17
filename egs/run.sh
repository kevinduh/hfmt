#!/bin/sh

# Generic Hydra launcher for HFMT experiments (feature 00_hydra; egs/ layout in feature 04;
# per-dataset/lang directory hierarchy in feature 06; two kinds of experiment in feature 07).
# An experiment recipe lives at egs/<dataset>/<src>-<trg>/<name>.yaml next to its dataset.yaml
# template; this script activates the env and submits to Slurm via the submitit launcher.
#
# Run from a LOGIN node; submitit submits the Slurm job for you (do NOT sbatch this):
#   export HFMT_ROOT=$(pwd)
#   bash egs/run.sh mmtc/fr-en/sft_lora.yaml          # single-run experiment -> 1 job
#   bash egs/run.sh mmtc/fr-en/sft_lora.sweep.yaml    # sweep experiment -> its full grid
#
# The sole positional arg is the experiment's path relative to egs/ (the .yaml file; a
# trailing .yaml is optional). Its directory selects the nested Hydra config group and names
# the dataset.yaml location; its basename is the experiment option (feature 06).
#
# Two kinds of experiment (feature 07): a single-run file has no grid; a sweep file carries
# its grid in Hydra's native `hydra.sweeper.params` and always runs all of it. The submitit
# launcher logs every job's overrides (`#<idx> : ...`) just before submitting.
#
# Override any knob on the CLI (forwarded via "$@"), e.g.:
#   bash egs/run.sh mmtc/fr-en/sft_lora.yaml lora.lora_r=16 train.learning_rate=2e-5
# Sweep ad hoc on any experiment with a comma list (adds an axis to any grid in the file):
#   bash egs/run.sh mmtc/fr-en/sft_lora.yaml lora.lora_r=8,16,32
#
# Run locally/in-process instead of submitting (e.g. already on a GPU node) -- pass the same
# nested selector + the dataset dir yourself:
#   cd ${HFMT_ROOT} && python -m hfmt.sft_translation \
#       +mmtc/fr-en=sft_lora data.dataset_dir=${HFMT_ROOT}/egs/mmtc/fr-en
#
# NOTE: we run as a module (python -m hfmt.sft_translation) with HFMT_ROOT on
# PYTHONPATH so that submitit can re-import hfmt.* when it unpickles the job on the
# compute node.

set -e

if [ -z "$1" ]; then
    echo "usage: bash egs/run.sh <experiment.yaml> [hydra overrides...]" >&2
    echo "       <experiment.yaml> is a path under egs/, e.g. mmtc/fr-en/sft_lora.yaml" >&2
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
    echo "       pass a path under egs/, e.g. mmtc/fr-en/sft_lora.yaml" >&2
    exit 1
fi

# The dataset.yaml template lives in the experiment's own directory; inject that directory and
# the per-experiment output root (feature 06). The experiment names the file itself via
# data.dataset_file (default dataset.yaml); run.sh only supplies the directory.
dataset_dir="${HFMT_ROOT}/egs/${exp_group}"
output_root="egs/${exp_group}/outputs"

# Fast-fail on the login node, before the Slurm submit, if the dataset template is missing.
# Only the default name is checked; main() validates data.dataset_file authoritatively.
if [ ! -f "${dataset_dir}/dataset.yaml" ]; then
    echo "egs/run.sh: dataset template not found: ${dataset_dir}/dataset.yaml" >&2
    exit 1
fi

# Always --multirun: the submitit Slurm launcher only runs in multirun. A single-run experiment
# yields one job; a sweep experiment's hydra.sweeper.params (plus any CLI comma lists) yields
# the grid. Each experiment SELECTS its launcher in its defaults list
# (`- override /hydra/launcher: slurm`); override on the CLI if needed, e.g.
# hydra/launcher=local. (An experiment that selects none falls back to Hydra's local `basic`
# launcher.)
python -m hfmt.sft_translation --multirun \
    "+${exp_group}=${exp_name}" \
    data.dataset_dir="${dataset_dir}" \
    output_root="${output_root}" \
    "$@"
