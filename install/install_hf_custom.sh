#!/bin/bash
# Install a custom version of Huggingface Transformers in a conda environment

set -e

CONDA_HOME=$HOME/.conda/envs

function errcho() {
  >&2 echo $1
}

function show_help() {
  errcho "Install environment for custom transformers"
  errcho "usage: install_hf_custom.sh [-h] -s TRANSFORMERS_LOCATION -e ENV_NAME [-f]"
  errcho ""
}

function check_dir_exists() {
  if [ ! -d $1 ]; then
    errcho "FATAL: Could not find directory $1"
    exit 1
  fi
}

FORCE_NEW_ENV=false

while getopts ":h?s:e:f" opt; do
  case "$opt" in
    h|\?)
      show_help
      exit 0
      ;;
    s) TRANSFORMERS=$OPTARG
      ;;
    e) ENV_NAME=$OPTARG
      ;;
    f) FORCE_NEW_ENV=true
      ;;
  esac
done

if [[ -z $TRANSFORMERS || -z $ENV_NAME ]]; then
  errcho "Missing arguments"
  show_help
  exit 1
fi

if [[ "$FORCE_NEW_ENV" == true ]]; then
  BASE_ENV_NAME=$ENV_NAME
  suffix=0
  while [ -d $CONDA_HOME/$ENV_NAME ]; do
    ENV_NAME=${BASE_ENV_NAME}_${suffix}
    suffix=$((suffix+1))
  done

  if [ $ENV_NAME != $BASE_ENV_NAME ]; then
    errcho "$BASE_ENV_NAME was already taken; we will use $ENV_NAME"
  fi
fi

check_dir_exists $TRANSFORMERS
pwd

# 1. setup python virtual environment 
venv=$ENV_NAME # set your virtual enviroment name
if [[ "$FORCE_NEW_ENV" == true || ! -d $CONDA_HOME/$ENV_NAME ]]; then
  errcho "Creating new Conda env : $ENV_NAME"
  conda create -y -n $venv python=3.9
fi

source activate $venv
export PYTHONNOUSERSITE=1

# 2. install Huggingface Transformers
cd $TRANSFORMERS
pip install -e .
pip install torch datasets evaluate torchaudio sacrebleu ipywidgets accelerate sentencepiece wandb
conda install -c conda-forge ipykernel

# 3. install other tools
pip install sacremoses nltk rouge_score
