#!/bin/bash
set -e

HF_COMMIT=19dabe96362803fb0a9ae7073d03533966598b17 # (huggingface transformers version 11/30/2024)

# Get this version of HF Transformers
rootdir="$(readlink -f "$(dirname "$0")/../")"
cd $rootdir
git submodule update --init --recursive
cd transformers
git checkout $HF_COMMIT
cd ..

$rootdir/install/install_hf_custom.sh -s $rootdir/transformers -e hfmt
