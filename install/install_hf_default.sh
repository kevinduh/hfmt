#!/bin/bash
set -e

HF_COMMIT=049d2bf1220747b6d39e2a978b9f5fe0defa1dca # (Release v5.8.0, 05/05/2026)
#HF_COMMIT=2b8a15cc3f1a0c94cf817a8fd8c87bca28737e09 # (Release v4.57? 03/24/2025)


# Get this version of HF Transformers
rootdir="$(readlink -f "$(dirname "$0")/../")"
cd $rootdir
git submodule update --init --recursive
cd transformers
git checkout $HF_COMMIT
cd ..

$rootdir/install/install_hf_custom.sh -s $rootdir/transformers -e hfmt
