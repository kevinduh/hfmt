#!/usr/bin/env bash
#
# run_on_free_gpu.sh -- launch egs/run.sh on whichever GPU type (a100 or l40s) has the
# MOST free GPUs right now, so you don't have to eyeball `tools/gpu_free.sh` and set
# hydra.launcher.gres by hand.
#
# Usage (run from a login node):
#   bash tools/run_on_free_gpu.sh mmtc_fr-en_sft1 +sweep=mmtc_fr-en_coarse
#   bash tools/run_on_free_gpu.sh mmtc_fr-en_sft1 train.seed=37,42 train.max_steps=20
#
# All args are forwarded to egs/run.sh verbatim; this wrapper only appends the chosen
# `hydra.launcher.gres=gpu:<type>:1`. So do NOT pass your own hydra.launcher.gres (that
# would be a duplicate override). Change the candidate types with GPU_TYPES:
#   GPU_TYPES="a100 l40s v100" bash tools/run_on_free_gpu.sh mmtc_fr-en_sft1 +sweep=...
#
# Policy: pick the type with the most free GPUs (tie -> first in GPU_TYPES). Note your
# 4gpu_tier caps you at 4 concurrent GPUs regardless, so "freest" mainly shortens queueing.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/.." && pwd)"
export HFMT_ROOT="${HFMT_ROOT:-$root}"

read -r -a cand <<<"${GPU_TYPES:-a100 l40s}"

# Ask the reporter (single source of truth for parsing) for free counts, pick the max.
best=""; bestf=-1
while read -r t f; do
  [[ -z "${t:-}" ]] && continue
  if (( f > bestf )); then bestf=$f; best=$t; fi
done < <(bash "$here/gpu_free.sh" --emit "${cand[@]}")

if [[ -z "$best" ]]; then
  echo "run_on_free_gpu: couldn't read free-GPU counts (is 'scontrol' on PATH?)." >&2
  exit 1
fi
if (( bestf <= 0 )); then
  echo "run_on_free_gpu: nothing free among [${cand[*]}] right now." >&2
  echo "  See the full picture with:  bash tools/gpu_free.sh" >&2
  exit 1
fi

echo ">> Freest GPU type: ${best} (${bestf} free). Launching on gpu:${best}:1" >&2
exec bash "$root/egs/run.sh" "$@" "hydra.launcher.gres=gpu:${best}:1"
