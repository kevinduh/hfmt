#!/bin/bash
#SBATCH --job-name=p16g_packing_leak
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0:30:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=1
#SBATCH --output=features/16_cpt/test_packing_leak.%j.out
# plan.md R1 quick test (16_cpt): packed vs one-at-a-time loss per example. From the repo root:
#   sbatch features/16_cpt/test_packing_leak.sh [--bf16] [model ...]
export HFMT_ROOT=${HFMT_ROOT:-$(pwd)}
source "${HFMT_ROOT}/install/path.sh"
export PYTHONPATH="${HFMT_ROOT}:${PYTHONPATH}"
cd "${HFMT_ROOT}"

# Wait until the GPU is usable, as egs/job.sh does (exclusive-process GPUs race at start-up).
for try in 1 2 3 4 5 6; do
    err=$(python -c "import torch; torch.zeros(1, device='cuda')" 2>&1) && break
    if [ "$try" = 6 ]; then echo "$err" >&2; echo "GPU still not usable after $try tries" >&2; exit 1; fi
    echo "GPU not usable (try $try); retrying in 30 s" >&2
    sleep 30
done

python features/16_cpt/test_packing_leak.py "$@"
