#!/bin/sh

# sbatch -p gpu --gres=gpu:h200:1 --time=48:00:00 egs/wmt25/ja-zh/sft1.sh

source ${HFMT_ROOT}/install/path.sh

src=ewe
trg=eng
evalset=/exp/kduh/p/integration/dso/data/bitext/flores200.dev.$src

checkpoint="CohereLabs/tiny-aya-base"
peft_ckpt=""
instruction="Translate Ewe to English:"

###########################
# hyperparameters:
# lr_scheduler_type: linear, reduce_lr_on_plateau
# learning_rate: 2e-5, 2e-4
# weight_decay: 0.01
# batch_size: 16, 32, 64
# seed 42, 37
peft=0
outdir=egs/synth_lrl/${src}-${trg}/models/aya2
cmdarg="--max_steps 50000 --logging_steps 500 --eval_steps 500 --warmup_steps 0 \
        --lr_scheduler_type reduce_lr_on_plateau --learning_rate 2e-4 --weight_decay 0.01 \
        --label_smoothing_factor 0.0 --seed 37 --batch_size 16"
###########################

mkdir -p $outdir
if [[ $peft -eq 1 ]]; then
    cmdarg="$cmdarg -p $peft_ckpt"
fi

python ${HFMT_ROOT}/hfmt/inf_translation.py -e $evalset -c $checkpoint -o $outdir $cmdarg -i "$instruction"
