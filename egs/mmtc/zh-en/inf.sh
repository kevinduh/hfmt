#!/bin/sh

# sbatch -p gpu --gres=gpu:h200:1 --time=48:00:00 egs/wmt25/ja-zh/sft1.sh

source ${HFMT_ROOT}/install/path.sh

src=zh
trg=en
evalset=/exp/kduh/data/mt/mmtc/test.${src}-${trg}.${src}

checkpoint="CohereLabs/tiny-aya-global"
peft_ckpt="egs/mmtc/zh-en/aya-qlora/checkpoint-5000"
instruction="Translate Chinese to English"

###########################
# hyperparameters:
# lr_scheduler_type: linear, reduce_lr_on_plateau
# learning_rate: 2e-5, 2e-4
# weight_decay: 0.01
# batch_size: 16, 32, 64
# seed 42, 37
peft=1
outdir=$peft_ckpt
cmdarg="--max_steps 50000 --logging_steps 500 --eval_steps 500 --warmup_steps 0 \
        --lr_scheduler_type reduce_lr_on_plateau --learning_rate 2e-4 --weight_decay 0.01 \
        --label_smoothing_factor 0.0 --seed 37 --batch_size 16"
###########################

mkdir -p $outdir
if [[ $peft -eq 1 ]]; then
    cmdarg="$cmdarg -p $peft_ckpt"
fi

python ${HFMT_ROOT}/hfmt/inf_translation.py -e $evalset -c $checkpoint -o $outdir $cmdarg -i "$instruction"
