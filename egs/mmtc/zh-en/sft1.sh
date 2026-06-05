#!/bin/sh

# sbatch -p gpu --gres=gpu:h200:1 --time=48:00:00 egs/mmtc/zh-en/sft1.sh

source ${HFMT_ROOT}/install/path.sh

src=zh
trg=en
train_yaml=${HFMT_ROOT}/egs/mmtc/zh-en/train+valid.yaml
evalset=/exp/kduh/data/mt/mmtc/test.${src}-${trg}.${src}

checkpoint="Qwen/Qwen2.5-1.5B-Instruct"
instruction="Translate Chinese to English"

###########################
# hyperparameters:
# lr_scheduler_type: linear, reduce_lr_on_plateau
# learning_rate: 2e-5, 2e-4
# weight_decay: 0.01
# batch_size: 16, 32, 64
# seed 42, 37
pretrain=1
outdir=egs/mmtc/zh-en/tv.qwen-1.5.1
cmdarg="--max_steps 50000 --logging_steps 500 --eval_steps 500 --warmup_steps 0 \
        --lr_scheduler_type reduce_lr_on_plateau --learning_rate 2e-4 --weight_decay 0.01 \
        --label_smoothing_factor 0.0 --seed 37 --batch_size 16"
###########################

mkdir -p $outdir
if [[ $pretrain -eq 1 ]]; then
    cmdarg="$cmdarg -p"
fi

python ${HFMT_ROOT}/hfmt/sft_translation.py -t $train_yaml -e $evalset -c $checkpoint -o $outdir $cmdarg -i "$instruction"
