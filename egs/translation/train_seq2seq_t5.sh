#!/bin/sh

#qlogin -q gpu.q@@v100 -l num_proc=10,mem_free=32G,h_rt=72:00:00,gpu=1
#qsub -S /bin/bash -V -cwd -j y -q gpu.q@@v100 -l gpu=1,h_rt=24:00:00,num_proc=8,mem_free=25G scripts/train_seq2seq_t5.sh

source ${HFMT_ROOT}/install/path.sh

src=de
trg=en
train_yaml=${HFMT_ROOT}/egs/data/example.yaml
evalset=${HFMT_ROOT}/egs/data/example.${src}-${trg}.test1.${src}

checkpoint="google-t5/t5-small"
instruction="translate German to English:"

###########################
# hyperparameters:
# lr_scheduler_type: linear, reduce_lr_on_plateau
# learning_rate: 2e-5, 2e-4
# weight_decay: 0.01
# batch_size: 16, 32, 64
# seed 42, 37
pretrain=1
outdir=egs/models/t5small.pretrain.1
cmdarg="--max_steps 5000 --logging_steps 100 --eval_steps 1000 --warmup_steps 0 \
        --lr_scheduler_type reduce_lr_on_plateau --learning_rate 2e-4 --weight_decay 0.01 \
        --label_smoothing_factor 0.0 --seed 37 --batch_size 32"
###########################

mkdir -p $outdir
if [[ $pretrain -eq 1 ]]; then
    cmdarg="$cmdarg -p"
fi

python ${HFMT_ROOT}/hfmt/train_seq2seq.py -t $train_yaml -e $evalset -c $checkpoint -o $outdir $cmdarg -i "$instruction"
