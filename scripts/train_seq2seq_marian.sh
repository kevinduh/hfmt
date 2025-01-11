#!/bin/sh

#qlogin -q gpu.q@@v100 -l num_proc=10,mem_free=32G,h_rt=72:00:00,gpu=1
#qsub -S /bin/bash -V -cwd -j y -q gpu.q@@v100 -l gpu=1,h_rt=24:00:00,num_proc=8,mem_free=25G scripts/train_seq2seq_marian.sh

ENV_NAME=hfmt
rootdir=/exp/nrobinson/xling_summarizn/hfmt

source $rootdir/install/path.sh
conda activate $ENV_NAME


trainset=$rootdir/egs/data/example.de-en.train.bitext
devset=$rootdir/egs/data/example.de-en.dev.bitext
evalset=$rootdir/egs/data/example.de-en.test1.de
checkpoint="Helsinki-NLP/opus-mt-de-en"
instruction=""

###########################
# hyperparameters:
# lr_scheduler_type: linear, reduce_lr_on_plateau
# learning_rate: 2e-5, 2e-4
# weight_decay: 0.01
# batch_size: 16, 32, 64
# seed 42, 37
pretrain=0
outdir=$rootdir/egs/models/marian.scratch.1
cmdarg="--max_steps 100000 --logging_steps 10 --eval_steps 1000 --warmup_steps 0 \
        --lr_scheduler_type reduce_lr_on_plateau --learning_rate 2e-5 --weight_decay 0.01 \
        --label_smoothing_factor 0.0 --seed 37 --batch_size 16"
###########################

mkdir -p $outdir
if [[ $pretrain -eq 1 ]]; then
    cmdarg="$cmdarg -p"
fi

python $rootdir/hfmt/train_seq2seq.py -t $trainset -d $devset -c $checkpoint -o $outdir $cmdarg -i "$instruction"


