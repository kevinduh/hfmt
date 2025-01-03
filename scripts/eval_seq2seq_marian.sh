#!/bin/sh

#qlogin -q gpu.q@@v100 -l num_proc=10,mem_free=32G,h_rt=72:00:00,gpu=1
#qsub -S /bin/bash -V -cwd -j y -q gpu.q@@2080 -l gpu=1,h_rt=24:00:00,num_proc=8,mem_free=25G scripts/eval_seq2seq_marian.sh

ENV_NAME=hfmt
rootdir=/exp/nrobinson/xling_summarizn/hfmt

source $rootdir/install/path.sh
conda activate $ENV_NAME


evalset=$rootdir/egs/data/example.de-en.test1.de
instruction=""

###########################
# hyperparameters:
# lr_scheduler_type: linear, reduce_lr_on_plateau
# learning_rate: 2e-5, 2e-4
# weight_decay: 0.01
# batch_size: 16, 32, 64
# seed 42, 37
pretrain=1
outdir=egs/models/marian.scratch.1
checkpoint="$rootdir/egs/models/marian.scratch.1/checkpoint-100000"
cmdarg=""
###########################

if [[ $pretrain -eq 1 ]]; then
    cmdarg="-p"
fi

python $rootdir/hfmt/eval_seq2seq.py -e $evalset -c $checkpoint -o $outdir -i "$instruction" $cmdarg


