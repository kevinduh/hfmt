#!/bin/sh

#qlogin -q gpu.q@@v100 -l num_proc=10,mem_free=32G,h_rt=72:00:00,gpu=1
#qsub -S /bin/bash -V -cwd -j y -q gpu.q@@v100 -l gpu=1,h_rt=24:00:00,num_proc=8,mem_free=25G scripts/cascade.sh

###########################
# Environment #############
###########################
ENV_NAME=hfmt
rootdir=/exp/nrobinson/xling_summarizn/hfmt
source $rootdir/install/path.sh
conda activate $ENV_NAME
outdir=egs/models/toy_cascade_outs
mkdir -p $outdir

###########################
# Machine Translate #######
###########################
evalset=$rootdir/egs/data/CrossSum-test/spanish-english.toy.jsonl
instruction=""
pretrain=1
outfile=$outdir/mt_outs.jsonl
checkpoint="$rootdir/egs/models/marian.scratch.1/checkpoint-100000"
language="spanish"
cmdarg=""
if [[ $pretrain -eq 1 ]]; then
    cmdarg="-p"
fi
python $rootdir/hfmt/cascade_seq2seq.py -e $evalset -c $checkpoint -o $outfile -i "$instruction" -l $language $cmdarg
echo "Check $outfile"

###########################
# Summarize ###############
###########################
evalset=egs/models/cascade_outs/mt_outs.jsonl
instruction="Summarize the following passage. Do not provide any explanations or text apart from the summary.\nPassage: "
pretrain=1
outfile=$outdir/final_outs.jsonl
checkpoint="meta-llama/Meta-Llama-3-8B-Instruct"
cmdarg="-s"
if [[ $pretrain -eq 1 ]]; then
    cmdarg="$cmdarg -p"
fi
python $rootdir/hfmt/cascade_seq2seq.py -e $evalset -c $checkpoint -o $outfile -i "$instruction" $cmdarg
echo "Check $outfile"

