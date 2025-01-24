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
evalset_1=$rootdir/egs/data/CrossSum-test/spanish-english.toy.jsonl
instruction_1=""
pretrain_1=1
outfile_1=$outdir/mt_outs.jsonl
checkpoint_1="$rootdir/egs/models/marian.scratch.1/checkpoint-100000"
language_1="spanish"
cmdarg_1=""
if [[ $pretrain_1 -eq 1 ]]; then
    cmdarg_1="-p"
fi
python $rootdir/hfmt/cascade_seq2seq.py -e $evalset_1 -c $checkpoint_1 -o $outfile_1 -i "$instruction_1" -l $language_1 $cmdarg_1
echo "Check $outfile_1"

###########################
# Summarize ###############
###########################
evalset_2=$outfile_1
instruction_2="Summarize the following passage. Do not provide any explanations or text apart from the summary.\nPassage: "
pretrain_2=1
outfile_2=$outdir/final_outs.jsonl
checkpoint_2="meta-llama/Meta-Llama-3-8B-Instruct"
cmdarg_2="-s"
if [[ $pretrain_2 -eq 1 ]]; then
    cmdarg_2="$cmdarg_2 -p"
fi
python $rootdir/hfmt/cascade_seq2seq.py -e $evalset_2 -c $checkpoint_2 -o $outfile_2 -i "$instruction_2" $cmdarg_2
echo "Check $outfile_2"

###########################
# Score ###################
###########################
outfile_3=$outdir/scores.jsonl

python $rootdir/hfmt/scoring.py -r $evalset_1 -t $outfile_2 -o $outfile_3

echo "Finished cascade and testing"
