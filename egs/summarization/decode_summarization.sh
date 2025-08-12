#!/bin/sh

#qsub -S /bin/bash -V -cwd -j y -q gpu.q@@a100 -l gpu=1,h_rt=24:00:00,num_proc=8,mem_free=25G egs/summarization/decode_summarization.sh

source ${HFMT_ROOT}/install/path.sh

evalset=${HFMT_ROOT}/egs/data/summarization.en-en.jsonl
#checkpoint=meta-llama/Llama-3.2-3B-Instruct
checkpoint=CohereLabs/aya-23-8B
#outprefix=egs/summarization/example.Llama-3.2-3B-Instruct
outprefix=egs/summarization/example.aya-23-8B
batch_size=1
prompt_choice=1

python ${HFMT_ROOT}/hfmt/decode_summarization.py -e $evalset -c $checkpoint -o $outprefix \
                                                --batch_size $batch_size \
                                                --prompt_choice $prompt_choice 
