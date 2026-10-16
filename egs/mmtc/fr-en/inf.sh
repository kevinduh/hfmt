#!/bin/sh

# sbatch -p gpu --gres=gpu:a100:1 --time=48:00:00 egs/mmtc/fr-en/inf.sh

source ${HFMT_ROOT}/install/path.sh

src=fr
trg=en
evalset=${HFMT_ROOT}/egs/mmtc/fr-en/test.${src}-${trg}.${src}

checkpoint="Qwen/Qwen2.5-1.5B-Instruct"
peft_ckpt="egs/mmtc/fr-en/qwen-qlora.1_b"
instruction="Translate French to English"

###########################
peft=1
outdir=$peft_ckpt
cmdarg=""
###########################

mkdir -p $outdir
if [[ $peft -eq 1 ]]; then
    cmdarg="$cmdarg -p $peft_ckpt"
fi

python ${HFMT_ROOT}/hfmt/inf_translation.py -e $evalset -c $checkpoint -o $outdir $cmdarg -i "$instruction"