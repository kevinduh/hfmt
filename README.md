# HFMT - Machine Translation scripts based on HuggingFace Transformers

## Installation

```bash
git clone https://github.com/kevinduh/hfmt.git
cd hfmt/
/bin/bash install/install_hf_default.sh
```

This will install a conda environment (default is named `hfmt`) 
with Huggingface Transformers and other necessary packages included. 

## Code Structure

Example Bash and Qsub scripts are in the subfolder `egs/`. 
* These scripts will first activate the conda environment by calling `install/path.sh`. Please modify this if needed.
* Then they call the relevant Python code in the subfolder `hfmt/`. 

Before running the scripts, please set the `HFMT_ROOT` variable, e.g. `export HFMT_ROOT=path/to/this/repo/`. This is needed for the scripts to find the path to everything.  

Currently implemented: 
* `hfmt/train_seq2seq.py`: Trains a Seq2Seq model (either by fine-tuning a pretrained model or training from scratch)
* todo: training decoder only models ...

Scripts that perform training integrate with (Weights & Biases)[https://wandb.ai/] for logging purposes, so it is recommended that you set up a free account on that service. 

## Usage example: training seq2seq MT model

As illustration, let's fine-tune model on a small dataset. 

First, we will create a yaml file that lists the training and dev data. Following the example template in `egs/data/example.template.yaml`, let's create a new file `egs/data/example.yaml` based on your own paths:

```bash
cd path/to/this/repo
export HFMT_ROOT=`pwd`
sed "s|__HFMT_ROOT__|$HFMT_ROOT|g" egs/data/example.template.yaml > egs/data/example.yaml
cat egs/data/example.yaml
```

The yaml file should point to sentence-aligned files like these:

```bash
head -2 egs/data/example.de-en.train.*

==> egs/data/example.de-en.train.de <==
(applaus) david gallo: das ist bill lange. ich bin dave gallo.
wir werden ihnen einige geschichten über das meer in videoform erzählen.

==> egs/data/example.de-en.train.en <==
-lrb- applause -rrb- david gallo : this is bill lange . i 'm dave gallo .
and we 're going to tell you some stories from the sea here in video .
```

Next, we run the training script:

```bash
# make sure to check that you've set HFMT_ROOT: export HFMT_ROOT=path/to/this/repo/
qsub -S /bin/bash -V -cwd -j y -q gpu.q@@v100 -l gpu=1,h_rt=24:00:00,num_proc=8,mem_free=25G egs/translation/train_seq2seq_t5.sh
```

This will probably take around 15 minutes. For a real run, we will likely want to increase the `--max_steps` hyperparameter in the script. Here are some of the important settings in `egs/translation/train_seq2seq_t5.sh`, which can be modified for your own runs:

```bash

train_yaml=${HFMT_ROOT}/egs/data/example.yaml # specifies training/dev bitext
evalset=${HFMT_ROOT}/egs/data/example.${src}-${trg}.test1.${src} # specifies (optional) eval set to decode after training

checkpoint="google-t5/t5-small" # Huggingface checkpoint to load 
instruction="translate German to English:" # Instruction string provided as prefix. May be empty depending on the model
pretrain=1 # Set to 1 to fine-tune a pretrained model. Set to 0 to train from scratch
cmdarg="--max_steps 5000 ..." # various additional hyperparameters

```

See `egs/translation/train_seq2seq_marian.sh` for a different example.
