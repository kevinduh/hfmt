# HFMT - Machine Translation scripts based on HuggingFace Transformers

## Installation

```bash
git clone https://github.com/kevinduh/hfmt.git
/bin/bash install/install_hf_default.sh
```

This will install a conda environment (default is named `hfmt`) 
with Huggingface Transformers and other necessary packages included. 

## Usage

Bash and Qsub scripts are in the subfolder `scripts/`. They call the Python code in the subfolder `hfmt/`. 

Currently implemented: 
* `hfmt/train_seq2seq.py`: Trains a Seq2Seq model (either by fine-tuning a pretrained model or training from scratch)
* todo: training decoder only models ...

The training and dev data is tab-separated formated, where source sentence is the first column and target sentence is the second column. See:

```bash
head -2 egs/data/example.de-en.train.bitext
(applaus) david gallo: das ist bill lange. ich bin dave gallo.	-lrb- applause -rrb- david gallo : this is bill lange . i 'm dave gallo .
wir werden ihnen einige geschichten über das meer in videoform erzählen.       and we 're going to tell you some stories from the sea here in video .
```

For example, the following trains a Marian (OPUS) model. Modify the `$rootdir` in the script to the directory where you cloned this repo.
```bash
qsub -S /bin/bash -V -cwd -j y -q gpu.q@@v100 -l gpu=1,h_rt=24:00:00,num_proc=8,mem_free=25G scripts/train_seq2seq_marian.sh
```

