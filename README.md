# HFMT - Machine Translation and Cross-lingual NLP scripts based on HuggingFace Transformers

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
* `hfmt/decode_summarization.py`: Runs inference on CausalLM models, with prompts for summarization
* `hfmt/sft_translation.py`: Supervised fine-tuning (QLoRA, LoRA or full) of decoder-only LLMs for MT, configured with Hydra (see below)
* `hfmt/inf_translation.py`: Inference with a fine-tuned run or a downloaded checkpoint; `hfmt/merge_model.py`: merges a LoRA/QLoRA run into a full checkpoint (see below)
* todo: cascading of models, ...

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

This will probably take around 15 minutes. For a real run, we will likely want to increase the `--max_steps` hyperparameter in the script. Here are some of the important settings in `egs/translation/train_seq2seq_t5.sh`, which should be modified for your own runs:

```bash

train_yaml=${HFMT_ROOT}/egs/data/example.yaml # specifies training/dev bitext
evalset=${HFMT_ROOT}/egs/data/example.${src}-${trg}.test1.${src} # specifies (optional) eval set to decode after training

checkpoint="google-t5/t5-small" # Huggingface checkpoint to load 
instruction="translate German to English:" # Instruction string provided as prefix. May be empty depending on the model
pretrain=1 # Set to 1 to fine-tune a pretrained model. Set to 0 to train from scratch
cmdarg="--max_steps 5000 ..." # various additional hyperparameters

```

See `egs/translation/train_seq2seq_marian.sh` for a different example.

## Running SFT experiments (decoder-only MT, Hydra)

`hfmt/sft_translation.py` (training) and `hfmt/inf_translation.py` (inference) are configured
with [Hydra](https://hydra.cc). The schema in `hfmt/hydra_config.py` defines every key and its
type but holds no default values; each default lives in exactly one yaml:

* `conf/config.yaml`: the defaults list and project-wide settings (`slurm`, `wandb`, `inf_name`).
* `conf/<group>/<option>.yaml`: swappable bundles for `train` (`default`), `peft`
  (`qlora_default`, `lora_default`, `full_finetune`) and `decode` (`default`).
* the experiment folder, e.g. `egs/mmtc/fr-en/`:
  * `dataset.yaml`: `data_root`, `src`, `trg` and the train/dev/test file lists. Several train or
    dev files are concatenated in order; several test files are each decoded and scored on their
    own (a `test.trg` entry may be `null` for a test set without reference, or `test.trg` may be
    left out). A dataset yaml may also list absolute paths instead.
  * `local_setting.sh` (untracked, one per machine): `export HFMT_DATA_ROOT=...`, where this
    machine keeps the data. Copy `local_setting.sh.example` to create it.
  * `sft_lora.yaml`: a **single-run** experiment (one job). It sets `run_name`, `checkpoint` and
    `instruction`, and picks options with `override <group>: <option>` lines.
  * `sft_lora.sweep.yaml`: a **sweep** experiment. Its grid in `hydra.sweeper.params` (here 2
    learning rates x 2 LoRA ranks = 4 jobs) always runs in full.

A key that no yaml sets, a typo, or a `peft.mode` that doesn't match its blocks fails before
anything is submitted.

### Training

Launch from a login node; jobs go to Slurm as one array, and the command returns right away:

```bash
export HFMT_ROOT=`pwd`
cp egs/mmtc/fr-en/local_setting.sh.example egs/mmtc/fr-en/local_setting.sh   # once; then edit the path
bash egs/launch.sh sft egs/mmtc/fr-en/sft_lora.yaml                       # 1 job
bash egs/launch.sh sft egs/mmtc/fr-en/sft_lora.sweep.yaml                 # 4 jobs
bash egs/launch.sh sft egs/mmtc/fr-en/sft_lora.yaml peft.lora.lora_r=16 train.max_steps=2000  # overrides
bash egs/launch.sh sft egs/mmtc/fr-en/sft_lora.yaml peft.lora.lora_r=8,16 # a comma list makes a grid
bash egs/launch.sh sft egs/mmtc/fr-en/sft_lora.yaml peft=lora_default     # another fine-tuning method
bash egs/launch.sh sft egs/mmtc/fr-en/sft_lora.yaml slurm.gres=gpu:l40s:1 # Slurm resources: slurm.*
bash tools/run_on_free_gpu.sh sft egs/mmtc/fr-en/sft_lora.yaml            # pick the freest GPU type
LOCAL=1 bash egs/launch.sh sft egs/mmtc/fr-en/sft_lora.yaml               # run here (e.g. on a GPU node)
DRYRUN=1 bash egs/launch.sh sft egs/mmtc/fr-en/sft_lora.sweep.yaml        # prepare, print sbatch, don't submit
```

Before submitting, every job's config is built and checked and saved, fully resolved, in the
job's **run folder**:

* one job: `egs/mmtc/fr-en/outputs/<run_name>/`
* grid: `egs/mmtc/fr-en/outputs/<run_name>-<idx>/`, with `<run_name>.grid.txt` listing each
  index's values. Change `run_name` when you change a grid's shape.

A run folder holds `run_config.yaml`, `git_rev`, the Slurm log, the training log,
`dev.step_*.pred`, `model/` (adapter or full weights, plus the tokenizer), and for the test sets
`test1.pred.trg`, `test2.pred.trg`, ... (in `data.test.src` order) with `scores.json`. Test
scores also go to the W&B run summary as `test/test<i>/bleu` etc. The W&B run has the same name as the
folder and is grouped by `run_name`; only config and aggregate metrics are sent to W&B. Launching
into an existing folder prints a warning and reuses it (training restarts from scratch).

Re-run a finished job from its saved config, in the same folder or under a new name:

```bash
bash egs/launch.sh sft egs/mmtc/fr-en/outputs/sft_lora_sweep-1/run_config.yaml train.max_steps=2000
bash egs/launch.sh sft egs/mmtc/fr-en/outputs/sft_lora_sweep-1/run_config.yaml train.seed=1 run_name=sft_lora-seed1
```

To debug one configuration directly (output goes to `outputs/<date>/<time>/` unless you pass
`hydra.run.dir=...`; add `--cfg job` to only print the config):

```bash
source egs/mmtc/fr-en/local_setting.sh
python -m hfmt.sft_translation --config-dir egs/mmtc/fr-en --config-name sft_lora train.max_steps=20
```

Decoding the dev set for BLEU during training is slow. `train.inference_callback_interval=N`
does it every N evaluations (and at the last step) instead of every evaluation; `0` turns it off.
`eval_loss` and early stopping still run at every evaluation, and a BLEU-based
`train.metric_for_best_model` keeps working (only evaluations with BLEU can be the best checkpoint).

`peft` selects `qlora_default` (default), `lora_default` or `full_finetune`. A full fine-tune
needs a lower learning rate and more memory and time, e.g.
`peft=full_finetune train.learning_rate=2e-5 checkpoint=Qwen/Qwen2.5-0.5B-Instruct slurm.mem_gb=64 slurm.time=12:00:00`.

### Inference

Decode one model per call. With a trained run's `run_config.yaml`, its `model/` is decoded; the
saved config gives the base model, fine-tuning method, prompt and decode settings. With an
experiment yaml, the experiment's `checkpoint` is decoded as downloaded (zero-shot); its `peft`
option only sets the precision (`lora_default` = bf16, `qlora_default` = 4-bit, `full_finetune` = fp32).

```bash
bash egs/launch.sh inf egs/mmtc/fr-en/outputs/sft_lora/run_config.yaml                     # data.test
bash egs/launch.sh inf egs/mmtc/fr-en/outputs/sft_lora/run_config.yaml decode.num_beams=5 inf_name=beam5
bash egs/launch.sh inf egs/mmtc/fr-en/outputs/sft_lora/run_config.yaml \
    'data.test.src=[/path/a.fr,/path/b.fr]' 'data.test.trg=[/path/a.en,null]' inf_name=ab
for rc in egs/mmtc/fr-en/outputs/sft_lora_sweep-*/run_config.yaml; do bash egs/launch.sh inf $rc; done
bash egs/launch.sh inf egs/mmtc/fr-en/sft_lora.yaml peft=lora_default                      # zero-shot
```

Output goes to `<run folder>/inf/<inf_name>/` for a trained run, and to
`<output_root>/<run_name>/base_ckpt_inf/<inf_name>/` for a downloaded checkpoint (`inf_name`
defaults to `test`). It holds `inf_config.yaml`, one `test<i>.pred.trg` per test file, and
`scores.json` with each test set's paths and, when it has a reference, BLEU/chrF/TER. A run
folder's `run_config.yaml` is never changed, and nothing is sent to W&B. With the run's own test
sets, the outputs are identical to those written at the end of training. `LOCAL=1`, `DRYRUN=1`
and `slurm.*` overrides work as for training.

### Merging and a second round of SFT

A LoRA or QLoRA run can be merged into a full checkpoint, e.g. to start another round of SFT from it:

```bash
bash egs/launch.sh merge egs/mmtc/fr-en/outputs/sft_lora/run_config.yaml   # -> sft_lora/model_merged/
```

`model_merged/` holds full bf16 weights, the tokenizer, and `merge_info.yaml` (source run, base
checkpoint, `git_rev`). A QLoRA adapter is merged into the 4-bit base converted back to bf16, the
weights it was trained against; a warning notes that the merged weights carry the 4-bit rounding.
Merged and adapter models score within a few tenths of BLEU of each other, but aren't identical.
Only `slurm.*` overrides apply; merging a run again replaces its `model_merged/`.

Use the merged model as the `checkpoint` of another experiment (or on the command line); its
training and inference work as for any run:

```bash
bash egs/launch.sh sft egs/mmtc/fr-en/sft_lora.yaml run_name=sft_lora_r2 \
    checkpoint=$PWD/egs/mmtc/fr-en/outputs/sft_lora/model_merged
bash egs/launch.sh inf egs/mmtc/fr-en/sft_lora.yaml run_name=sft_lora_merged \
    checkpoint=$PWD/egs/mmtc/fr-en/outputs/sft_lora/model_merged peft=lora_default   # merged model alone
```

The second round depends on the first run's folder: re-running or re-merging it changes the base
of the second round. Copy `model_merged/` somewhere outside `outputs/` for anything long-lived.
A full fine-tune's `model/` already is a full checkpoint (no merge needed).

## Usage example: decoding with a Summarization model

To do text summarization, we can run inference on a LLM using the appropriate prompt. 
Here is an example dataset in English, where the `text` field is an article to be summarized and the `summary` field is a reference: `egs/data/summarization.en-en.jsonl`

Here is an example script that runs `hfmt/decode_summarization.py` and compute ROUGE scores. 
```bash
qsub -S /bin/bash -V -cwd -j y -q gpu.q@@a100 -l gpu=1,h_rt=24:00:00,num_proc=8,mem_free=25G egs/summarization/decode_summarization.sh
```

