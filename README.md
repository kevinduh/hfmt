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
* `hfmt/sft_translation.py`: Supervised fine-tuning (QLoRA, LoRA or full) of decoder-only LLMs for MT, configured with Hydra (see below)
* `hfmt/inf_translation.py`: Inference with a fine-tuned run or a downloaded checkpoint; `hfmt/merge_model.py`: merges a LoRA/QLoRA run into a full checkpoint (see below)

Scripts that perform training integrate with (Weights & Biases)[https://wandb.ai/] for logging purposes, so it is recommended that you set up a free account on that service. 

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

### Training

Launch from a login node; jobs go to Slurm as one array, and the command returns right away:

```bash
export HFMT_ROOT=`pwd`
cp egs/mmtc/fr-en/local_setting.sh.example egs/mmtc/fr-en/local_setting.sh   # once; then edit the path
bash egs/launch.sh sft egs/mmtc/fr-en/sft_lora.yaml                       # 1 job
bash egs/launch.sh sft egs/mmtc/fr-en/sft_lora.sweep.yaml                 # 4 jobs
bash egs/launch.sh sft egs/mmtc/fr-en/sft_lora.yaml peft.lora.r=16 train.max_steps=2000       # overrides
bash egs/launch.sh sft egs/mmtc/fr-en/sft_lora.yaml peft.lora.r=8,16      # a comma list makes a grid
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

The LoRA options set `peft.lora.r`, `alpha`, `dropout` and `target_modules`. `target_modules` is a
list of exact module names. Names differ between model families, so look them up for a new checkpoint:

```bash
python tools/inspect_model.py Qwen/Qwen2.5-1.5B-Instruct     # a model id or a local folder
```

### Inference

Decode one model per call. With a trained run's `run_config.yaml`, its `model/` is decoded; the
saved config gives the base model, fine-tuning method, prompt and decode settings. With an
experiment yaml, the experiment's `checkpoint` is decoded as downloaded (zero-shot); its `peft`
option only sets the precision (`lora_default` = bf16, `qlora_default` = 4-bit, `full_finetune` = fp32).

```bash
bash egs/launch.sh inf egs/mmtc/fr-en/outputs/sft_lora/run_config.yaml  # decoding with trained model
bash egs/launch.sh inf egs/mmtc/fr-en/sft_lora.yaml                     # decoding with base model (checkpoint)

bash egs/launch.sh inf egs/mmtc/fr-en/outputs/sft_lora/run_config.yaml decode.num_beams=5 inf_name=beam5
bash egs/launch.sh inf egs/mmtc/fr-en/outputs/sft_lora/run_config.yaml \
    'data.test.src=[/path/a.fr,/path/b.fr]' 'data.test.trg=[/path/a.en,null]' inf_name=ab
for rc in egs/mmtc/fr-en/outputs/sft_lora_sweep-*/run_config.yaml; do bash egs/launch.sh inf $rc; done
```

Output goes to `<run folder>/inf/<inf_name>/` for a trained run, and to
`<output_root>/<run_name>/base_ckpt_inf/<inf_name>/` for a base model checkpoint (`inf_name`
defaults to `test`). It holds `inf_config.yaml`, one `test<i>.pred.trg` per test file, and
`scores.json` with each test set's paths and, when it has a reference, BLEU/chrF/TER. A run
folder's `run_config.yaml` is never changed, and nothing is sent to W&B. With the run's own test
sets, the outputs are identical to those written at the end of training. `LOCAL=1`, `DRYRUN=1`
and `slurm.*` overrides work as for training.

### Merging and a second round of SFT

A LoRA run can be merged into a full checkpoint, e.g. to start another round of SFT from it:

```bash
bash egs/launch.sh merge egs/mmtc/fr-en/outputs/sft_lora/run_config.yaml   # -> sft_lora/model_merged/
```

`model_merged/` holds full weights, the tokenizer, and `merge_info.yaml` (source run, base
checkpoint, `git_rev`). Use the merged model as the `checkpoint` of another experiment.

```bash
bash egs/launch.sh sft egs/mmtc/fr-en/sft_lora.yaml run_name=sft_lora_r2 \
    checkpoint=$PWD/egs/mmtc/fr-en/outputs/sft_lora/model_merged
```
