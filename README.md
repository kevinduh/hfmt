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
* `hfmt/sft_translation.py`: Decoder-only MT via QLoRA SFT. **Hydra-configured** (see "Running QLoRA SFT experiments with Hydra" below)
* `hfmt/inf_translation.py`: Inference for decoder-only MT (optionally with a LoRA adapter)
* `hfmt/decode_summarization.py`: Runs inference on CausalLM models, with prompts for summarization
* todo: cascading of models, ...

Scripts that perform training integrate with (Weights & Biases)[https://wandb.ai/] for logging purposes, so it is recommended that you set up a free account on that service. 

## Running QLoRA SFT experiments with Hydra

The decoder-only QLoRA SFT workflow (`hfmt/sft_translation.py`) is configured with
[Hydra](https://hydra.cc): every tunable parameter lives in a config file under `conf/`,
not in the training script or a per-experiment shell script.

### Config layout (`conf/`)

* `conf/config.yaml` — top-level defaults + Hydra settings (output dir, launcher).
* `conf/model/` — checkpoint, 4-bit quantization, LoRA (`r`/`alpha`/`dropout`/`target`).
* `conf/data/` — train/dev/test paths (a YAML manifest) + instruction prefix. **Paths only, never data content.**
* `conf/train/` — learning rate, schedule, steps, batch size, early stopping, etc.
* `conf/decode/` — generation knobs (`max_length`, `max_new_tokens`, `num_beams`, ...).
* `conf/wandb/` — W&B project/entity/group/tags.
* `conf/experiment/<name>.yaml` — a preset composing the above into one named experiment (the successor to the old per-experiment `.sh` recipes).
* `conf/sweep/<name>.yaml` — a committed parameter-sweep preset (a grid for Hydra's basic sweeper), selected with `+sweep=<name>` (see **Sweeping parameters**).
* `conf/hydra/launcher/{slurm,local}.yaml` — how/where to run (Slurm via submitit, or local).

The typed schema in `hfmt/hydra_config.py` validates all config at compose time — wrong types or unknown keys are rejected before anything loads.

### Running an experiment

Set `HFMT_ROOT`, then launch by experiment name. Run from a **login node** — the submitit launcher submits the Slurm job for you (do **not** `sbatch` this script):

```bash
export HFMT_ROOT=`pwd`
bash egs/run.sh mmtc_fr-en_sft1
```

Override any parameter on the command line (forwarded to Hydra):

```bash
bash egs/run.sh mmtc_fr-en_sft1 model.lora_r=16 train.learning_rate=2e-5
```

Preview the fully-composed config without running anything:

```bash
python hfmt/sft_translation.py +experiment=mmtc_fr-en_sft1 --cfg job
```

Run in-process instead of submitting (e.g. when already on a GPU node):

```bash
python hfmt/sft_translation.py +experiment=mmtc_fr-en_sft1
```

### Adding a new experiment

Add a file `conf/experiment/<name>.yaml` (copy `mmtc_fr-en_sft1.yaml`), point it at your data config and set any overrides, then `bash egs/run.sh <name>`. No new shell script needed.

Outputs (checkpoints, predictions, logs) land under `outputs/<experiment>/<timestamp>/`, and the W&B run is named/grouped by the experiment and linked to that output dir.

### Sweeping parameters

A sweep runs an experiment over multiple parameter values, fanning out into **one Slurm job per combination** via the submitit launcher (`egs/run.sh` already runs in `--multirun` mode). Two ways to drive it:

**Ad-hoc** — pass comma-separated values on the CLI; Hydra takes the cross-product:

```bash
bash egs/run.sh mmtc_fr-en_sft1 model.lora_r=8,16,32 train.learning_rate=2e-5,2e-4
```

The override grammar also supports `range(1,4)`, `choice(a,b)`, and `glob(*)`, and you can sweep a whole config group (e.g. `model=qwen2.5-1.5b,qwen2.5-7b`).

**Committed preset** — define a reusable grid in `conf/sweep/<name>.yaml` and select it:

```bash
bash egs/run.sh mmtc_fr-en_sft1 +sweep=mmtc_fr-en_coarse
```

See `conf/sweep/mmtc_fr-en_coarse.yaml` for the format (it sets `hydra.sweeper.params`).

Notes:

* **Grids are cross-products** — the job count multiplies fast (a 7-way space can be ~200 runs). At most `hydra.launcher.array_parallelism` jobs (default **4**, the node-tier size) run at once; the rest queue. Override per-sweep with `hydra.launcher.array_parallelism=<N>`. Start from a coarse preset and refine.
* **Runs don't clobber.** Each job gets its own `outputs/<experiment>/<n>_<timestamp>/` dir and a distinct W&B run id.
* **Runs are identifiable.** A sweep's runs share one W&B **group** (`<experiment>-sweep-<timestamp>`) and each run is **named/tagged by its swept values** (e.g. `model.lora_r=16,train.seed=37`), so you can compare them at a glance. Only hyperparameters appear in names/tags — never data content (data paths are stripped).

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

## Usage example: decoding with a Summarization model

To do text summarization, we can run inference on a LLM using the appropriate prompt. 
Here is an example dataset in English, where the `text` field is an article to be summarized and the `summary` field is a reference: `egs/data/summarization.en-en.jsonl`

Here is an example script that runs `hfmt/decode_summarization.py` and compute ROUGE scores. 
```bash
qsub -S /bin/bash -V -cwd -j y -q gpu.q@@a100 -l gpu=1,h_rt=24:00:00,num_proc=8,mem_free=25G egs/summarization/decode_summarization.sh
```

## Exporting W&B results for analysis

`analysis/pull_wandb.py` exports finished runs from Weights & Biases into a single JSON file
(`analysis/wandb_export.json`) that you can then review — e.g. by handing it to Claude Code to
discuss what to change. It only reads metrics + config (no dataset content), and the output is
gitignored.

Run it **outside the sandbox**, on a machine with outbound access to W&B, using the project
environment (which already includes `wandb`; see [Installation](#installation)):

1. Set your W&B API key. It is read from the environment or `~/.netrc` — it is **never** passed
   as a command-line argument (so it can't leak into shell history or logs):

   ```bash
   export WANDB_API_KEY=<your key>
   # or run: wandb login
   ```

2. Pull the runs (defaults: project `hfmt`, latest 10 finished runs):

   ```bash
   python analysis/pull_wandb.py
   # optional filters:
   #   --project hfmt --entity <you> --group <group> --run-id <id> --limit 20 --out path.json
   ```

The export is written to `analysis/wandb_export.json` (override with `--out`).

By default it is **reduced** to stay small enough to compare many runs at once: each run
keeps its `config.hfmt` knobs plus per-metric `[step, value]` trajectories (`eval` and
`train`) at the run's own logging cadence. Pass `--raw` to export the full history + full
config for a single run you want to drill into.

> Note: `analysis/pull_wandb.py` is added in a later step of this feature; this section
> documents how it will be run.

