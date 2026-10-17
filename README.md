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

`conf/` holds only **atomistic, mix-and-match components** (feature 04); an experiment composes them and lives under `egs/`:

* `conf/config.yaml` — top-level defaults + Hydra settings (output dir, launcher, and `hydra.searchpath` pointing at `egs/`).
* `conf/basemodel/` — the pretrained base model: checkpoint, 4-bit quantization, load `dtype`.
* `conf/lora/` — the LoRA adapter (`r`/`alpha`/`dropout`/`target`). Whether it is used is decided by `train.finetune_mode` (`qlora` | `lora` | `full`); see "Full fine-tuning" below.
* `conf/train/` — learning rate, schedule, steps, batch size, early stopping, etc.
* `conf/decode/` — generation knobs (`max_length`, `max_new_tokens`, `num_beams`, ...).
* `conf/hydra/launcher/` — launcher profiles (where/how to run): `slurm` (the QLoRA baseline), `local`. These are selectable components too — each experiment picks one.
* `egs/<dataset>/<src>-<trg>/<name>.yaml` — a preset composing the components above into one named experiment, plus an inline `data` block and `wandb`; a *sweep* experiment also carries a `hydra.sweeper.params` grid (feature 06/07; the successor to the old `egs/experiment/` layout and the per-experiment `.sh` recipes). Selected as the nested Hydra group `+<dataset>/<src>-<trg>=<name>`, found via `hydra.searchpath`.
* `egs/<dataset>/<src>-<trg>/dataset.yaml` — a per-lang-pair **template** giving the train/dev/test file paths with `$DATA_ROOT`/`$SRC`/`$TRG` placeholders, inflated in code from the experiment's `data.data_root`/`src`/`trg` (feature 06, replacing the old `conf/data` group). **Paths only, never data content.** `test` is optional — if its files are absent, the final held-out inference pass is skipped.

An experiment is a flat list of component *selections* — base model, train, lora, decode, **and launcher** — plus an inline `data` block (`data_root`/`src`/`trg`/`instruction`), so the end user reads only choices, never buried settings; the complex presets live in their own `conf/` files. Every experiment selects a launcher (`- override /hydra/launcher: slurm`); one that selects none falls back to Hydra's local `basic` launcher (it will not submit to Slurm).

W&B settings are no longer a `conf/` group: their defaults come from the `WandbConfig` dataclass and each experiment sets `wandb.group`/`wandb.tags` inline. The typed schema in `hfmt/hydra_config.py` validates all config at compose time — wrong types or unknown keys are rejected before anything loads.

### Running an experiment

Set `HFMT_ROOT`, then launch by the experiment's path under `egs/` (the `.yaml` file; a trailing `.yaml` is optional). Run from a **login node** — the submitit launcher submits the Slurm job for you (do **not** `sbatch` this script):

```bash
export HFMT_ROOT=`pwd`
bash egs/run.sh mmtc/fr-en/sft_lora.yaml
```

Override any parameter on the command line (forwarded to Hydra):

```bash
bash egs/run.sh mmtc/fr-en/sft_lora.yaml lora.lora_r=16 train.learning_rate=2e-5
```

Preview the fully-composed config without running anything (pass the nested selector + the dataset dir yourself; `--cfg` can't go through `run.sh`, which always uses `--multirun`):

```bash
python hfmt/sft_translation.py +mmtc/fr-en=sft_lora data.dataset_dir=$HFMT_ROOT/egs/mmtc/fr-en --cfg job
```

Run in-process instead of submitting (e.g. when already on a GPU node):

```bash
python hfmt/sft_translation.py +mmtc/fr-en=sft_lora data.dataset_dir=$HFMT_ROOT/egs/mmtc/fr-en
```

### Adding a new experiment

Add a file `egs/<dataset>/<src>-<trg>/<name>.yaml` (copy `egs/mmtc/fr-en/sft_lora.yaml`, or `sft_lora.sweep.yaml` for a sweep) next to a `dataset.yaml` template, select your components (`override /basemodel:`, `/lora:`, `/hydra/launcher:`, ...), fill the inline `data` block (`data_root`/`src`/`trg`/`instruction`; `dataset_file` defaults to `dataset.yaml`), set any value overrides, then `bash egs/run.sh <dataset>/<src>-<trg>/<name>.yaml`. No new shell script needed. Remember to select a launcher (`- override /hydra/launcher: slurm`).

#### Full fine-tuning

`train.finetune_mode` selects the method: `qlora` (default, LoRA on a 4-bit base), `lora` (LoRA on an unquantized base), or `full` (all parameters, unquantized base, no adapter). A full fine-tune needs more host RAM and walltime than QLoRA, so raise the launcher limits too, e.g.:

```bash
bash egs/run.sh mmtc/fr-en/sft_lora.yaml train.finetune_mode=full train.learning_rate=2e-5 \
    basemodel=qwen2.5-0.5b hydra.launcher.mem_gb=64 hydra.launcher.timeout_min=720
```

(These are the values the former `slurm_fullft` launcher and `sft1_fullft` experiment used. 7B does not fit a single GPU for a full fine-tune.) To make it permanent, put the same values in an experiment file (`train:` block, `override /basemodel:`, and a `hydra.launcher:` block).

Outputs (checkpoints, logs, the trained adapter under `model/`, and predictions `eval.pred.trg` / `dev.step_*.pred`) land under `egs/<dataset>/<src>-<trg>/outputs/<experiment>/<timestamp>/` (feature 06; gitignored), and the W&B run is named/grouped by the experiment and linked to that output dir. Score the predictions with:

```bash
python analysis/run_sacrebleu.py --ref <ref.trg> --hyp egs/<dataset>/<src>-<trg>/outputs/<experiment>/<timestamp>/eval.pred.trg
```

### Sweeping parameters

A sweep runs an experiment over multiple parameter values, fanning out into **one Slurm job per combination** via the submitit launcher. There are two kinds of experiment file (feature 07):

* **Single-run experiment** (e.g. `egs/mmtc/fr-en/sft_lora.yaml`): no grid, so it always launches exactly one job.
* **Sweep experiment** (e.g. `egs/mmtc/fr-en/sft_lora.sweep.yaml`): carries a grid in Hydra's native `hydra.sweeper.params`, and **always runs the full grid** when invoked.

```bash
bash egs/run.sh mmtc/fr-en/sft_lora.yaml          # 1 job
bash egs/run.sh mmtc/fr-en/sft_lora.sweep.yaml    # 4 jobs (2 learning_rate x 2 lora_r)
```

The grid in a sweep experiment is a plain block in the yaml. Each axis is an ordinary Hydra override: a leaf value **or a whole component file**:

```yaml
hydra:
  sweeper:
    params:
      train.learning_rate: 1e-4,2e-4
      lora.lora_r: 8,16
      # basemodel: qwen2.5-0.5b,qwen2.5-1.5b   # sweep across entire base-model configs
```

Why two files instead of an on/off switch: `egs/run.sh` always runs `--multirun` (the submitit launcher requires it), and Hydra cannot switch off a grid stored in config from the command line. So a grid is its own experiment file. To run one setting from a sweep file, pin each swept key on the CLI (a single value replaces that axis, e.g. `train.learning_rate=2e-4 lora.lora_r=8`), or use the single-run file.

Runs are labelled by the swept values — and for a component-file axis, by the **selected YAML file name** (e.g. `basemodel-qwen2.5-1.5b`), not the value inside it. Before submitting, the launcher prints the sweep output dir and one `#<idx> : <overrides>` line per job, so you can see the grid that is about to be submitted.

**Ad-hoc** — pass comma-separated values on the CLI for any experiment; Hydra takes the cross-product (combined with any grid in the file):

```bash
bash egs/run.sh mmtc/fr-en/sft_lora.yaml lora.lora_r=8,16,32 train.learning_rate=2e-5,2e-4
```

The override grammar also supports `range(1,4)`, `choice(a,b)`, and `glob(*)`, and you can sweep a whole config group (e.g. `basemodel=qwen2.5-1.5b,qwen2.5-7b`).

Notes:

* **Grids are cross-products** — the job count multiplies fast (a 7-way space can be ~200 runs). At most `hydra.launcher.array_parallelism` jobs (default **4**, the node-tier size) run at once; the rest queue. Override per-sweep with `hydra.launcher.array_parallelism=<N>`. Start from a coarse grid and refine it in a new sweep file.
* **Outputs mirror W&B, so a run maps 1:1 to its files.** A sweep's jobs land under one shared `outputs/<experiment>/sweep-<timestamp>/` (the `sweep-<timestamp>` dir == the W&B **group** `<experiment>-sweep-<timestamp>`), and each job's dir is named by its swept values — e.g. `learning_rate-0.0002__lora_r-16/`. **The W&B run name prefixes the experiment onto that label** (e.g. `mmtc_fr-en_sft_lora_sweep-learning_rate-0.0002__lora_r-16`) so a flat run list stays unambiguous across experiments that sweep the same axis; the run *directory* leaf keeps the bare label, so from a run in the W&B UI you strip the `<experiment>-` prefix to `cd` to its `eval.pred.trg`. Names are shell-safe (`[A-Za-z0-9._-]`; keys verbatim, pairs joined by `__`). Runs never clobber (distinct dir + W&B run id).
* **Runs are identifiable.** Each run is **named/tagged by its swept values**, so you can compare them at a glance. Only hyperparameters appear in names/tags — never data content (data paths are stripped). Note values are shown as resolved numbers, so `learning_rate=2e-4` appears as `learning_rate-0.0002`.

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

