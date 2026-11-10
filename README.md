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

* `hfmt/`: the Python entry points. `sft_translation.py` (supervised fine-tuning of decoder-only LLMs
  for MT: QLoRA, LoRA or full), `inf_translation.py` (inference with a trained run or a downloaded
  checkpoint) and `merge_model.py` (merge a LoRA/QLoRA run into a full checkpoint) are configured with
  Hydra and launched with `egs/launch.sh` (next section). `train_seq2seq.py` (encoder-decoder MT) and
  `decode_summarization.py` take command-line arguments.
* `egs/`: recipes. `egs/mmtc/<src>-<trg>/` holds the Hydra SFT experiments. `egs/translation/` and
  `egs/summarization/` are plain shell scripts for the two command-line entry points: they activate
  the conda environment with `install/path.sh` (edit it for your system) and need
  `export HFMT_ROOT=path/to/this/repo` to find everything; `egs/launch.sh` finds the repo itself.
* `conf/`: the SFT base config and its option groups. `tools/`: cluster helpers (GPU availability,
  run status, model inspection, checkpoint cleanup). `analysis/`: a standalone BLEU/chrF/TER scorer.

Training logs to [Weights & Biases](https://wandb.ai/), so set up a free account. Only config and
aggregate metrics are sent there, never data (see `CLAUDE.md`).

## Running SFT experiments (decoder-only MT, Hydra)

`hfmt/sft_translation.py` (training) and `hfmt/inf_translation.py` (inference) are configured
with [Hydra](https://hydra.cc). The schema in `hfmt/hydra_config.py` defines every key and its
type but holds no default values; defaults live in yaml only:

* `conf/config.yaml`: the defaults list and project-wide settings (`slurm`, `wandb`, `inf_name`).
* `conf/<group>/<option>.yaml`: swappable bundles, selected with `override <group>: <option>` in an
  experiment or `<group>=<option>` on the command line. Each file's header comment says what it is for.
  * `train/`: the schedules. `shortrun` is the default; `midrun`, `longrun`, `full_finetune_short` and
    `lora_anneal_short` are the others.
  * `peft/`: the fine-tuning method. `qlora_default` (4-bit base + LoRA), `lora_default` (bf16 base +
    LoRA), `full_finetune`, and per-model all-linear LoRA options (`lora_qwen3_5_all_linear`,
    `lora_gemma3_all_linear`).
  * `decode/`: generation settings (`default`).
* the experiment folder, e.g. `egs/mmtc/fr-en/`:
  * `dataset.yaml`: `data_root`, `src`, `trg` and the train/dev/test file lists. Several train or
    dev files are concatenated in order; several test files are each decoded and scored on their
    own (a `test.trg` entry may be `null` for a test set without reference, or `test.trg` may be
    left out). A dataset yaml may also list absolute paths instead.
  * `local_setting.sh` (untracked, one per machine): `export HFMT_DATA_ROOT=...`, where this
    machine keeps the data. Copy `local_setting.sh.example` to create it.
  * `sft_lora.yaml`: a **single-run** experiment (one job): LoRA on Qwen3.5-2B with the default
    schedule. It sets `run_name`, `checkpoint` and `instruction`, and picks options with
    `override <group>: <option>` lines.
  * `sft_lora.sweep.yaml`: a **sweep** experiment. Its grid in `hydra.sweeper.params` (here 2
    learning rates x 2 LoRA ranks = 4 jobs) always runs in full.
  * the other yamls in the folder are the experiments run so far (Qwen3.5 0.8B to 9B and Gemma 3 4B;
    LoRA, QLoRA and full fine-tunes); each header says what it tests and why.

**Prompt.** `instruction` is prefixed to each source sentence as the user turn and rendered with the
checkpoint's chat template; every render passes `enable_thinking=False` (`hfmt/mt_model.py`
`CHAT_TEMPLATE_KWARGS`), which turns off Qwen3.5-4B's default thinking and is ignored by templates
without it.

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
`dev.step_*.pred`, `trainer_state.json` (the Trainer's final state: every evaluation's loss and
dev BLEU/chrF/TER in `log_history`, `best_metric`, `best_global_step`), `model/` (adapter or full
weights, the tokenizer, and `training_info.yaml`: the step whose weights it holds, the best
`metric_for_best_model` value and step, whether training stopped early, and the dev scores of the saved
weights — `dev.model.pred` — with `model_dev_check`, which compares them with that step's dev BLEU), and for the test sets
`test1.pred.trg`, `test2.pred.trg`, ... (in `data.test.src` order) with `scores.json`. Test
scores also go to the W&B run summary as `test/test<i>/bleu` etc. The W&B run has the same name as the
folder and is grouped by `run_name`; only config and aggregate metrics are sent to W&B. Launching
into an existing folder prints a warning and reuses it (training restarts from scratch).

**How the saved model is checked.** `model/` is meant to hold the best checkpoint
(`load_best_model_at_end`). For a full fine-tune, training loads it itself with
`mt_model.load_checkpoint_weights` (`from_pretrained`, then a strict `load_state_dict`), because the
Trainer's own reload matches saved weight names against module names and for Qwen3.5 loads nothing and
only warns. Training then decodes dev once more with the saved `model/` (`dev.model.pred`) and compares
its BLEU with the callback's BLEU at that step: within 1 BLEU is `model_dev_check: ok` in
`training_info.yaml`, otherwise `MISMATCH: ...` (logged, not fatal). `tools/run_status.py` reports
mismatches and known-bad library warnings found in the logs.

Full fine-tunes trained before 2026-11-05 (Qwen3.5 for sure; any run whose Slurm log has
"There were missing keys in the checkpoint model loaded" and no `model_dev_check`): `model/`, both
`scores.json` and `training_info.yaml`'s `model_global_step` are the *last* step, not the best
checkpoint — the Trainer's reload loaded nothing (see `decisions.md`). `checkpoint-<best_global_step>/`
is kept, so decode it instead (an experiment yaml with `checkpoint:` set to that folder,
`peft=full_finetune`).

**Resume** a job that was cancelled midway (time limit, OOM, node failure, `scancel`) from its
last `checkpoint-N/`, by passing its run folder's `run_config.yaml`:

```bash
bash egs/launch.sh sft egs/mmtc/fr-en/outputs/sft_lora_sweep-1/run_config.yaml
bash egs/launch.sh sft egs/mmtc/fr-en/outputs/sft_lora/run_config.yaml slurm.gres=gpu:h100:1   # e.g. after an OOM
```

**What carries over.** The run continues in the same folder with the same config, using TRL's
`resume_from_checkpoint`. It restores the weights, optimizer, LR schedule, step count and position
in the data. The W&B run continues too (same run ID).

**What may change: `slurm.*` overrides only.** Anything else is refused.

**When it's refused:**
- the run has no checkpoint yet (launch the experiment again);
- it already finished (`model/` exists; use `launch.sh inf`);
- its newest checkpoint is incomplete (no `trainer_state.json`; delete that folder and resume
  again).

If the code changed since the launch (`git_rev`), you get a warning.

**Not restored (TRL's defaults):**
- the early-stopping patience counter starts again at 0;
- TRL's `num_tokens` metric starts again from 0;
- steps after the last checkpoint are trained again (their W&B rows appear twice).

Check `squeue` first: never resume a job that is still running.

**To change a run, don't resume it.** Launch the experiment yaml again with overrides and a new
`run_name`, e.g. `bash egs/launch.sh sft egs/mmtc/fr-en/sft_lora.yaml train.seed=1 run_name=sft_lora-seed1`.

**Checking many runs at once.** `tools/run_status.py` lists every run folder with its training and
test-inference state and prints the command that fixes each unfinished one. It only prints; you
review and run the commands. States:
- finished: trained, and `launch.sh inf` has scored the test sets; or queued/running (from `squeue`);
- trained, inference not run, failed or queued (it prints `launch.sh inf` when nothing is queued);
- interrupted (it prints the resume command above);
- failed before its first checkpoint (it prints an `sbatch egs/job.sh sft <run folder>` that reruns
  just that folder from its saved config; resume refuses these);
- a `checkpoint-N/` or `model/` cut off while saving (it says which folder to delete).

Each row shows the test scores of training's own end-of-run decoding and of `launch.sh inf`, and
whether inference reproduced training (same scores, identical `test<i>.pred.trg`). For failed jobs it
also shows the error line and the node, and points out nodes where several jobs failed:

```bash
python tools/run_status.py egs/mmtc/fr-en/outputs                         # every run + fix commands
python tools/run_status.py egs/mmtc/fr-en/outputs --slurm exclude=<node>  # fix commands avoid that node
```

**Housekeeping.** Runs you are done with can be moved into `egs/mmtc/fr-en/outputs/complete/`;
`run_status.py` skips that folder unless you pass it itself. Checkpoints are the bulk of the disk use
(a 4B full fine-tune checkpoint is ~50 GB); `bash tools/delete_checkpoints.sh egs/mmtc/fr-en/outputs`
lists and, after you confirm, deletes the `checkpoint-*` folders of finished runs only (those with
`model/`), so unfinished runs stay resumable. `trainer_state.json` and `model/training_info.yaml` keep
the per-evaluation scores and the best step.

To debug one configuration directly (output goes to `outputs/<date>/<time>/` unless you pass
`hydra.run.dir=...`; add `--cfg job` to only print the config):

```bash
source egs/mmtc/fr-en/local_setting.sh
python -m hfmt.sft_translation --config-dir egs/mmtc/fr-en --config-name sft_lora train.max_steps=20
```

Decoding the dev set for BLEU during training is slow. `train.inference_callback_interval=N`
does it every N evaluations (and at the last step) instead of every evaluation; `0` turns it off.
Early stopping still runs at every evaluation. With a BLEU-type `metric_for_best_model`, the
evaluations without decoding get a never-best placeholder so the Trainer's best-checkpoint selection
keeps working (the placeholder never reaches W&B or `trainer_state.json`).

The LoRA options set `peft.lora.r`, `alpha`, `dropout` and `target_modules`. `target_modules` is a
list of exact module names. Names differ between model families, so look them up for a new checkpoint:

```bash
python tools/inspect_model.py Qwen/Qwen3.5-2B                # a model id or a local folder
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
for rc in egs/mmtc/fr-en/outputs/sft_lora_sweep-*/run_config.yaml; do bash egs/launch.sh inf $rc; sleep 3; done
```

Output goes to `<run folder>/inf/<inf_name>/` for a trained run, and to
`<output_root>/<run_name>/base_ckpt_inf/<inf_name>/` for a base model checkpoint (`inf_name`
defaults to `test`). It holds `inf_config.yaml`, one `test<i>.pred.trg` per test file, and
`scores.json` with each test set's paths and, when it has a reference, `bleu` (sacrebleu, flores200
tokenization), `bleu_13a_lowercase` (13a tokenization, lowercased, exp smoothing), `chrf` and `ter`. A
run folder's `run_config.yaml` is never changed, and nothing is sent to W&B. With the run's own test
sets, the outputs are identical to those written at the end of training (training unwraps
Accelerate's autocast wrapper before its test decoding, and a QLoRA base gets the same fp32
preparation in both). `LOCAL=1`, `DRYRUN=1` and `slurm.*` overrides work as for training.

Saved configs aren't backward compatible: a run saved before a new required key was added fails the
checks (pass the key on the command line), and one that carries a key the schema has since dropped
(e.g. `prompt_format`, written before 2026-11-07) is refused with the `sed` command that removes the line.

### Merging and a second round of SFT

A LoRA run can be merged into a full checkpoint, e.g. to start another round of SFT from it:

```bash
bash egs/launch.sh merge egs/mmtc/fr-en/outputs/sft_lora/run_config.yaml   # -> sft_lora/model_merged/
```

`model_merged/` holds full bf16 weights, the tokenizer, and `merge_info.yaml` (source run, base
checkpoint, `git_rev`). Only `slurm.*` overrides apply. A QLoRA run is merged onto its 4-bit base
dequantized to bf16, i.e. the weights the adapter was trained against (a warning notes the 4-bit
rounding; merged and adapter outputs were within ~0.3 BLEU). Use the merged model as the `checkpoint`
of another experiment.

```bash
bash egs/launch.sh sft egs/mmtc/fr-en/sft_lora.yaml run_name=sft_lora_r2 \
    checkpoint=$PWD/egs/mmtc/fr-en/outputs/sft_lora/model_merged
```
