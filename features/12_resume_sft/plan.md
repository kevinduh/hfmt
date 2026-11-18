# 12_resume_sft — plan (v2, final)

Status: **v2, final.** Developer answers (2026-10-19):
- Q1: continue the same W&B run.
- Q2: no checkpoint: follow TRL's behavior (see below).
- Q3: dev outputs: overwriting is fine; follow TRL and change as little code as possible.
- Q4: keep TRL's default (early-stopping patience resets on resume).
- Q5: `LOCAL=1` appends to `local.log`.
- Q6: a finished run exits with an error.
- Names: the job kind is `sft_resume`, the environment variable `HFMT_SFT_RESUME` (it's an SFT thing).

## Request (summary)

`bash egs/launch.sh sft <run folder>/run_config.yaml` means **resume**: continue a training job
that was cancelled midway (time limit, OOM, node failure, scancel) from its last checkpoint,
with TRL's `trainer.train(resume_from_checkpoint=...)`.
- **Nothing about the run changes:** same config, same `run_name`, same folder. Only `slurm.*`
  may be overridden, e.g. a bigger GPU after an OOM.
- **Out of scope:** re-running, retargeting (`run_name=`, another folder) and changing
  hyperparameters. Use the experiment yaml + overrides for those.

## Starting point (`hydra3` @ 3a707c6)

**What `launch.sh sft <run_config.yaml>` does today** (`grid.py` `main_sft`):
- It composes the saved config + any overrides, and **rewrites** `run_config.yaml` and `git_rev`.
- `reuse_dir` (`grid.py:154`) puts the job in the same folder unless `run_name=` is given; with
  `run_name=` it goes to a new folder next to it.
- `job.sh sft` then trains **from scratch** (`trainer.train()` with no argument,
  `sft_translation.py:287`). The run's `model/`, test outputs and checkpoints are overwritten, and
  it starts a new W&B run with the same name.

**Group overrides:** a `run_config.yaml`'s defaults list is only `base_config`, so overrides like
`peft=qlora_default` already fail there. Leaf overrides (`train.max_steps=…`) work today; this
feature forbids them.

**TRL / Transformers behavior** (pinned Transformers 5.8.0, TRL 1.3.0; `SFTTrainer.train` is
`Trainer.train`):
- **`resume_from_checkpoint=True`** calls `get_last_checkpoint(output_dir)`, the highest-numbered
  `checkpoint-N/` folder. If there is none, it raises
  `ValueError: No valid checkpoint found in output directory (...)` (`trainer.py:1405-1408`).
  **Q2 answer:** no checkpoint means the job fails. We check for one on the login node too, so
  the failure comes before submitting, not after a queue wait.
- **What is restored:**
  - the model/adapter weights, optimizer, LR scheduler, grad scaler and RNG states;
  - `TrainerState` (global step, `log_history`, `best_metric`, `best_model_checkpoint`);
  - the position in the data: batches already seen in the current epoch are skipped.
- **Callback state is not restored by default.** `restore_callback_states_from_checkpoint`
  defaults to `False`. So our early-stopping patience counter restarts at 0 after a resume.
  `best_metric` lives in `TrainerState` and *is* restored. See Q4.
- **Checkpoints are not saved atomically.** No temp-folder + rename in this version: a job killed
  while saving can leave a half-written last `checkpoint-N/`, and resuming from it fails or
  loads garbage. `trainer_state.json` is written at the end of a save, so its presence is a
  cheap completeness check.
- **Steps between the last checkpoint and the kill are redone.** Their `dev.step_N.pred` files
  are overwritten (Q3: fine).

**Logs:**
- `sft_translation.log` uses Hydra's `FileHandler`, whose default mode is append, so a resume
  continues the same log.
- Each Slurm attempt already has its own `slurm-<jobid>.out`.
- `LOCAL=1` writes `local.log` with `>` (truncates).

**W&B:**
- `sft_translation.py` calls `wandb.init(name=<folder>, group=run_name, dir=outdir, ...)` with no
  `id`, so every launch is a new W&B run.
- The run ID is in the local folder name, `<run folder>/wandb/run-<date>_<time>-<id>/`, and
  `wandb/latest-run` links to the newest one. Example: m1 has three such folders, from three
  launches.

## Design

### 1. Login node: `grid.py`

`main_sft` gets one early branch: if `exp_name == SAVED`, call `main_resume` and return.

`main_resume` follows the `main_merge` pattern:
1. **Overrides:** only `slurm.*` is allowed. Anything else exits:
   `grid: resume only takes slurm.* overrides (use the experiment yaml to change a run)`.
2. **Config check:** compose + check the saved config with the overrides (`_compose_checked`).
   **Nothing is written:** `run_config.yaml` and `git_rev` stay as they are.
3. **Checks on the folder:**
   - `model/` exists: exit, "run already finished". If it was killed during test decoding, use
     `launch.sh inf`.
   - No `checkpoint-*`: exit, saying the run has no checkpoint and so can't be resumed.
   - The highest `checkpoint-N/` has no `trainer_state.json`: exit, saying it's incomplete and
     should be deleted. Not deleted automatically.
   - Current `git_rev` ≠ saved: **warn only.**
   - Log `resuming <folder> from checkpoint-N (step N of max_steps)`.
4. **Output:** print `JOBS=(<run folder>)`, `SBATCH_ARGS` from the (overridden) `slurm` block, and
   `KIND=sft_resume`. Slurm log goes to `<run folder>/slurm-%A.out`, as for sft.

The `reuse_dir` line and the `run_name=` escape are deleted from `main_sft`. A plain sft launch
(experiment yaml) always goes to `<output_root>/<run_name>[-idx]`, as now.

### 2. Passing the flag: `launch.sh` → `job.sh` → `sft_translation.py`

- **`launch.sh`:** `job=(egs/job.sh "${KIND:-$kind}")`. grid.py sets `KIND` only for a resume, so
  `squeue`/`DRYRUN` show `job.sh sft_resume <folder>`.
- **`job.sh`:** one new case, the same command as `sft` with an environment variable:
  `sft_resume) HFMT_SFT_RESUME=1 exec python -m hfmt.sft_translation --config-dir "$1" --config-name run_config hydra.run.dir="$1" ;;`
- **Why an environment variable:** Hydra owns the Python command line, and a config key
  (`resume:`) would end up in `run_config.yaml` and W&B and need a `+key` override. An
  environment variable keeps it out of the config.

### 3. Job: `sft_translation.py`

- `resume = os.environ.get("HFMT_SFT_RESUME") == "1"`.
- **Training:** `trainer.train(resume_from_checkpoint=True if resume else None)`. TRL finds the
  last checkpoint and raises if none exists.
- **W&B (Q1):** on resume, find the ID from `wandb/latest-run` (`run-…-<id>` → `<id>`) and call
  `wandb.init(..., id=<id>, resume="allow")`. With `"allow"`, a run missing on the server (e.g.
  deleted) starts a new run with that ID instead of failing. With no local `wandb/` folder (W&B
  was off), it starts a new run.
  - **Rows for redone steps are logged again.** The x-axis is `train/global_step`, so those steps
    show two points in the charts.
  - `_step` keeps counting from where the run stopped.
- Log `Resuming from checkpoint` (Transformers also logs the checkpoint path).

### 4. Docs

- **`launch.sh` comment:** `# resume a cancelled job from its last checkpoint (slurm.* overrides only)`.
- **README** "Running SFT experiments": replace the re-run paragraph (lines 81–85) with a resume
  paragraph. To re-run with changes, use the experiment yaml + overrides and a new `run_name`.
- **CLAUDE.md** "Launch path" bullet and the module docstring of `grid.py`.
- **`decisions.md`:**
  - `run_config.yaml` + `sft` means resume only;
  - why it is an environment variable and not a config key;
  - W&B continues the same run.

Expected size: roughly +40 lines in grid.py (mostly checks and messages) and −3 for `reuse_dir`,
+1 each in launch.sh and job.sh, about +10 in sft_translation.py, plus docs.

## Risks

1. **Callback state (Q4: TRL default kept):** patience restarts at 0. A resumed run may train
   up to `early_stopping_patience` evaluations longer than an uninterrupted one would.
   - **Not done:** `restore_callback_states_from_checkpoint=True`. But Transformers rebuilds the
     callback as `type(cb)(**saved_args)`, and our `EarlyStopping_MT_Callback.__init__()` takes no
     arguments, so that would raise `TypeError`. It would need `__init__(self, **kwargs)` plus the
     `SFTConfig` flag: about 3 lines.
2. **Not bit-identical to an uninterrupted run:**
   - a different GPU type (the point of `slurm.*` overrides) changes numerics;
   - the debug batch dump before training and dataloader worker RNG may differ slightly.

   Expected: close, not equal. Validation compares curves, not exact numbers.
3. **Code changed since the run:** resuming runs the current code on the old config. A new
   required schema key makes the old `run_config.yaml` fail the check, as for `inf`/`merge`
   (saved configs aren't backward compatible). The `git_rev` mismatch only warns.
4. **`save_total_limit` + `load_best_model_at_end`:** the best checkpoint is kept besides the
   last ones. `get_last_checkpoint` takes the highest step, which is the right one to resume.
5. **A sweep:** one resume per call. Loop in the shell, as for `inf`:
   `for rc in .../sft_lora_sweep-*/run_config.yaml; do bash egs/launch.sh sft $rc; done`.
   Folders that finished fail the `model/` check with a message, which is harmless in a loop but
   noisy (Q6: kept as an error).
6. **`LOCAL=1`** truncates `local.log`, losing the first attempt's local log. Switched to `>>`
   (Q5); `merge-local.log` too, for consistency.
7. **Two concurrent jobs in one folder** (resuming while the original still runs) would corrupt
   it. No lock. Docs say: check `squeue` first.

8. **TRL's `num_tokens` metric restarts from 0 on resume** (found in phase B: p12b logged
   11,949 at step 50 vs 60,130 uninterrupted). TRL keeps the count on the trainer, not in the
   checkpoint's `TrainerState`. Loss, LR, steps and epoch are right. Left as TRL behavior: compare
   resumed runs on `train/global_step` or epoch, not `num_tokens`.

## Decided questions

- **Q4.** TRL's default: no callback-state restore.
- **Q5.** `LOCAL=1` appends to `local.log`.
- **Q6.** A finished run (`model/` exists) exits with an error.

## Validation (sketch)

1. **Login node (`DRYRUN=1`, on a copy of a run folder in the scratchpad):**
   - a non-slurm override fails;
   - a `slurm.gres=` override shows in the sbatch line;
   - the line reads `job.sh sft_resume`;
   - a finished run, a run with no checkpoint, and an incomplete last checkpoint each fail with
     their message;
   - `run_config.yaml` is unchanged afterwards (checked with `cmp`).
2. **GPU:**
   - Start a short run (`p12a`, e.g. max_steps 60, save/eval every 20) and `scancel` it after
     checkpoint-40.
   - Resume it. The log shows resuming from checkpoint-40, and training ends at step 60 with
     `model/`, `test1.pred.trg` and `scores.json`.
   - W&B: one run (same ID), curves continue past step 40.
   - Compare against an uninterrupted run of the same config: steps line up, losses are close.
3. **Tests:** `HFMT_ROOT=$(pwd) python hfmt/test_grid.py`, plus new cases for the overrides check and
   the folder checks.
