# Decisions

Important, non-obvious decisions and their context, in chronological order. Update infrequently.
Headings are `## YYYY-MM-DD: title (feature directory, if any)`. The decisions of the first Hydra
integration (features 00–07, branch `hydra2`), superseded by the 2026-10-17 rebuild, are in
`decisions_hydra2.md`.

---

## 2026-10-17: Hydra redone from scratch: sbatch launcher, run_name folders, saved configs (08_hydra_redo)

**Decision:** On branch `hydra3` (from `main`), the SFT workflow's Hydra integration was rebuilt
in one feature, using the lessons of features 00–07 (branch `hydra2`; their decisions are kept in
`decisions_hydra2.md`). Those entries are **superseded** where they conflict: submitit and `--multirun` (00, 02),
run labels from `override_dirname` and the `hfmt_runlabel` resolver (03, 05), the
`hydra.searchpath` + nested-group selector and the sweep probe (04), and the `dataset.yaml`
template filled in by code (06). Feature 07's two kinds of experiment are kept.

- **Launch: `egs/sft.sh` → `hfmt/grid.py` → `sbatch --array` of `egs/job.sh`; no submitit.** The
  submitit launcher forced `--multirun` for every run, which caused the plain-run/sweep special
  cases, a login process that blocks until all jobs finish, at least three compositions per job,
  and `main` being pickled by value from `__main__` (so module code didn't re-run on the node,
  which hydra2's comments misdescribed). Now `grid.py` expands the grid on the login node with
  Hydra's own `BasicSweeper.split_arguments` (same rules as before: file axes + command-line
  comma lists, a single value pins), and each job is a plain single Hydra run. `LOCAL=1` runs
  the same jobs locally. Given up: submitit's results collection and resubmit-on-preemption.
  `grid.py` uses Hydra internals (`OverridesParser`, `BasicSweeper`); `hfmt/test_grid.py`
  covers it.
- **Every job's config is composed before submitting and saved fully resolved** as
  `<run folder>/run_config.yaml` (with `git_rev`). Typos and bad grid values fail on the login
  node; queued jobs run exactly the validated config even if the yaml sources are edited; a
  finished job can be re-run from its folder. Named `run_config.yaml`, not `config.yaml`: with
  `--config-dir <job dir> --config-name config`, Hydra silently loads `conf/config.yaml`
  instead (tested), so `sft.sh` also refuses experiments named `config.yaml`.
- **Run identity is just `run_name`** (required in the experiment yaml). Folders are
  `<output_root>/<run_name>` for one job and `<run_name>-<idx>` for a grid (suffix by job count),
  with `output_root` defaulting to `<experiment folder>/outputs`; the W&B run name is the folder
  name and the group is `run_name`. No timestamps: re-using a folder (e.g. continuing chosen grid
  jobs longer) only warns; changing `run_name` when a grid's shape changes is the user's
  responsibility. Hyperparameters are compared via W&B config and the saved configs, not names.
  This removed all of hydra2's label code, whose weakness was that Hydra can't tell a swept axis
  from any other override (a one-job run with an override was treated as a sweep).
- **Experiment file is Hydra's primary config** (`--config-dir <its folder> --config-name
  <stem>`); its defaults start with `- config` and `- dataset`. No searchpath, `@package
  _global_` or `+group=` selector. `dataset.yaml` (one per folder, not sweepable) is a Hydra
  config with `${data.*}` paths, replacing the template code and `dataset_dir`/`dataset_file`.
- **Every schema field is read by code.** hydra2 validated fields that nothing read
  (`lora.enabled`, `basemodel.dtype`, `bnb_4bit_compute_dtype`), so setting them silently did
  nothing. Fine-tuning modes (`train.finetune_mode: qlora|lora|full`) were ported from hydra2 as
  is; the `slurm:` block and `output_root` are read by `grid.py`.
- **W&B console capture is off.** Found while checking W&B calls: by default W&B uploads the
  run's stdout/stderr as `output.log`, which contained example text (a training example,
  decoded training batches) on hydra2 runs, and would on `main`'s `report_to="wandb"` runs too.
  `WANDB_CONSOLE=off` + `Settings(console="off")`; noted in CLAUDE.md.
- **Scope.** Only `sft_translation.py` uses Hydra; `inf_translation.py` and the other recipes keep
  their command-line arguments. Hydra inference from a saved job config is the planned next
  feature (09). Resume-from-checkpoint is deferred.

---

## 2026-10-18: Types-only schema, peft group, local data location, Hydra inference (09_hydra_detailed_reorg)

**Decision:** Each config key has exactly one home, and the code is split so training and
inference share model handling.

- **The schema holds types only, no default values.** Every value field in `hfmt/hydra_config.py`
  is `MISSING`; optional blocks and keys are `Optional[...] = None` (absent). Every default lives in
  exactly one yaml: `conf/config.yaml` (defaults list; project-wide `slurm`, `wandb`, `inf_name`),
  `conf/<group>/<option>.yaml` (`train`, `peft`, `decode`), or the experiment folder. 08 gave every
  field a default *and* repeated most of them in `conf/`, so there were two sources of truth, and a
  yaml quietly overrode a misleading dataclass default. Now a key that no yaml sets fails in
  `grid.py`. Cost: default values are read in `conf/`, not in the schema.
- **`peft` is both the group and the key.** `peft/{qlora_default,lora_default,full_finetune}` set
  `peft.mode` plus optional `peft.lora` / `peft.quant` blocks (a full fine-tune has neither);
  `check_config` rejects a mode that disagrees with the blocks. This replaces the `basemodel` and
  `lora` groups and `train.finetune_mode`. The alternative (keep old key paths, `@package
  _global_` option files writing into three nodes) was rejected because the group name would no
  longer match a key. `checkpoint` and `instruction` became top-level experiment keys;
  `load_in_4bit` was kept, in `peft.quant`. Sweep/override keys changed (`lora.lora_r` →
  `peft.lora.lora_r`); 08 saved configs no longer validate and were deleted, not migrated.
- **Data location is a per-folder, untracked `local_setting.sh`** (`export HFMT_DATA_ROOT=...`),
  sourced by `egs/launch.sh sft` after unsetting the variable; `dataset.yaml` reads
  `${oc.env:HFMT_DATA_ROOT}` and also holds `src`/`trg` (fixed per folder, the same on every
  machine, so tracked). A single machine-wide data variable and an untracked Hydra paths group were
  considered; the per-folder file keeps tracked yamls generic and switches automatically with the
  experiment folder. Only needed at launch: saved configs hold resolved paths. `data_root/src/trg`
  are optional, so a dataset yaml may list absolute paths instead.
- **Inference runs on a trained run folder's saved config**, one folder per call:
  `egs/launch.sh inf <run folder>/run_config.yaml [overrides]` → `grid.py --inf` checks it and
  saves `<run folder>/inf/<inf_name>/inf_config.yaml` (with `model_dir`), never touching
  `run_config.yaml`. Accepting experiments and sweeps too (mapping them back to run folders) was
  planned and dropped as too much code for a shell loop's worth of convenience. `inf_translation.py`
  moved to Hydra in place; its old argparse CLI (used by the zh-en and synth_lrl recipes) is gone.
  No W&B in inference. One launcher, `egs/launch.sh <sft|inf>`, replaces `egs/sft.sh`; job scripts
  are `egs/sft_job.sh` (was `egs/job.sh`) and `egs/inf_job.sh`.
- **Inference reproduces training's test predictions exactly** (checked on full, lora and qlora:
  0/2000 lines differ). `hfmt/mt_model.py` holds model loading, prompting, decoding and scoring for
  both. Two causes of mismatch were found and fixed: (1) training leaves Accelerate's bf16 autocast
  wrapper on `model.forward` (it patches the model object in place and the Trainer never removes
  it), so full fine-tune test decoding ran under autocast; `sft_translation.py` now calls
  `accelerator.unwrap_model(model, keep_fp32_wrapper=False)` after training, before test decoding
  (dev decoding during training stays bf16). PEFT models were unaffected because
  `PeftModel.generate` bypasses the patched forward. (2) `prepare_model_for_kbit_training` casts a
  qlora base's non-quantized parameters to fp32; inference now applies it too
  (`use_gradient_checkpointing=False`). The model is saved before test decoding so a crash there
  doesn't lose it.

---

## 2026-10-18: Base-model inference, per-file test sets, merging, dev-decoding interval (10_inference_more_options)

**Decision:** Inference, merging and training share one launcher, and the yaml you give it decides
what happens.

- **The yaml kind chooses the model for `launch.sh inf`.** A trained run's `run_config.yaml`
  decodes its `model/` into `<run folder>/inf/<inf_name>/`; an experiment yaml decodes its
  `checkpoint` as downloaded (`model_dir` unset) into `<output_root>/<run_name>/base_ckpt_inf/<inf_name>/`.
  A `peft=none` option was considered: more explicit in saved configs, but it needs its own
  precision rule and training would have to reject it. Instead the experiment's `peft` option sets
  the load precision. The separate `base_ckpt_inf/` folder means zero-shot output can't collide with
  a trained run's `inf/`, so nothing is refused. Sweep experiments and comma lists are refused.
- **Each test file is its own test set.** `data.test` is a `TestSet` (`trg` optional; a `null`
  entry = no reference). `mt_model.decode_test_sets` writes `test<i>.pred.trg` (i from 1, in list
  order, even for one file) and `scores.json` (paths + scores), and both training and inference
  call it, so inference still reproduces training exactly. Train/dev stay concatenated. Named
  test sets in `dataset.yaml` were considered; numbering kept the schema and the
  `'data.test.src=[...]'` override simple. Training now scores its test sets too and puts the
  numbers (no paths) in the W&B summary.
- **Merging goes through Slurm: `launch.sh merge <run>/run_config.yaml`** writes
  `<run>/model_merged/` (bf16 weights, tokenizer, `merge_info.yaml`). QLoRA merges onto the 4-bit
  base dequantized to bf16 (`dequantize()`), i.e. the weights the adapter was trained against,
  with a warning about the 4-bit rounding; merging into the original bf16 base gives better base
  weights but not the evaluated model. Measured: merged vs adapter within ~0.3 BLEU, 10-12% of
  lines differ (bf16 rounding). A second round is just an experiment with
  `checkpoint: <run>/model_merged`; nothing else changed. Training also saves the tokenizer in
  `model/`, so a full fine-tune's `model/` is a checkpoint as is.
- **One job script, `egs/job.sh <sft|inf|merge>`,** replaces `sft_job.sh`/`inf_job.sh` (and a third
  merge script): they differed only in the Python entry point.
- **`train.inference_callback_interval`** (N): the MT callback decodes dev + BLEU every N
  evaluations and at the last step (0 = never); early stopping runs every evaluation. With a
  BLEU-type `metric_for_best_model`, the Trainer checks that metric at *every* evaluation and raises
  if it's missing. So on skipped evaluations the callback runs early stopping first (it finds no
  value and leaves its patience counter alone), then sets a never-best placeholder (±inf) for the
  Trainer's best-checkpoint check. The placeholder never reaches W&B or `log_history`, because
  the Trainer logs eval metrics before callbacks run. Refusing BLEU + N > 1 was the alternative.
- **Saved configs aren't kept backward compatible.** A new required key (here
  `train.inference_callback_interval`) makes older `run_config.yaml` files fail `inf`/`merge`
  checks; the developer chose this over `Optional = None` keys or filling defaults on load.
- `grid.py` was split into `main_sft` / `main_inf` / `main_merge` with step comments, and
  `launch.sh` was commented block by block, since both had become hard to follow.

---

## 2026-10-18: LoRA target modules as explicit lists, short keys, model inspection (11_hydra_lora)

**Decision:** LoRA layers are named, not chosen from presets, and a script shows the names.

- **`peft.lora.target_modules` is a required list of exact module names** (no `null`, no
  `all-linear`). It replaces `lora_target` (`qv | all-linear | attention | mlp`), which
  `sft_translation.py` mapped to names. That mapping was Qwen/Llama-specific and wrong even there:
  `attention` listed `out_proj`, but Qwen's attention output is `o_proj`, so it silently trained only
  q/k/v; `mlp` left out `gate_proj`. PEFT's built-in per-architecture default was rejected as the
  fallback: it exists for qwen2 but not for cohere2 (tiny-aya). Names aren't checked at launch;
  PEFT fails at job start on a name that matches nothing, and training logs what it wrapped.
- **Short keys:** `peft.lora.{r,alpha,dropout,target_modules}` (were `lora_r`, `lora_alpha`,
  `lora_dropout`, `lora_target`); the block is already `peft.lora`. No backward compatibility:
  older run configs and W&B columns keep the old names.
- **`tools/inspect_model.py <checkpoint>`** builds the model on the meta device (no weights, login
  node, seconds) and prints module names, LoRA candidates, parameters, config, chat-template
  overhead and a training memory estimate. The estimate uses fixed, printed equations (weights by
  mode; gradients 4T and AdamW 8T for T trainable parameters; checkpointed activations; logits
  14·b·s·V; CUDA context separately), not fitted; on four A100 runs it was within ±15% of peak
  allocated memory. Peak *reserved* memory, logged before, was 2–4× allocated and grows to fill
  larger GPUs, so training now also logs peak allocated. For large-vocabulary models (tiny-aya,
  262k) the logits term dominates.

---

## 2026-10-19: Resuming a cancelled SFT run (12_resume_sft)

**Decision:** `launch.sh sft <run folder>/run_config.yaml` means *resume*, and nothing else.
It continues a run cancelled midway from its last checkpoint with TRL's `resume_from_checkpoint`,
in the same folder with the same config.

- **What it replaces.** Before this, the same command re-ran the saved config from scratch, with
  any overrides. Without `run_name=` it reused the folder: it overwrote `model/`, the test
  outputs and `run_config.yaml` itself, and started a second W&B run with the same name. With
  `run_name=` it made a new folder.
- **Why the re-run was dropped.** An experiment yaml + overrides does everything it did and more:
  a `run_config.yaml` can't swap config options (its defaults list is only the schema). The only
  thing the re-run gave was reproducing exactly the old values, and `git_rev` plus committed
  yamls cover that.
- **Only `slurm.*` may change** (a bigger GPU after an OOM, a longer time limit). Anything else
  would make the resumed run something other than the cancelled one.
- **The flag is an environment variable** (`job.sh sft_resume` sets `HFMT_SFT_RESUME=1`), not a
  config key. A key would land in `run_config.yaml` and W&B, and need a `+key` override on a
  saved config; resuming is how a job starts, not part of the experiment.
- **Checks before submitting.** `grid.py` refuses:
  - a finished run (`model/`);
  - a run with no checkpoint (TRL itself raises `No valid checkpoint found`, but only after the
    queue wait);
  - an incomplete newest checkpoint. Checkpoints aren't saved atomically, and
    `trainer_state.json` is the last file written.

  A changed `git_rev` only warns.
- **W&B continues the same run** (ID from `wandb/latest-run`, `resume="allow"`). Steps between
  the last checkpoint and the kill are trained and logged again.
- **TRL's defaults kept, to change as little as possible:**
  - callback state isn't restored (`restore_callback_states_from_checkpoint=False`), so the
    early-stopping patience counter restarts at 0. Turning it on would also need our callback's
    `__init__` to take the saved arguments;
  - TRL's `num_tokens` restarts at 0.

  A GPU check against an uninterrupted run matched in steps and LR, and was within noise in losses
  and test BLEU (61.08 vs 61.02).

---

## 2026-11-03: GPU startup race and requeue

**Decision:** `egs/job.sh` waits for its GPU before starting, every sbatch gets `--requeue
--open-mode=append`, and a requeued training task resumes from its last checkpoint.

- **Why.** GPUs in exclusive-process mode (one process per GPU) make tasks that start in the same
  second race for GPUs, and the losers exit at once with "CUDA-capable device(s) is/are busy or
  unavailable", so some tasks of a submission fail and others run. The recommended practice is `%N`
  throttling (already `slurm.array_parallelism`), arrays instead of sbatch loops (already one
  array per launch), and `--requeue`.
- **`--requeue` doesn't retry a failed exit here.** Slurm requeues a task only after a node
  failure, preemption or `scontrol requeue`, never because it exited with an error (it is not
  configured to requeue on exit codes). So the retry is in `job.sh`: up to 6 tries to create a CUDA
  context, 30 s apart, before Python (and the W&B run) starts. `%N` only caps the burst: the first
  N tasks still start together.
- **Requeue resumes training.** A requeued task runs `job.sh` from the top; for `sft` with
  `SLURM_RESTART_COUNT > 0` and a `checkpoint-*` it switches to `sft_resume` (feature 12) instead
  of overwriting its folder from step 0 with a second W&B run. `--open-mode=append` keeps the
  first attempt's log (same job id, same `slurm-*.out`).
- **Hard-coded, not `slurm:` keys:** a new required key would break resume/inf/merge for every
  saved `run_config.yaml`. `tools/run_status.py`'s rerun command adds the same two flags.

---

## 2026-11-05: Full fine-tunes: the best checkpoint is reloaded through `from_pretrained`

**Decision:** for `peft.mode=full`, `sft_translation.py` loads the checkpoint the Trainer is
supposed to reload in place — the best one at the end of training, the last one on resume —
with `mt_model.load_checkpoint_weights`: `from_pretrained` on the CPU, then a strict
`load_state_dict` into the training model. The Trainer's own reload is left in place (harmless).

- **Why.** The Trainer reloads full-model checkpoints with a plain non-strict `load_state_dict`
  (`_load_best_model`, `_load_from_checkpoint`), but `save_pretrained` writes the Hub's weight
  names, which `from_pretrained` converts to the module names on load. For Qwen3.5 they differ
  (`model.language_model.*` vs `model.*`), so no key matched, nothing loaded, and the Trainer only
  warned ("There were missing keys in the checkpoint model loaded"). Every Qwen3.5 full run's
  `model/`, `scores.json` (training's and `launch.sh inf`'s) and `training_info.yaml`
  (`model_global_step`) are therefore the last step, not the best checkpoint: dev
  `metric_for_best_model` 24.74 instead of 34.33 for `sft_full_qwen-3.5-2b_zh-en`, 62.63 instead
  of 65.01 for `sft_full_qwen-3.5-4b_sweep-1`. A resumed full run would have continued from the
  base weights with the checkpoint's optimizer state. LoRA/QLoRA runs reload through PEFT's
  `load_adapter` and were correct. Gemma 3 has a rename too (`language_model.model.*`); a run is
  affected iff its Slurm log has the warning.
- **Why this fix.** `from_pretrained` is the public path that applies the conversion (it is how
  `inf_translation.py` loads `model/` already); `strict=True` turns the next silent mismatch into
  a failed run. Saving checkpoints under module names instead would need Transformers' private
  `_weight_conversions`. Cost: one CPU copy of the weights, as the Trainer's reload already needs.
- **Old runs:** `checkpoint-<best_global_step>/` is kept by `save_total_limit`, so an affected run
  can be decoded without retraining (`checkpoint:` set to that folder, `peft=full_finetune`).
- **Guards added the same day.** (1) Training decodes dev once more with the saved `model/` and
  records `model_dev_scores` and `model_dev_check` in `training_info.yaml`: the measured dev BLEU
  against the MT callback's at `model_global_step` (`DEV_SCORE_TOLERANCE` 1 BLEU: the callback decodes
  under Accelerate's autocast wrapper, the final decode unwrapped). Before, the file asserted what the
  Trainer should have done. A mismatch is logged as an error, not fatal: the weights and scores are
  still written, and the check is what makes them suspect. (2) `tools/run_status.py` searches whole
  training logs for known-bad library warnings (`BAD_WARNINGS`: this reload warning, TRL's prompt-prefix
  mismatch) and prints "Checks failed" with any `MISMATCH`. This bug would have been visible the day it
  first ran.
- **Correction (2026-11-06).** The Trainer's reload is still there, so it still warns in every Qwen3.5
  full run after the fix too: the warning marks an affected run only when `training_info.yaml` has no
  `model_dev_check` (runs from before this fix). The first `_short` runs were flagged by the log scan
  although their dev check was `ok`. `run_status.py` now reports the reload warning only for runs
  without a dev check; for the others, the measured check decides.

---

## 2026-11-07: Prompt rendering: one builder, hardcoded `enable_thinking=False` (13_port_without_translategemma)

**Decision:** causal-LM prompts are built by the single `mt_model.format_input_prompt` (instruction,
newline, source) and every `apply_chat_template` call passes `mt_model.CHAT_TEMPLATE_KWARGS`
(`enable_thinking=False`). The `hydra3-translategemma` branch had a `prompt_format` registry
(`chat` | `translategemma`), a chat-template patch for TranslateGemma and a vision-tower freeze
(`NON_LANGUAGE_MODULES`); this branch ports that work without TranslateGemma support
(`features/13_port_without_translategemma/porting.md`), so none of that exists here.

- **Why hardcode.** Of the two model families kept (Qwen3.5 and Gemma 3), only Qwen3.5 needs anything
  from a prompt format: Qwen3.5-4B's template thinks by default (generation prompt `<think>\n`, so the model writes its
  reasoning as the translation; 0.8B/2B default to off). Off, the generation prompt ends with the empty
  `<think>\n\n</think>\n\n` the training targets start with. Templates without the variable (Gemma)
  ignore it. One constant used by training (TRL's per-example `chat_template_kwargs` column), dev/test
  decoding, inference and `tools/inspect_model.py` keeps every render identical; a schema key that
  every experiment must set bought nothing once there is one format.
- **`processing_class=tokenizer` for the `SFTTrainer`** (kept). Qwen3.5 and Gemma 3 ship a VL
  processor, so without this TRL loads `AutoProcessor`, switches to its multimodal path and rewrites
  the assistant turn until the template rejects it. Passing the tokenizer keeps TRL on the text path
  and reuses one tokenizer object for training and our decoding.
- **No vision-tower freeze.** Qwen3.5 under `AutoModelForCausalLM` has no vision modules. Gemma 3 4B
  (a kept LoRA experiment) does: its tower's `q/k/v_proj` are wrapped by the all-linear adapter too,
  but with no `pixel_values` the tower never runs, so those adapters get no gradient and the result is
  unchanged; only the trainable-parameter count and adapter size grow. Bring back `3743e17` + `9c66f6a`
  from `hydra3-translategemma` if that matters.
- **Saved configs.** `run_config.yaml`s written on `hydra3-translategemma` carry `prompt_format:`,
  which this schema has no key for. Since 2026-11-10 `grid.py` refuses such a file (`RETIRED_KEYS`)
  and prints the `sed` command that removes the line; before, Hydra failed with a struct error.

---

## 2026-11-10: Docs and config consolidation

**Decision:** a review of `decisions.md`, `CLAUDE.md` and `README.md` against the repo found stale
statements and duplication; the fixes below change how the docs are organized and which train
schedule is the default. No training behaviour changes: every committed experiment except the two
tutorial files composes to a byte-identical `run_config.yaml` before and after (checked over all 28
experiments).

- **Each document has one job.** README says *how* to use the workflow and is the only place that
  lists the `conf/` options; `CLAUDE.md` holds the rules (confidentiality, workflow, conventions) with
  one-line summaries that point into README; `decisions.md` says *why*. The Inference, Merging, Dev
  decoding and Resume bullets of `CLAUDE.md` had become near-copies of README sections and were cut
  to summaries. The superseded hydra2 history (features 00–07, 245 lines) moved to
  `decisions_hydra2.md`, and the remaining entries got dated headings in chronological order.
- **Train schedules stay self-contained.** Having each of the five `conf/train` schedules inherit a
  shared `common.yaml` (Hydra `defaults: [common, _self_]`) was tried and reverted the same day: the
  developer prefers every schedule to be readable as one complete file, the same choice feature 07
  made for experiment files. The cost is ~25 repeated lines per schedule.
- **`shortrun` is the project default; `conf/train/default.yaml` is gone.** The old default (200
  steps, reduce-on-plateau, `eval_loss` selection, batch 16) was reached only by the two tutorial
  experiments and matched nothing anyone ran. The tutorial files (`sft_lora.yaml`,
  `sft_lora.sweep.yaml`) now fine-tune Qwen3.5-2B (was Qwen2.5-1.5B-Instruct, a model no other
  experiment uses) with the default schedule.
- **`grid.py` refuses saved configs with retired keys** (`RETIRED_KEYS`, currently `prompt_format`)
  and prints the exact `sed` fix, instead of Hydra's struct error. Removing the line by hand was the
  documented procedure; the check is cheap and the message is the procedure.
- **Smaller fixes.** `install/install_hf_custom.sh` now installs `hydra-core==1.3.7` (it never
  installed Hydra; `check_versions.py` reports it). `wandb.entity` is `Optional[str] = None` like
  every other optional key (it was `MISSING` with `null` in `config.yaml`). The stale
  `egs/synth_lrl/*/inf*.sh` recipes (the argparse CLI they called was removed in feature 09) are
  deleted. The zh-en `local_setting.sh.example` uses a placeholder path like fr-en's. Only
  `analysis/run_sacrebleu.py` is tracked under `analysis/`; the dated analysis folders and the W&B
  export script stay local. References to a non-existent `overview.md` are gone.
  Feature numbers in code comments are kept on purpose: they link code to `features/` directories.
