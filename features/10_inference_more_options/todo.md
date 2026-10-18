# 10_inference_more_options — todo

Generated from `plan.md` (v3, final). Groups follow the plan's phases; items within a group are
ordered by dependency. Commit once per phase, with subject `10_inference_more_options-<phase>: <summary>`.
Claude presents the message and the developer commits. Each phase ends with its "Done when" checks
before the next phase starts.

Ground rules for every phase:
- **The Rule:** `hydra_config.py` holds types only. Value fields are `MISSING`; optional keys are
  `Optional[...] = None`. Every default lives in exactly one yaml. A new `train.*` key goes into
  **every** `conf/train/*.yaml` (`default`, `longrun`).
- **No dataset content to W&B** (CLAUDE.md). Only aggregate scores go there, never test-set paths
  or text. Inference and merging make no W&B calls.
- **Don't touch the untracked leftovers** (`data/`, `egs/experiment/`, `egs/mmtc/fr-en/old_outputs/`,
  `egs/synth_lrl/tuk-eng/*`). Stage files by name.
- **Comment as you go (request 4).** Any code added to `grid.py`/`launch.sh` gets its explaining
  comment in the same phase.
- **Tests:** `HFMT_ROOT=$(pwd) python hfmt/test_grid.py` (no pytest). GPU test runs go to the
  gitignored `egs/mmtc/fr-en/outputs/` under `run_name=p10<phase>…`, and are deleted once confirmed.

## A. Dev-decoding interval (request 5)

### A1. Config
- [x] Schema: `TrainConfig.inference_callback_interval: int = MISSING`, with a docstring line
      (N ≥ 1: decode dev every N evaluations and at the last step; 0: never).
- [x] `conf/train/default.yaml` and `conf/train/longrun.yaml`: `inference_callback_interval: 1`,
      with a short comment.
- [x] `check_config`: `inference_callback_interval >= 0`.

### A2. Callback (`sft_translation.make_mt_callback`)
- [x] Pass the interval in. Add `_decode_now(args, state)`:
      `N > 0 and (global_step % (N * args.eval_steps) == 0 or global_step >= state.max_steps)`.
- [x] On a decode step, keep today's body: decode, score, add `eval_bleu/chrf/ter`, `wandb.log`
      with `train/global_step`. Otherwise log "dev decoding skipped at step S (interval N)".
- [x] Call `super().on_evaluate(...)` at every evaluation, **before** the placeholder.
- [x] After `super()`, on a skipped step where `metric_for_best_model` is in
      `("eval_bleu", "eval_chrf", "eval_ter")`, set `metrics[best]` to −inf (greater is better) or
      +inf. The comment explains the Trainer ordering (`trainer.py:2083-2085`, `:3127`).

### A3. Tests
- [x] `test_grid.py`: the saved `run_config.yaml` has `train.inference_callback_interval: 1`;
      `train.inference_callback_interval=-1` fails; `train=longrun` composes.

### Done when
- [x] Tests pass.
- [x] GPU, `eval_loss` as best: `train.max_steps=45 train.eval_steps=10 train.inference_callback_interval=2`.
      - `dev.step_*.pred` exists only for 20, 40 and 45.
      - W&B history has `eval/bleu` only at those steps and `eval/loss` at every evaluation.
- [x] GPU, same with `train.metric_for_best_model=eval_bleu train.greater_is_better=true`:
      - no KeyError;
      - `best_model_checkpoint` is a decode step;
      - no ±inf in W&B history or in `trainer_state.json`'s `log_history`;
      - the patience counter doesn't move on skipped steps (log).
- [x] `todo.md` A ticked; commit message presented.

## B. Several test sets (request 2)

### B1. Schema and checks
- [x] Add `TestSet` dataclass: `src: List[str] = MISSING`, `trg: Optional[List[Optional[str]]] = None`;
      `DataConfig.test: Optional[TestSet] = None`.
- [x] `check_config`: if `test.trg` is given, `len(trg) == len(src)`, with a clear message.
- [x] Check that `'data.test.trg=[a,null]'` composes (OmegaConf `List[Optional[str]]`).
- [x] Update the `dataset.yaml` header comment (several test files, one output each, `null` = no reference).

### B2. Shared helper (`hfmt/mt_model.py`)
- [x] `decode_test_sets(cfg, tokenizer, model, outdir, device)`. For each `i` (from 1):
      load `test.src[i-1]`, decode with `inference_on_eval_data` into `<outdir>/test<i>.pred.trg`,
      and score it if it has a reference.
- [x] Write `<outdir>/scores.json`: `{"test<i>": {"src", "trg", ["bleu", "chrf", "ter"]}}`.
      Return the scores dict.

### B3. Callers
- [x] `sft_translation.py` test step: replace the single `eval.pred.trg` decode with
      `decode_test_sets(...)` (still after `unwrap_model` and the model save).
- [x] Training: if a W&B run is active, put each scored set's numbers in `wandb.run.summary` as
      `test/test<i>/{bleu,chrf,ter}`. Numbers only.
- [x] `inf_translation.py`: replace the single decode/score with `decode_test_sets(...)`. Update the
      docstring (outputs).
- [x] Update stale `pred.trg`/`eval.pred.trg` mentions in `launch.sh` comments.

### B4. Tests
- [x] `test_grid.py`: two test files with a `null` reference compose; mismatched `trg` length fails.

### Done when
- [x] Tests pass.
- [x] GPU: 20-step lora and qlora runs with `data.test.src=[test, valid]`, `data.test.trg=[test_ref, null]`.
      - Training writes `test1.pred.trg`, `test2.pred.trg` and `scores.json` (test2 unscored).
      - The W&B summary has `test/test1/*` only.
- [x] `launch.sh inf` on both runs reproduces both `.pred.trg` files exactly (0 differing lines) with the same `scores.json`.
- [x] `todo.md` B ticked; commit message presented.

## C. Base-model inference (request 1)

### C1. `grid.py --inf`
- [x] `main_inf`: branch on the input name.
  - **`run_config`:** unchanged path; `model_dir=<run>/model`, output `<run>/inf/<inf_name>/`.
  - **Anything else (an experiment):** refuse a sweep (`hydra.sweeper.params` set) and comma lists;
    compose and check without `model_dir`; output
    `<output_root or <exp dir>/outputs>/<run_name>/base_ckpt_inf/<inf_name>/`; job name `<run_name>-base-inf`.
- [x] Numbered step comments in `main_inf`; module docstring lists the flows.

### C2. Launcher and inference
- [x] `launch.sh`: unset `HFMT_DATA_ROOT` and source `local_setting.sh` for every kind, with a comment
      (a run folder has none, so trained runs are unaffected).
- [x] `inf_translation.py`: `model_dir` unset → `load_base_model(cfg, device)` + `eval()`; a set but
      missing `model_dir` still fails. Docstring: the two modes.

### C3. Tests
- [x] `test_grid.py`:
  - `--inf` on an experiment saves `base_ckpt_inf/test/inf_config.yaml` with `model_dir: null`;
  - `--inf` on a sweep experiment fails;
  - `--inf` on an experiment with a comma list fails;
  - the existing `run_config` tests still pass.

### Done when
- [x] Tests pass. `DRYRUN=1` on an experiment prints the expected folder and job name.
- [x] GPU: zero-shot `launch.sh inf egs/mmtc/fr-en/sft_lora.yaml run_name=p10c peft=lora_default`.
      `base_ckpt_inf/test/` has `test1.pred.trg` and `scores.json`; the log shows bf16 and no adapter.
- [x] `todo.md` C ticked; commit message presented.

## D. Merging and a second round (request 3)

### D1. One job script (Q10)
- [ ] `egs/job.sh <sft|inf|merge> <folders…>`: shift to folder `$SLURM_ARRAY_TASK_ID`, then by kind:
  - `sft` → `hfmt.sft_translation --config-name run_config`;
  - `inf` → `hfmt.inf_translation --config-name inf_config`;
  - `merge` → `hfmt.merge_model <folder>`.

  Comment each case.
- [ ] Remove `egs/sft_job.sh` and `egs/inf_job.sh`. `launch.sh` calls `egs/job.sh "$kind"` (sbatch
      and `LOCAL=1`). Update `tools/run_on_free_gpu.sh` and the comment in `sft_translation.py:76`.

### D2. Training saves the tokenizer
- [ ] `sft_translation.py`: `tokenizer.save_pretrained(<run>/model)` next to `model.save_pretrained`.

### D3. Merge code
- [ ] `mt_model.merge_trained_model(cfg, model_dir, out_dir)`:
  - **lora:** bf16 base + `PeftModel.from_pretrained(...).merge_and_unload()`;
  - **qlora:** 4-bit base → `dequantize()` to bf16 → merge, and log a warning
    (4-bit rounding of the base weights);
  - **full:** raise.

  Save the weights and the tokenizer (from `cfg.checkpoint`).
- [ ] First check that `dequantize()` works on a bnb 4-bit model in this Transformers (5.8). If not,
      stop and bring options back to the developer.
- [ ] `hfmt/merge_model.py <run folder>`: `OmegaConf.load(run_config.yaml)`, warn if `model_merged/`
      exists, merge to `<run>/model_merged/`, write `merge_info.yaml` (source run folder, `checkpoint`,
      `peft.mode`, `git_rev`, the qlora warning if any). Device: cuda if available.

### D4. `grid.py --merge` and `launch.sh merge`
- [ ] `grid.py --merge <run>/run_config.yaml`: requires `run_config`, no comma lists, `model/` exists,
      `peft.mode != full`; writes no config; prints `JOBS=(<run>)` and sbatch arguments from the
      run's `slurm:` block, job name `<run_name>-merge`, log `<run>/merge-slurm-%A.out`. Commented
      like `main_inf`.
- [ ] `launch.sh`: accept `merge` as a kind; usage header lists all three kinds with an example each.

### D5. Tests
- [ ] `test_grid.py`:
  - `--merge` prints the run folder and job name;
  - it fails without `model/`, on a full run, and on an experiment yaml;
  - `--inf` and `sft` still pass.

### Done when
- [ ] Tests pass.
- [ ] GPU: `launch.sh merge` on a short lora run and a short qlora run (Slurm). Each gets `model_merged/`
      with weights, tokenizer and `merge_info.yaml`; the qlora run logs its warning.
- [ ] Base-model inference (C) on each `model_merged` (an experiment with `checkpoint=<model_merged>`)
      gives BLEU very close to that run's `inf/test` scores. Record both.
- [ ] Round 2: 20-step `launch.sh sft` with `checkpoint=<model_merged>`, then `launch.sh inf` on it;
      inference reproduces its `test1.pred.trg`.
- [ ] `todo.md` D ticked; commit message presented.

## E. Readability and docs (request 4)

- [ ] Read `grid.py` top to bottom.
  - The module docstring covers the four flows (sft grid, inf on a run, inf on an experiment, merge),
    what each writes, and the `JOBS=(…)`/`SBATCH_ARGS=(…)` stdout contract with an example.
  - Each helper has a one-line purpose/caller comment.
- [ ] Read `launch.sh` and `egs/job.sh` top to bottom: one comment per block; usage header current.
- [ ] CLAUDE.md:
  - structure (`job.sh`, `merge_model.py`);
  - the Inference bullet (two modes, `test<i>.pred.trg`, `scores.json`, `base_ckpt_inf/`);
  - a new merging bullet;
  - `inference_callback_interval`.
- [ ] README:
  - Inference (base-model mode, several test sets);
  - a new "Merging and a second round" section;
  - the interval setting in Training.
- [ ] `decisions.md`: feature 10 entry (yaml kind chooses the model, `base_ckpt_inf/`, numbered test
      outputs, qlora dequantize merge, one job script, the interval placeholder).
- [ ] Line counts since 27b8cc3 (code/config/tests vs docs). Tests pass.
- [ ] `todo.md` E ticked; commit message presented.

## F. Developer validation

- [ ] A real multi-test-set run with `inference_callback_interval > 1`.
- [ ] A zero-shot baseline of a downloaded model on the same test sets.
- [ ] One merge plus a round-2 SFT, then inference on both rounds.
