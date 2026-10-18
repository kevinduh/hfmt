# 09_hydra_detailed_reorg — todo

Generated from `plan.md` (v3, final). Groups follow the plan's phases; items within a group are
ordered by dependency. One commit per phase, subject `09_hydra_detailed_reorg-<phase>: <summary>`;
Claude presents the message, the developer commits. Each phase ends with its "Done when" checks
before the next phase starts.

Ground rules for every phase:
- **The Rule:** `hydra_config.py` holds types only. Value fields are `MISSING`; optional blocks/keys
  are `Optional[...] = None`. Every default lives in exactly one yaml.
- Every schema field is read by code (training/inference code, or `grid.py` for `slurm`,
  `output_root`, `inf_name`), or, for `data.data_root/src/trg`, by `dataset.yaml`'s interpolations.
- Don't touch the untracked leftovers other than `conf/hydra/` (Q8): `data/`, `egs/experiment/`,
  `egs/mmtc/fr-en/outputs/`. Stage files by name, never `git add conf/` or `git add egs/`.
- No dataset content to W&B (CLAUDE.md). Inference makes no W&B calls at all.

## A. Config reorganization (schema, conf/, experiments, data location)

### A1. Baseline before changing anything
- [x] Save the 08 composed config for comparison: `--cfg job --resolve` for `sft_lora` and
      `sft_lora.sweep` (scratchpad, not the repo).

### A2. Schema (`hfmt/hydra_config.py`)
- [x] Remove `BaseModelConfig`; add top-level `checkpoint: str = MISSING`.
- [x] `QuantConfig`: `load_in_4bit`, `bnb_4bit_use_double_quant`, `bnb_4bit_quant_type` (all `MISSING`).
- [x] `LoraConfig`: `lora_r`, `lora_alpha`, `lora_dropout`, `lora_target` (all `MISSING`).
- [x] `PeftConfig`: `mode: str = MISSING`, `lora: Optional[LoraConfig] = None`,
      `quant: Optional[QuantConfig] = None`.
- [x] `TrainConfig`: drop `finetune_mode`; every other field `MISSING`.
- [x] `DataConfig`: drop `instruction`; `train`, `dev` `MISSING`; `data_root`, `src`, `trg` and
      `test` `Optional = None` (a dataset yaml may list absolute paths instead).
- [x] Top-level `instruction: str = MISSING`, `inf_name: str = MISSING`,
      `model_dir: Optional[str] = None`; `output_root: Optional[str] = None` unchanged.
- [x] `DecodeConfig`, `WandbConfig`, `SlurmConfig`: every field `MISSING` (`wandb.entity` stays
      `Optional[str]`, set to `null` in B).
- [x] `check_config(cfg)`: `peft.mode` in `full|lora|qlora`; `lora`/`qlora` need `peft.lora`;
      `qlora` needs `peft.quant`; `full` has neither, `lora` has no `quant`. Clear one-line errors.
- [x] Update module/class docstrings (types only; where defaults live).

### A3. `conf/`
- [x] `conf/peft/qlora_default.yaml` (mode, lora, quant incl. `load_in_4bit: true`),
      `conf/peft/lora_default.yaml` (mode, lora), `conf/peft/full_finetune.yaml` (mode only).
- [x] `git mv conf/train/qlora_default.yaml conf/train/default.yaml`; remove `finetune_mode`.
- [x] `conf/decode/default.yaml`: unchanged values (header comment only).
- [x] `conf/config.yaml`: defaults `[base_config, train: default, peft: qlora_default,
      decode: default, _self_]`; `slurm:` (unchanged); `wandb: {enabled, project, entity: null,
      tags: []}`; `inf_name: test`; `hydra.job.chdir: false`.
- [x] Delete `conf/basemodel/` and `conf/lora/` (`git rm`).
- [x] Delete untracked `conf/hydra/` (Q8; 3 files, not recoverable from git).

### A4. Data location (`local_setting.sh`)
- [x] `.gitignore`: add `egs/**/local_setting.sh`.
- [x] `egs/mmtc/fr-en/local_setting.sh.example` (tracked, one line plus a comment).
- [x] `egs/mmtc/fr-en/local_setting.sh` for this machine (untracked), pointing at `$HFMT_ROOT/data/fr-en`.
- [x] `dataset.yaml`: add `data_root: ${oc.env:HFMT_DATA_ROOT}`, `src: fr`, `trg: en`; header comment.
- [x] `egs/sft.sh`: `unset HFMT_DATA_ROOT`, then source `<experiment folder>/local_setting.sh` if it
      exists, before `grid.py`. Header comment.
- [x] `grid.py`: when resolving fails on a missing `HFMT_DATA_ROOT`, exit with a hint to create
      `<experiment folder>/local_setting.sh` (see `.example`).

### A5. Experiments
- [x] `sft_lora.yaml`: drop `override basemodel`/`override train`; add `override peft: qlora_default`,
      `checkpoint`, top-level `instruction`; remove the `data:` block; header says to source
      `local_setting.sh` before a direct debugging run.
- [x] `sft_lora.sweep.yaml`: same, axis `peft.lora.lora_r: 8,16`.

### A6. Code using the renamed keys
- [x] `sft_translation.py`: `cfg.checkpoint`, `cfg.peft.mode`, `cfg.peft.quant.*`, `cfg.peft.lora.*`,
      `cfg.instruction`; call `check_config(cfg)` at the start of `main`.
- [x] `grid.py`: call `check_config` on every composed job (before writing anything); a hint when an
      override writes into an unset block (`peft=full_finetune peft.lora.lora_r=16`).
- [x] grep for remaining `basemodel`, `finetune_mode`, `data.instruction`, `cfg.lora` in tracked
      code and scripts (`tools/`, `egs/*.sh`). Only README/CLAUDE.md remain (phase D).

### A7. Tests (`hfmt/test_grid.py`)
- [x] Set a dummy absolute `HFMT_DATA_ROOT`; switch `lora.lora_r` → `peft.lora.lora_r`.
- [x] Each `peft` option composes (`peft=full_finetune|lora_default|qlora_default`).
- [x] Bad combinations fail before writing (e.g. `peft=full_finetune peft.lora.lora_r=16`).
- [x] Unset `HFMT_DATA_ROOT` fails with the hint.
- [x] A key missing from every yaml fails (MISSING check). (Note: a `~key` deletion override removes
      the key instead of leaving it unset, so the MISSING check can't see it.)

### A8. Phase A checks
- [x] `--cfg job --resolve` for `sft_lora` and `sft_lora.sweep` vs the A1 baseline: only the
      intended renames/moves differ (diff reviewed).
- [x] `python hfmt/test_grid.py` passes.
- [x] `DRYRUN=1 bash egs/sft.sh egs/mmtc/fr-en/sft_lora.yaml` and the sweep produce the expected
      folders and saved configs (scratch `output_root`).
- [x] Present the commit message.

## B. Shared model code and Hydra inference

### B1. `hfmt/mt_model.py`
- [ ] Move `format_input_prompt` and `inference_on_eval_data` from `sft_translation.py`.
- [ ] `load_base_model(cfg, device)`: the qlora/lora/full branches now in `sft_translation.py`
      (dtype, 4-bit config, device placement), returning the model and `use_bf16`.
- [ ] `load_trained_model(cfg, model_dir, device)`: `full` loads `model_dir` with the training dtype;
      `lora`/`qlora` load the base via `load_base_model` and add `PeftModel.from_pretrained`.
- [ ] `score(preds, refs)`: BLEU (flores200, as in training), chrF, TER; used by the training
      callback and by inference.
- [ ] Heavy imports inside functions (config-only use stays fast).

### B2. `sft_translation.py`
- [ ] Use `mt_model` (prompt, decode, `load_base_model`, `score`); behavior unchanged.

### B3. `inf_translation.py` on Hydra
- [ ] Replace the argparse CLI (Q4d) with `@hydra.main(config_path="../conf", config_name="config")`;
      `register_configs()` at module level; `check_config(cfg)`.
- [ ] Fail clearly if `model_dir` is unset/missing or `data.test` is `None`.
- [ ] Tokenizer from `cfg.checkpoint`; decode `data.test.src` with `cfg.instruction` and `decode.*`;
      write `<run dir>/pred.trg`.
- [ ] If `data.test.trg` is given, write aggregate scores to `<run dir>/scores.json` (Q7). Logs stay local.
- [ ] No W&B import or call (confidentiality).

### B4. Phase B checks (GPU, via Slurm)
- [ ] 20-step SFT runs for `peft=full_finetune`, `lora_default` and `qlora_default` (scratch
      `output_root`, `train.max_steps=20`) complete.
- [ ] Direct inference on each run folder (`--config-dir <run> --config-name run_config
      model_dir=<run>/model hydra.run.dir=<scratch>`) reproduces that run's `eval.pred.trg`.
- [ ] `test_grid.py` still passes. Present the commit message.

## C. Inference launcher

### C1. `egs/job.sh`
- [ ] Take the module and the saved-config name as the first two arguments, then the job folders
      (`job.sh hfmt.sft_translation run_config <dirs...>`).
- [ ] `sft.sh` (sbatch and `LOCAL=1`) passes them.

### C2. `grid.py --inf`
- [ ] Refactor the job-folder naming so training and inference share it.
- [ ] Refuse command-line comma lists in inference mode.
- [ ] Map the input (experiment, sweep, or `run_config.yaml`) to its run folders; require each
      folder's `run_config.yaml` and `model/`.
- [ ] Per folder: compose `<run>/run_config` + overrides + `model_dir=<run>/model`; `check_config`;
      resolve; MISSING check; all before writing anything.
- [ ] Warn when a folder's saved config disagrees with the expanded experiment on a grid axis.
- [ ] Save `<run>/inf/<inf_name>/inf_config.yaml` (+ `git_rev`); warn if the folder exists; never
      write `run_config.yaml`.
- [ ] Print `JOBS` (inference folders) and `SBATCH_ARGS` (job name `<run_name>-inf`, logs in the
      inference folders).

### C3. `egs/inf.sh`
- [ ] Same path resolution, `local_setting.sh` sourcing, `LOCAL=1`, `DRYRUN=1` as `sft.sh`; share
      code with `sft.sh` rather than duplicating it.
- [ ] Calls `grid.py --inf`, then `job.sh hfmt.inf_translation inf_config <dirs...>`.

### C4. Tests and checks
- [ ] `test_grid.py`: inference on a single run and a sweep (fake `model/` folders) writes
      `inf/<inf_name>/inf_config.yaml`; `run_config.yaml` unchanged; comma list, missing `model/`
      and a key typo fail before writing.
- [ ] `LOCAL=1 bash egs/inf.sh` on a B4 run folder (GPU node or Slurm) matches B4's predictions.
- [ ] `DRYRUN=1` on a sweep prints one array over all inference folders.
- [ ] Present the commit message.

## D. Docs
- [ ] CLAUDE.md: structure (`conf/` groups `train`/`peft`/`decode`, `inf.sh`, `mt_model.py`),
      the Rule, `local_setting.sh`, renamed keys, inference.
- [ ] README "Running SFT experiments": new keys and groups, `local_setting.sh` setup, inference.
- [ ] `decisions.md`: 09 entry (the Rule, H2 `peft` group, `local_setting.sh`, inference from saved
      run configs).
- [ ] Check `overview.md` for stale references.
- [ ] Measure net added lines (not strict). Present the commit message.

## E. Validation (developer)
- [ ] Short Slurm sweep with `peft: lora_default,qlora_default`, then `inf.sh` on it.
- [ ] Present the commit message.
