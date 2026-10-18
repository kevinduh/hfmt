# 08_hydra_redo — todo

Generated from `plan.md` (draft v1, all questions decided). Groups follow the plan's phases;
items within a group are ordered by dependency. One commit per phase, subject
`08_hydra_redo-<phase>: <summary>`; Claude presents the message, the developer commits.
Each phase ends with its "Done when" checks before the next phase starts.

Ground rules for every phase:
- Don't touch the untracked hydra2 leftovers (Q11): `conf/hydra/launcher/slurm_{h100,v100}.yaml`,
  `egs/experiment/`, `egs/mmtc/fr-en/sft1_lora.yaml.old`, the old `egs/mmtc/fr-en/outputs/`, `data/`.
  Stage new files by name, never `git add conf/` or `git add egs/`.
- Every schema field must be read by code (training code, or `grid.py`/`sft.sh` for `slurm:` and
  `output_root`) (lesson 3).
- No dataset content to W&B (CLAUDE.md). Check every W&B call before running it.

## A. Config, schema and training code (local single run)

### A1. Repo hygiene (before any run)
- [x] Add `egs/**/outputs/` and `outputs/` to `.gitignore` (prediction dumps are data-derived, R4).
- [x] Add empty `hfmt/__init__.py` (Q6 #6).

### A2. Defaults decision (R5) — ask the developer
- [x] Present `main`'s argparse defaults vs `egs/mmtc/zh-en/sft1.sh` recipe values vs hydra2's
      `conf/train/qlora_default.yaml` (lr, max_steps, eval_steps, scheduler, warmup, seed, batch
      size, early-stopping) and get the developer's choice for the `train` component defaults.

### A3. Schema (`hfmt/hydra_config.py`)
- [x] `BaseModelConfig`: `checkpoint`, `load_in_4bit`, `bnb_4bit_use_double_quant`,
      `bnb_4bit_quant_type` (no `dtype`, `bnb_4bit_compute_dtype`, `pretrain`).
- [x] `LoraConfig`: `lora_r`, `lora_alpha`, `lora_dropout`, `lora_target` (no `enabled`).
- [x] `DataSplit` (`src`, `trg`: `List[str]`) and `DataConfig`: `data_root`, `src`, `trg`,
      `instruction`, `train`, `dev`, `test` (`Optional[DataSplit]`). No `dataset_dir`/`dataset_file`.
- [x] `TrainConfig`: `finetune_mode` + trainer fields actually passed to `SFTConfig`, plus the
      early-stopping/best-model fields (Q6 #3). Cross-check each field against the code.
- [x] `DecodeConfig`: `max_length`, `max_new_tokens`, `num_beams`, `do_sample`, `batch_size` (Q6 #2).
- [x] `WandbConfig`: `enabled`, `project`, `entity`, `tags` (no `group`/`name`: derived, Q2).
- [x] `SlurmConfig`: `partition`, `gres`, `time`, `mem_gb`, `cpus`, `array_parallelism`
      (read by `grid.py`, phase B).
- [x] Top-level `Config`: `run_name: str = MISSING`, `output_root: Optional[str] = None`,
      `basemodel`, `lora`, `data`, `train`, `decode`, `wandb`, `slurm`.
- [x] `register_configs()` storing `base_config`; module importable without torch.

### A4. `conf/` components
- [x] `conf/config.yaml`: `defaults: [base_config, basemodel: …, lora: default, train: …,
      decode: default, _self_]`; `slurm:` defaults (partition gpu, gres gpu:a100:1, time 4h,
      mem 25G, cpus 1, array_parallelism 4, per hydra2's tuned values); `hydra.job.chdir: false`.
      No `hydra.searchpath`, no `hydra.run.dir` (Q13).
- [x] `conf/basemodel/qwen2.5-{0.5b,1.5b,3b,7b}.yaml` (checkpoint ids; quantization defaults).
- [x] `conf/lora/default.yaml`.
- [x] `conf/train/qlora_default.yaml` with the A2 defaults and `finetune_mode: qlora`.
- [x] `conf/decode/default.yaml`.

### A5. `hfmt/sft_translation.py` on Hydra
- [x] Replace argparse with `@hydra.main(config_path="../conf", config_name="config",
      version_base=None)`; call `register_configs()` at import.
- [x] Move heavy imports (torch, transformers, peft, trl, datasets) inside `main()` /
      helpers (Q6 #7); keep the `EarlyStoppingCallback` subclass importable (guarded import or
      define inside `main`).
- [x] Output folder = `HydraConfig.get().runtime.output_dir`; logging to `<run dir>/sft_translation.log`.
- [x] Data: build train/dev from `cfg.data.train` / `cfg.data.dev` lists (each side loaded in
      list order, joined line by line, as `main`'s `get_data`).
- [x] Port hydra2's fine-tuning-mode blocks as is (Q5): model loading for `full`/`lora`/`qlora`,
      bf16/fp16 from GPU support, LoRA block, trainer `bf16`/`fp16` flags.
- [x] Wire `train.*` into `SFTConfig`, `decode.*` into dev and test decoding, early-stopping
      settings into `EarlyStopping_MT_Callback` (Q6 #2–3).
- [x] Test decoding from `cfg.data.test` to `<run dir>/eval.pred.trg`; skip when `test` is
      absent (Q6 #1).
- [x] Save the model to `<run dir>/model` (Q6 #4).
- [x] Use `%s`-style logging for the trainer fp16/bf16 lines (not hydra2's broken call).
- [x] W&B for now: keep `main`'s behaviour (`report_to="wandb"` when `wandb.enabled`, aggregate
      BLEU/chrF/TER logging only); full W&B identity is phase C.

### A6. `hfmt/inf_translation.py`
- [x] Port hydra2's help-text change for `-c`/`-p` (Q6 #9). No other changes (Q8).

### A7. fr-en example
- [x] `egs/mmtc/fr-en/dataset.yaml` (`# @package data`, `${data.*}` paths; train/dev/test,
      dev = `valid.*` files as fixed in feature 07).
- [x] `egs/mmtc/fr-en/sft_lora.yaml` per the plan's example (`- config`, `- dataset`,
      overrides, `_self_`; `run_name: sft_lora`).

### A8. Done when
- [x] On a login node: `python -m hfmt.sft_translation --config-dir egs/mmtc/fr-en
      --config-name sft_lora --cfg job` prints the expected config (absolute data paths,
      components chosen) and runs without importing torch.
- [x] A typo override (`train.lerning_rate=1`) is rejected by the schema.
- [x] Every schema field is read somewhere (grep each field name in `hfmt/`). Still pending:
      `slurm.*`, `output_root` (B: grid.py); `run_name`, `wandb.entity`/`tags` (C); `data.data_root/src/trg`
      are read by `dataset.yaml` interpolation.
- [ ] On a GPU node: direct run with `train.max_steps=20 train.eval_steps=20
      hydra.run.dir=<tmp>` finishes; `<tmp>` holds `dev.step_20.pred`, `eval.pred.trg`, `model/`,
      the log.
- [ ] Same 20-step run with `train.finetune_mode=lora` and `=full` (full with
      `basemodel=qwen2.5-0.5b`) start training without error.
- [ ] `git status` shows no outputs or prediction files as untracked-to-add.
- [ ] Present commit message `08_hydra_redo-A: …`.

## B. Launch path (`grid.py`, `sft.sh`, `job.sh`)

Depends on A (schema, `conf/`, example).

### B1. `hfmt/grid.py`
- [ ] CLI: `python -m hfmt.grid <exp_dir> <exp_name> [overrides…]`; imports only
      `hfmt.hydra_config` (+ Hydra/OmegaConf).
- [ ] Split overrides into sweep (comma) and plain with `OverridesParser`; compose once
      (`initialize_config_dir(<exp_dir>)`, `hydra.searchpath=[file://$HFMT_ROOT/conf]`,
      `return_hydra_config=True`) to read `hydra.sweeper.params`.
- [ ] Cross product via `BasicSweeper.split_arguments(params + overrides)`.
- [ ] For each job: compose its config (fails on typo/bad value before anything is written).
- [ ] `output_root`: if unset, `<exp_dir>/outputs`.
- [ ] Job folders: n = 1 → `<output_root>/<run_name>`; n > 1 → `<output_root>/<run_name>-<idx>`,
      zero-padded when n > 10 (Q2d, Q7).
- [ ] Saved-config rule: if `<exp_name>` is `run_config` and `run_name` isn't overridden, the job
      folder is `<exp_dir>` itself.
- [ ] Existing folder → print a warning and proceed (Q2a).
- [ ] Write `<job_dir>/run_config.yaml`: `defaults: [base_config, _self_]` + the job config with
      `OmegaConf.resolve` applied (fully resolved; no `hydra:` section).
- [ ] Write `<job_dir>/git_rev` (`git rev-parse HEAD`, plus `-dirty` if the tree has changes).
- [ ] Write `<output_root>/<run_name>.grid.txt` (`idx: overrides` per job).
- [ ] Emit for `sft.sh`: job count, the job-folder list file, and sbatch flags from `slurm:`.

### B2. `egs/sft.sh`
- [ ] Usage check; resolve the path with `realpath` (repo-relative, cwd-relative or absolute);
      split into folder + stem; refuse a file named `config.yaml`.
- [ ] `source install/path.sh`; `PYTHONPATH=$HFMT_ROOT`; run `grid.py`; stop on its error.
- [ ] Print the grid lines and warnings.
- [ ] Slurm: `sbatch --array=0-$((n-1))%<array_parallelism> <flags> -o <job_dir>/slurm.out …
      egs/job.sh <job list file>`; print the job id; return.
- [ ] `LOCAL=1`: loop `i=0..n-1`, `SLURM_ARRAY_TASK_ID=$i bash egs/job.sh <list>`, log per job,
      report failures, continue.

### B3. `egs/job.sh`
- [ ] `source install/path.sh`; pick line `$SLURM_ARRAY_TASK_ID` of the job list; run
      `python -m hfmt.sft_translation --config-dir <job_dir> --config-name run_config
      hydra.run.dir=<job_dir>`.

### B4. Sweep example
- [ ] `egs/mmtc/fr-en/sft_lora.sweep.yaml` (standalone; `run_name: sft_lora_sweep`; grid lr
      1e-4,2e-4 × lora_r 8,16).

### B5. Done when
- [ ] `grid.py` dry checks on a login node: single → 1 job; sweep → 4; `lora.lora_r=8`
      pins → 2; `train.seed=1,2` adds → 8; bad value (`lora.lora_r=abc`) fails before writing.
- [ ] `run_config.yaml` contains no `${` and absolute paths.
- [ ] `LOCAL=1 bash egs/sft.sh egs/mmtc/fr-en/sft_lora.yaml train.max_steps=20
      train.eval_steps=20` on a GPU node → `egs/mmtc/fr-en/outputs/sft_lora/`.
- [ ] Slurm: single (20 steps) → `outputs/sft_lora/`; sweep (20 steps) → `outputs/sft_lora_sweep-{0..3}/`
      with the right values in each `run_config.yaml`; `sft_lora_sweep.grid.txt` written.
- [ ] Re-run from a saved config with a new `run_name` → new folder; without → same folder.
- [ ] `bash egs/sft.sh` with a file named `config.yaml` is refused.
- [ ] Present commit message `08_hydra_redo-B: …`.

## C. W&B and GPU tools

Depends on B (job folders and `run_name`).

### C1. W&B
- [ ] `init_wandb` in `main()` (main process only): `project`, `entity`, `name` = run-folder
      basename, `group` = `cfg.run_name`, `tags`, `dir` = run folder, `config` = resolved cfg
      (data-free); `WANDB_LOG_MODEL=false`; no `id`/`resume` (Q6 #5).
- [ ] Keep `report_to="wandb"` reusing that run; remove any duplicate init.
- [ ] Confidentiality review: list every W&B call and what it sends; nothing data-bearing
      (no samples, tables, artifacts, predictions).

### C2. GPU tools
- [ ] `tools/gpu_free.sh` from hydra2, as is.
- [ ] `tools/run_on_free_gpu.sh` from hydra2, calling `egs/sft.sh` and appending
      `slurm.gres=gpu:<type>:1`; update its usage comments.

### C3. Done when
- [ ] A 20-step Slurm sweep shows 4 W&B runs named `sft_lora_sweep-{0..3}` in group
      `sft_lora_sweep`; a single run named `sft_lora`.
- [ ] W&B run pages contain only config, aggregate metrics and system stats; no text, tables or
      files with examples.
- [ ] `bash tools/gpu_free.sh` prints counts; `bash tools/run_on_free_gpu.sh
      egs/mmtc/fr-en/sft_lora.yaml train.max_steps=20` submits with the chosen gres.
- [ ] Present commit message `08_hydra_redo-C: …`.

## D. Docs and tests

Depends on A–C (documents the final behaviour).

### D1. Tests
- [ ] `hfmt/test_grid.py` (standalone-runnable, no GPU): grid counts (single, sweep, pin, add
      axis), folder naming (n = 1, n > 1, zero-padding), saved-config rule, resolved save,
      typo rejected.
- [ ] Run tests standalone (`python hfmt/test_grid.py`); report results.

### D2. Docs
- [ ] CLAUDE.md: rewrite the Configuration / Parameter sweeps conventions and the project tree
      for the 08 design (Q12).
- [ ] `decisions.md`: copy `git show hydra2:decisions.md` as is, then append an 08 entry
      (key decisions + why; which hydra2 decisions it supersedes) (Q12).
- [ ] README: section on running SFT experiments (`egs/sft.sh`, single vs sweep, `LOCAL=1`,
      output folders, re-run from a saved config) (Q12).
- [ ] Headers/usage comments in `conf/*.yaml`, example experiments, `sft.sh`, `job.sh`,
      `grid.py` match the behaviour.

### D3. Done when
- [ ] Tests pass.
- [ ] Line budget (Q10): `git diff --shortstat main -- hfmt conf egs/sft.sh egs/job.sh`
      minus `hfmt/test_*.py` ≤ 400 added lines, or the commit message explains the excess.
- [ ] Grep docs for stale terms (`run.sh`, `submitit`, `hydra.launcher`, `+mmtc/`, `sweep_probe`,
      `lora.enabled`, `dataset_dir`) outside `decisions.md` history.
- [ ] Present commit message `08_hydra_redo-D: …`.

## E. Full-length validation

Depends on A–D.

- [ ] Full-length Slurm single run (`sft_lora.yaml`) and the 4-job sweep.
- [ ] Compare BLEU/chrF with `main`'s recipe or hydra2's last fr-en results (R5); note
      differences and their causes.
- [ ] Record results in `analysis/<YYYY-MM-DD>/` per CLAUDE.md if the developer wants an
      analysis write-up.
- [ ] Present commit message `08_hydra_redo-E: …` (todo/plan updates).
