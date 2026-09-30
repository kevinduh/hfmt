# TODO — 00_hydra

Actionable breakdown of `plan.md` (decisions D1–D6 locked). Scope: **`sft_translation.py`
only**. Groups are ordered by dependency; items within a group are roughly ordered too.
Check items off as completed.

---

## A. Dependencies & environment (D3)
*No GPU here: install/validate only what runs on CPU; GPU deps are declared but verified on-cluster.*

- [x] Add Hydra deps to `install/install_hf_custom.sh`: `hydra-core`, `omegaconf`,
      `hydra-submitit-launcher` (pulls `submitit`), `hydra-colorlog`.
- [x] Pin + record tested versions of the new deps in `install/check_versions.py`
      (hydra 1.3.7, omegaconf 2.3.1, submitit 1.5.4).
- [x] Add an explicit `requirements.txt` (or `environment.yaml`) declaring the full dep set
      in one place; note GPU-only deps (`torch+cu128`, `bitsandbytes`) and their index URL.
- [x] Create/refresh a local CPU-only conda env with just the Hydra/OmegaConf/submitit deps
      to validate config composition (no torch-cuda / bitsandbytes needed for this).
      → env `hfmt-hydra` (python 3.13).
- [x] Confirm `hydra`, `omegaconf`, `submitit`, and the submitit launcher plugin import
      cleanly in that env. → `SlurmLauncher` + `LocalLauncher` discovered; colorlog plugin
      registered.

## B. Config schema & `conf/` skeleton (D2, D4)
*Depends on A. Structured/dataclass configs give validation; groups hold every knob.*

- [x] Create `conf/config.yaml` with the defaults list + `hydra` block (output dir, chdir —
      see group E). `hydra.job.chdir=false`; run dir `${output_root}/${experiment}/${now}`.
- [x] Define structured-config dataclasses (typed schema) and register them with Hydra's
      `ConfigStore` for validation. → `hfmt/hydra_config.py` (`register_configs()`).
- [x] `conf/model/qwen2.5-1.5b.yaml` — checkpoint id, quantization (4-bit nf4, double-quant,
      bf16 compute), LoRA `target`/`r`/`alpha`/`dropout` (D4: `lora_dropout` no longer hardcoded).
- [x] `conf/data/mmtc_fr-en.yaml` — train/dev/test manifest paths + instruction prefix
      (paths only, never data content — CLAUDE.md). Paths anchored to `${oc.env:HFMT_ROOT}`.
- [x] `conf/train/qlora_default.yaml` — lr, `lr_scheduler_type`, `max_steps`, `warmup_steps`,
      `batch_size`, `grad_accumulation`, `weight_decay`, `label_smoothing_factor`, `seed`,
      `eval_steps`, `logging_steps`, `save_total_limit`, early-stopping patience/threshold,
      `metric_for_best_model`.
- [x] `conf/decode/default.yaml` — `max_length`, `max_new_tokens`, `num_beams`, eval
      `batch_size` (D4: surface these previously-hardcoded magic numbers).
- [x] `conf/wandb/default.yaml` — project, entity, `group`, `tags` (absorbs the stubbed
      `--wandb_group`/`--wandb_tags`), run-naming template, enable/disable flag.
- [x] `conf/experiment/mmtc_fr-en_sft1.yaml` (`# @package _global_`) — a preset composing
      the above that reproduces today's `egs/mmtc/fr-en/sft1.sh` values exactly
      (select with `+experiment=mmtc_fr-en_sft1`).
- [x] Validate composition on CPU (via the compose API in env `hfmt-hydra`, since the
      `@hydra.main` entry point lands in Stage C): base + `+experiment` compose and resolve
      correctly; interpolations (`save_steps`, HFMT_ROOT paths) resolve; struct schema
      rejects unknown keys and wrong types.

## C. Port `sft_translation.py` to `@hydra.main` (D1, D2)
*Depends on B. Replace argparse; body logic unchanged, values read from `cfg`.*

- [x] Replace the argparse block + `main()` signature with a `@hydra.main(config_path,
      config_name)` entry receiving the typed config. Output dir = Hydra runtime output dir.
- [x] Remove module-level globals (`instruction_prefix`, `experiment_id`); pass values
      explicitly from `cfg` into helpers (`format_input_prompt`, `preprocess_fn`, etc.).
      `instruction_prefix` passed/closed-over; `experiment_id` was dead → deleted.
- [x] Map every former arg to its `cfg.<group>.<field>` location; delete dead/unused args.
      Also dropped the unreachable `attention`/`mlp` lora_target branches + fixed the
      misleading error message (overview #6), and removed an unused decode-debug assignment.
- [x] Route all newly-configurable knobs (decode `max_length`/`max_new_tokens`/`num_beams`/
      eval batch size, `lora_dropout`) to read from `cfg` (finishes D4).
- [x] Guard imports so config composition/validation does not require torch/bitsandbytes
      (heavy imports inside the run path, not at module top) — keeps CPU validation working.
      → verified: `--cfg job`/overrides run in env `hfmt-hydra` (no torch/transformers).
- [x] Keep local file/logging behavior as-is (local preds/logs are allowed per CLAUDE.md).
      Pred dumps still written locally; logging now flows through Hydra job logging.
- [x] Validation: full-file `py_compile` passes; `--cfg job` + CLI overrides compose;
      `hydra.run.dir` resolves. Real GPU training run deferred to on-cluster (Stage G).

## D. W&B ↔ output ↔ config linking (D6)
*Depends on C. Only metrics + data-free config may reach W&B (CLAUDE.md).*

- [x] Compute a single `run_id`/experiment slug and use it for: output dir, `WANDB_RUN_ID`,
      and W&B run `name`; set W&B `group=<experiment>` and `name=<experiment>-<job id>`.
      → `derive_run_identity()`; id = pinned `wandb.init(id=...)`, requeue-stable via run dir.
- [x] Point training output (checkpoints, preds, logs) at the Hydra run dir. (done in Stage C)
- [x] Initialize W&B with project/entity/group/tags from `conf/wandb/` and the pinned run id.
      → `init_wandb()` (rank-0 guard; HF Trainer's WandbCallback reuses the run).
- [x] Log the resolved config to `wandb.config` **after stripping/confirming no data
      content** (paths OK); add a guard/comment tying this to CLAUDE.md (Risk R7).
- [x] Log `run_dir` to `wandb.config` so the W&B run links back to on-disk outputs.
- [x] Verify only aggregate metrics + config reach W&B — no sample text / tables / artifacts.
      → `WANDB_LOG_MODEL=false`; callback `wandb.log` guarded; verified via an offline
      `wandb.init` test (config holds paths only). Real online run confirmed on-cluster (G).

## E. Slurm launcher (submitit) (D3)
*Depends on B/C. Cannot run real jobs here; validate config + dry run only.*

- [ ] `conf/hydra/launcher/slurm.yaml` — submitit_slurm params mirroring `sft1.sh`:
      partition `gpu`, `--gres=gpu:a100:1`, time, mem, cpus-per-task, job name.
- [ ] `conf/hydra/launcher/local.yaml` — local launcher for dev/CPU validation.
- [ ] Add `setup:` commands to the Slurm launcher that bootstrap the node env
      (`source install/path.sh`: `conda activate hfmt` + `module load cuda/...`) — **Risk R2**.
- [ ] Set `hydra.job.chdir` deliberately and make the code robust to CWD (absolute paths or
      `hydra.runtime.output_dir`) so relative `HFMT_ROOT`/`egs/...` paths don't break — **Risk R3**.
- [ ] Ensure run dirs resolve to `outputs/${experiment}/${run_id}/` and are unique across
      Slurm requeues (stable id) — **Risk R6**.
- [ ] Dry-run the job graph locally (`-m ... --cfg hydra`, launcher plugin loads, sbatch
      script renders) without submitting.

## F. Recipe migration & wiring
*Depends on C–E.*

- [ ] Replace/supplement `egs/mmtc/fr-en/sft1.sh` with the Hydra invocation (single
      `python hfmt/sft_translation.py -m experiment=mmtc_fr-en_sft1 hydra/launcher=slurm ...`),
      keeping the old `sbatch` comment for reference.
- [ ] Leave `sweep.yaml` / `sweep_run.sh` untouched (D5); add a one-line note that Hydra
      sweeps arrive in the future sweep request.
- [ ] Add `outputs/` and `multirun/` to `.gitignore`.

## G. Docs & validation handoff
*Depends on all above.*

- [ ] Update `README.md` with the Hydra run/override workflow and the Slurm launcher command.
- [ ] Update `CLAUDE.md`: note Hydra as the config system, `conf/` layout, and the W&B
      config-logging confidentiality guard.
- [ ] Write an on-cluster validation checklist (the GPU smoke run the **user** executes):
      submit `experiment=mmtc_fr-en_sft1` via Slurm, confirm env bootstrap, one eval cycle,
      W&B run appears with correct group/name and links to the output dir, metrics only.
- [ ] Record any notable decisions from implementation in the root `decisions.md`.

---

### Out of scope (this feature)
- Porting `inf_translation.py`, `train_seq2seq.py`, `decode_summarization.py` (later request).
- Hydra parameter **sweeps** and retiring the W&B-sweep tooling (future sweep request).
- Any real GPU training/validation in this sandbox (done on-cluster by the user).
