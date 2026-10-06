# Validation — 02_hydra_sweep

Off-cluster checks run during development, plus the on-cluster smoke test the **user** runs.
No GPU / training deps in the sandbox, so training itself is never executed here.

Env: conda `hfmt-hydra` (hydra 1.3.7, omegaconf 2.3.1, submitit launcher plugin present).
All commands are run with `HFMT_ROOT=$(pwd)` and `PYTHONPATH=$HFMT_ROOT` from the repo root.

---

## Group A — baseline (existing behavior, verified before any change)

The sweep *mechanism* already shipped in `00_hydra` (`egs/run.sh` runs `--multirun` +
`hydra/launcher=slurm`). Verified at the compose / override-grammar level
(`scratchpad/verify_sweep_baseline.py`); training was not executed.

- **A1 — fan-out.** A comma-list override is a sweep: `train.seed=37,42` →
  `is_sweep_override=True`, 2 values → 2 jobs. The first-preset grid
  (`learning_rate{2} × lora_target{2} × lora_r{3} × seed{2}`) cross-products to **24 jobs**,
  matching D-space.
- **A1b — composition.** Each swept combo composes and resolves against the structured
  schema with the distinct value applied (e.g. `cfg.train.seed` = 37 then 42).
- **A2 — non-clobbering output dirs.** `hydra.sweep.dir=${output_root}/${experiment}`,
  `hydra.sweep.subdir=${hydra.job.num}_${now:%Y-%m-%d_%H-%M-%S}` → a distinct run dir per job
  (job-num prefixed).
- **A3 — non-clobbering W&B ids.** `derive_run_identity()` builds `run_id` from the run-dir
  basename, so each sweep job gets a distinct id/name
  (`mmtc_fr-en_sft1_0_<ts>`, `..._1_<ts>`) → **request item 3 satisfied**.
  *Observed gap (motivates group B):* `group` is identical (`mmtc_fr-en_sft1`) across all
  jobs, and the swept values are not in the name — so runs are distinct but not legible or
  separable per sweep in the W&B UI.
- **A4 — swept values reach W&B.** `init_wandb()` logs the resolved, data-free config to
  `wandb.config` (`config={"hfmt": cfg_container, "run_dir": outdir}`), and Hydra writes
  `.hydra/overrides.yaml` into each run dir → **request item 4's durable record exists**
  today; group B makes it legible at a glance.

**Conclusion:** fan-out + non-clobbering dirs/ids already work (items 1–3). The feature's net
new work is W&B legibility (B), committed presets (C), the concurrency cap (D), and retiring
the dead tooling (E).

---

## Group B — W&B identifiability (code change)

Change in `hfmt/sft_translation.py`: `derive_run_identity()` now returns
`(run_id, run_name, group, tags)` and `main()` passes `is_multirun` +
`HydraConfig.overrides.task` into it. Verified off-cluster (`scratchpad/verify_b_identity.py`,
`py_compile` clean); the `wandb.init()` call itself runs on-cluster.

Key finding that shaped the design (probed with a throwaway `@hydra.main` script, now
deleted): for a sweep, all jobs share one timestamp in the run dir
(`<job.num>_<ts>`, only `job.num` differs) while `hydra.sweep.dir` is merely
`outputs/<experiment>` — so the **sweep id is the shared timestamp from the run-dir
basename**, not `sweep.dir` (Risk R3). `HydraConfig.overrides.task` gives clean
`['+experiment=...','model.lora_r=8','train.seed=37']` tokens to label from.

- **Single run:** unchanged behavior — `group=<experiment>` (or configured),
  `name=<experiment>-<ts>`, tags = configured only.
- **Sweep:** all jobs of one launch share `group=<base-group>-sweep-<ts>` (clusters a sweep
  together, separate from other sweeps/plain runs); each run is **named by its swept params**
  (`model.lora_r=16,train.seed=37`) and those tokens are added as **tags**; `run_id` stays
  per-job unique (no clobber). Verified across a 2×2 grid: 4 unique ids, 1 shared group.
- **Confidentiality (R5):** `data`/`data.*` overrides and any `/`-bearing token are stripped
  from name/tags, so data paths/content never reach W&B labels (verified: a
  `data.train_yaml=/proprietary/...` override leaves no trace in name/tags).
- **Fallback (R4):** when a job has no swept-param tokens, the name falls back to `job<num>`;
  labels are sanitized + length-capped (`_sanitize_label`).

## Group F — on-cluster smoke test (user-run)

*(To be filled in when groups B–E land.)*
