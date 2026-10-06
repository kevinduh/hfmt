# TODO — 02_hydra_sweep

Actionable breakdown of `plan.md` (decisions D1–D4, D-cap, D-space locked). Scope: **Hydra
parameter sweeps for the `sft_translation.py` workflow** — basic grid sweeper, committed
presets + ad-hoc CLI, W&B legibility, concurrency cap, and retiring the dead W&B-sweep
tooling. Groups are ordered by dependency; items within a group are roughly ordered too.
Check items off as completed.

*No GPU / training deps in this sandbox: validate config + job-graph composition only; a real
multi-job sweep is an on-cluster, user-run step (see group F).*

---

## A. Verify the existing baseline (no code change)
*The fan-out already exists (`egs/run.sh` runs `--multirun` + `hydra/launcher=slurm`). Confirm
and document what already works before adding to it — avoids rebuilding it.*

- [x] Compose a 2-point sweep off-cluster and inspect the job matrix. (`--cfg` can't combine
      with `-m` in Hydra 1.3, so verified via the override-grammar parser + compose API in
      `scratchpad/verify_sweep_baseline.py`: `train.seed=37,42` → 2-value sweep; the D-space
      grid cross-products to 24 jobs; each combo composes with the distinct resolved `seed`.)
- [x] Confirm non-clobbering **output dirs**: `--cfg hydra` shows
      `hydra.sweep.subdir=${hydra.job.num}_${now:...}` → a distinct run dir per job.
- [x] Confirm non-clobbering **W&B ids**: `derive_run_identity()` yields distinct
      `run_id`/`name` per job (`..._0_<ts>`, `..._1_<ts>`) → request item 3 holds. *(Observed:
      `group` is shared across all jobs and the name omits the swept values — the gap group B fixes.)*
- [x] Confirm swept values already reach `wandb.config` (`init_wandb` logs the resolved config)
      and Hydra writes `.hydra/overrides.yaml` per job → request item 4's record exists today.
- [x] Note the baseline findings in `features/02_hydra_sweep/validation.md` (Group A) so the
      "what's new" scope stays clear.

## B. W&B identifiability: per-sweep group + legible names (D3)
*Depends on A. The main code change — localized to `hfmt/sft_translation.py`
(`derive_run_identity()` / `init_wandb()`). Only metrics + data-free config reach W&B (CLAUDE.md).*

- [x] In `derive_run_identity()`, detect sweep mode via
      `HydraConfig.get().mode == RunMode.MULTIRUN` (added `from hydra.types import RunMode`;
      `is_multirun` passed in from `main()`).
- [x] Derive a **sweep id shared by all jobs of one launch** — from the **timestamp in the
      run-dir basename** (`<job.num>_<ts>`), **not** `sweep.dir` (which is just
      `outputs/<experiment>`, shared across *all* sweeps of the experiment) — **Risk R3
      resolved**. In multirun set `group = <base-group>-sweep-<ts>` (appends to the configured
      group); single runs keep `group=<experiment>`.
- [x] Fold this job's swept params into the W&B run **name** from
      `HydraConfig.overrides.task` (cleaner than `override_dirname`), e.g.
      `model.lora_r=16,train.seed=37`; the same tokens are added as W&B **tags** (+ a `sweep` tag).
- [x] **Sanitize + truncate** name/tags via `_sanitize_label` (alnum + `_.=,+-`, 128-char cap).
- [x] **Fallback** to `job<num>` when no swept-param tokens remain (e.g. preset-only sweep) —
      **Risk R4**.
- [x] **Confidentiality guard (Risk R5):** `_sweep_param_tokens` drops `experiment`/`sweep`
      selectors, `hydra*` keys, and any `data`/`data.*` or `/`-bearing token, so data paths
      never reach name/tags. Comment ties it to CLAUDE.md. (Verified with a
      `data.train_yaml=/proprietary/...` override leaving no trace.)
- [x] Keep `run_id` run-dir-derived + `resume="allow"` unchanged (unique per job, requeue-stable).
- [x] Validate off-cluster (`scratchpad/verify_b_identity.py`, `py_compile`): a 2×2 sweep →
      one shared group, 4 distinct ids, param-encoded names/tags, no data content. (Real online
      `wandb.init` confirmed on-cluster, group F.)

## C. Committed sweep presets + ad-hoc CLI (D1, D2)
*Depends on B. Basic grid sweeper (Hydra default under `-m`); no new deps; no change to
`main()`'s signature.*

- [ ] Create the `conf/sweep/` config group. Author presets as `# @package _global_` files
      that set `hydra.sweeper.params:` (the basic sweeper's grid), selectable with `+sweep=<name>`.
- [ ] Add the first preset `conf/sweep/mmtc_fr-en_coarse.yaml` (D-space):
      `train.learning_rate: 2e-5,2e-4` × `model.lora_target: qv,all-linear` ×
      `model.lora_r: 8,16,32` × `train.seed: 37,42` = **24 jobs**. Document in a header comment
      that `model.lora_alpha` stays fixed at 32 so `alpha/r` varies (confound; constant-ratio
      is a future refinement).
- [ ] Confirm the structured schema accepts every swept key (all are existing fields — **Risk
      R9**); document that `+key=...` *appends* and that group sweeps (`model=a,b`) also work.
- [ ] Validate preset composition off-cluster: `python hfmt/sft_translation.py -m
      +experiment=mmtc_fr-en_sft1 +sweep=mmtc_fr-en_coarse --cfg hydra` expands to the 24-job
      matrix (verify `hydra.sweeper.params` packaging actually lands — Hydra `@package` is finicky).
- [ ] Confirm ad-hoc CLI sweeps still work unchanged
      (`bash egs/run.sh mmtc_fr-en_sft1 model.lora_r=8,16`); document the override grammar
      (`choice` lists, `range(...)`, `glob(...)`).

## D. Concurrency cap + grid-size safety (D-cap)
*Depends on C. submitit submits a Slurm job array; cap simultaneous tasks for the 4-node tier.*

- [ ] Add `array_parallelism: 4` to `conf/hydra/launcher/slurm.yaml` (matches the developer's
      4-node Slurm tier) with a comment; keep it CLI-overridable — **Risk R2**.
- [ ] Verify the cap renders into the submitit/sbatch array spec via `--cfg hydra` /
      dry-run (no real submission here).
- [ ] Add docs guidance warning that grids are cross-products (the legacy space was **192**);
      recommend starting from the coarse preset and refining — **Risk R2**.

## E. Retire legacy W&B-sweep tooling (D4)
*Depends on C (the replacement preset must exist first so intent isn't lost — **Risk R7**).*

- [ ] Confirm the search space / metric intent of `egs/mmtc/fr-en/sweep.yaml`
      (`eval/bleu`, maximize; lr, scheduler, weight_decay, batch, seed, qlora_r/target) is
      represented (as a subset) by the new preset before deleting.
- [ ] Delete `egs/mmtc/fr-en/sweep.yaml` and `egs/mmtc/fr-en/sweep_run.sh` (dead — target the
      removed argparse CLI).
- [ ] Grep the repo for references to the deleted files (README, `egs/`, comments) and update
      them to point at the Hydra sweep workflow.

## F. Docs & validation handoff
*Depends on all above.*

- [ ] Add a **Sweeps** subsection to `README.md`: ad-hoc CLI grammar, `+sweep=<name>` presets,
      `array_parallelism` cap + grid-size warning, where outputs/W&B land, and how sweep runs
      are named/grouped.
- [ ] Update `CLAUDE.md`: add `conf/sweep/` to the project structure and a one-line note on the
      Hydra grid sweep workflow (replacing the legacy W&B-sweep mention).
- [ ] Append a short `decisions.md` entry: basic grid sweeper (Optuna deferred), committed
      `conf/sweep/` presets, the per-sweep W&B grouping/naming scheme, and `array_parallelism=4`.
      Mark the "migrate W&B sweep tooling to Hydra's sweeper" deferral in `decisions.md` as done.
- [ ] Write an on-cluster validation checklist (**user-run** GPU smoke test) as
      `features/02_hydra_sweep/validation.md`: launch a tiny 2-point sweep
      (`bash egs/run.sh mmtc_fr-en_sft1 train.seed=37,42 train.max_steps=20`), confirm **two**
      Slurm array tasks, two **non-clobbering** W&B runs sharing one sweep **group** with
      **legible** names/tags, metrics-only, and that one failed/OOM trial doesn't abort the
      others (**Risk R2**). Then a note to run the full `+sweep=mmtc_fr-en_coarse` preset.

---

### Out of scope (this feature)
- **Optuna / Bayesian / pruning** smart search (D1: basic grid only; would need a new dep +
  `main()` returning an objective + trial reporting).
- Constant-ratio `lora_alpha` (computed resolver) — deferred to a refinement sweep.
- Sweeps for the other entry points (`inf_translation`, `train_seq2seq`, `decode_summarization`).
- Any real GPU training/validation in this sandbox (done on-cluster by the user).
