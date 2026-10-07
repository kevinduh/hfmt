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

- [x] Create the `conf/sweep/` config group. Presets are `# @package _global_` files that set
      `hydra.sweeper.params:` (basic sweeper grid), selectable with `+sweep=<name>`.
- [x] Add the first preset `conf/sweep/mmtc_fr-en_coarse.yaml` (D-space):
      `train.learning_rate: 2e-5,2e-4` × `model.lora_target: qv,all-linear` ×
      `model.lora_r: 8,16,32` × `train.seed: 37,42` = **24 jobs**. Header comment notes
      `model.lora_alpha` stays fixed at 32 (so `alpha/r` varies — confound; constant-ratio is a
      future refinement) and the confidentiality rule.
- [x] Confirmed the structured schema accepts every swept key (all existing fields — **Risk
      R9**). *(Doc that `+key=...` appends + group sweeps `model=a,b` → README in group F.)*
- [x] Validated preset composition off-cluster: `+experiment=mmtc_fr-en_sft1
      +sweep=mmtc_fr-en_coarse --cfg hydra` lands the 4 keys under `hydra.sweeper.params`
      (`@package _global_` works), and a no-torch probe multirun expanded to the **24-job** matrix.
- [x] Confirmed ad-hoc CLI sweeps still work unchanged (Group A: comma-lists expand).
      *(Override-grammar docs → README in group F.)*

## D. Concurrency cap + grid-size safety (D-cap)
*Depends on C. submitit submits a Slurm job array; cap simultaneous tasks for the 4-node tier.*

- [x] Added `array_parallelism: 4` to `conf/hydra/launcher/slurm.yaml` (matches the 4-node
      tier) with a comment; CLI-overridable via `hydra.launcher.array_parallelism=<N>` — **Risk R2**.
- [x] Verified the cap renders: `+experiment=... hydra/launcher=slurm --cfg hydra` shows
      `hydra.launcher.array_parallelism: 4` under the `SlurmLauncher` target (schema accepts it).
- [x] Grid-size warning (cross-products grow fast) added in the slurm.yaml comment.
      *(Fuller README guidance + the legacy 192-combo example → group F.)*

## E. Retire legacy W&B-sweep tooling (D4)
*Depends on C (the replacement preset must exist first so intent isn't lost — **Risk R7**).*

- [x] Confirmed the legacy space/metric intent (`eval/bleu` maximize; lr, scheduler,
      weight_decay, batch, seed, qlora_r/target) is represented as a **subset** by the coarse
      preset (lr, lora_target, lora_r, seed); `eval/bleu` is already logged by the callback;
      bayes+hyperband is out of scope (D1).
- [x] Deleted `egs/mmtc/fr-en/sweep.yaml` and `egs/mmtc/fr-en/sweep_run.sh` (kept the unrelated
      `inf.sh`). *(Both were untracked on `hydra`, so this removes dead working-tree files.)*
- [x] Grepped the repo: the only remaining mentions are in `features/00_hydra/` (historical
      records, correctly left as-is) and the `transformers/` submodule (unrelated). No README/
      code referenced the deleted files, so nothing to re-point.

## F. Docs & validation handoff
*Depends on all above.*

- [x] Added a **Sweeping parameters** subsection to `README.md` (ad-hoc CLI + override grammar,
      `+sweep=<name>` presets, `array_parallelism` cap + grid-size warning, no-clobber +
      identifiability notes) and a `conf/sweep/` bullet to the config-layout list.
- [x] Updated `CLAUDE.md`: `conf/sweep/` in the project structure + a Parameter-sweeps bullet
      noting it replaces the old W&B-sweep tooling.
- [x] Appended a `decisions.md` entry (basic grid; `conf/sweep/` presets; per-sweep W&B
      grouping/naming from the run-dir timestamp; `array_parallelism=4`) and marked the
      W&B-sweep→Hydra deferral done.
- [x] Wrote the on-cluster validation checklist in `features/02_hydra_sweep/validation.md`
      (Group F): a 2-point smoke test (array of 2, env bootstrap, non-clobber, shared group +
      legible names/tags, metrics-only, failure-resilience) plus the full `+sweep=` preset run.

---

### Out of scope (this feature)
- **Optuna / Bayesian / pruning** smart search (D1: basic grid only; would need a new dep +
  `main()` returning an objective + trial reporting).
- Constant-ratio `lora_alpha` (computed resolver) — deferred to a refinement sweep.
- Sweeps for the other entry points (`inf_translation`, `train_seq2seq`, `decode_summarization`).
- Any real GPU training/validation in this sandbox (done on-cluster by the user).
