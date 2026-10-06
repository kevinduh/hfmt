# Plan — 02_hydra_sweep

**Status:** finalized — all decisions locked; see `todo.md` for the actionable breakdown.
See `request.md`.

## Request (restated)

1. Implement Hydra parameter sweeps to experiment over multiple parameters.
2. Keep the **submitit** approach already in use — a sweep fans out into **multiple Slurm jobs**.
3. Per-job W&B results must **not clobber** each other.
4. Enough info must reach W&B that each run's results are **identifiable** (likely already
   true — confirm).

## Decisions (locked)

- **D1 — Sweeper backend: Hydra's built-in *basic* grid sweeper.** Zero new deps; no change
  to `main()`'s signature (it need not return an objective). Exhaustive grid / discrete
  choices over config knobs. **Optuna / Bayesian / pruning is explicitly out of scope** for
  this feature (noted, not built).
- **D2 — Sweep definition: committed presets + ad-hoc CLI.** Add a `conf/sweep/<name>.yaml`
  group (sets `hydra.sweeper.params`) selected via `+sweep=<name>`, and keep ad-hoc
  comma-list overrides working. Reproducible sweeps match the "experiment = committed config"
  ethos.
- **D3 — W&B identifiability: per-sweep group + override-encoded run name/tags.** Localized
  change to `derive_run_identity()`/`init_wandb()`: a shared per-sweep **group**
  (`<experiment>-sweep-<sweepid>`) in multirun, plus the swept params folded (sanitized +
  truncated) into the run **name/tags**, with fallback to `job.num`. Single runs keep today's
  behavior. (Swept values already land in `wandb.config` — that stays as the durable record.)
- **D4 — Retire legacy W&B-sweep tooling.** Delete `egs/mmtc/fr-en/sweep.yaml` +
  `sweep_run.sh` (dead; target the old argparse CLI) and port a subset of their search space
  into the first `conf/sweep/` preset.
- **D-cap — `array_parallelism: 4`** in `slurm.yaml`, matching the developer's 4-node Slurm
  tier (at most 4 sweep jobs run concurrently); overridable on the CLI, plus a docs warning
  about grid size. *(Locked — developer is on the 4-node tier.)*
- **D-space — first preset sweeps** `train.learning_rate` (2e-5, 2e-4) ×
  `model.lora_target` (qv, all-linear) × `model.lora_r` (8, 16, 32) × `train.seed` (37, 42)
  = **24 jobs** (6 waves of 4 under D-cap). `model.lora_alpha` is **held fixed at 32** (the
  basic sweeper does cross-products, not zipped pairs, so the effective scaling `alpha/r`
  varies across `lora_r` — documented confound; constant-ratio deferred to a refinement sweep).

## Current state (what already works — read before planning edits)

Feature `00_hydra` already wired most of the sweep mechanism. Confirming this up front
keeps the feature correctly scoped (it is smaller than it looks):

- **Fan-out already exists.** `egs/run.sh` already runs `python -m hfmt.sft_translation
  --multirun ... hydra/launcher=slurm`. Comma-separated overrides therefore already sweep:
  `bash egs/run.sh mmtc_fr-en_sft1 model.lora_r=8,16,32` submits **3 Slurm jobs** today via
  the submitit launcher (Hydra's default *basic* sweeper takes the cross-product). Request
  items 1–2 are largely satisfied by existing code.
- **Output dirs don't clobber.** `conf/config.yaml` sets
  `hydra.sweep.subdir=${hydra.job.num}_${now:%Y-%m-%d_%H-%M-%S}` under
  `hydra.sweep.dir=${output_root}/${experiment}`, so each sweep job gets a distinct run dir.
- **W&B runs don't clobber.** `derive_run_identity()` builds `run_id` from the run-dir
  basename (`<experiment>_<jobnum>_<ts>`), so each sweep job is a **distinct** W&B run;
  `resume="allow"` means only a Slurm *requeue* of the *same* job re-attaches. Request item 3
  appears satisfied — to be **verified**, not rebuilt.
- **Swept values are already recoverable.** The resolved, data-free config (including the
  swept knobs) is logged to `wandb.config` per run, and Hydra writes `.hydra/overrides.yaml`
  into each run dir. So request item 4's data *exists* per-run today.

**Therefore the real gaps are ergonomics and legibility, not fan-out:**
- (a) Sweeps are **ad-hoc only** (retyped CLI comma-lists); no committed, reproducible sweep
  definition — at odds with the project's "experiment = a committed config file" ethos.
- (b) Sweep runs are **not legible at a glance** in the W&B UI: the run *name* is
  `<experiment>-<jobnum>_<ts>` and the *group* is just `<experiment>`, so every run of every
  sweep (and plain runs) pile into one group and you must open each run's config to see which
  parameter values it used. The swept values aren't in the name/tags, and sweeps aren't
  separated from one another.
- (c) **No concurrency cap** — a large grid floods the partition (`array_parallelism` unset).
- (d) The **legacy W&B-sweep tooling** (`egs/mmtc/fr-en/sweep.yaml` + `sweep_run.sh`) is
  non-functional (targets the old argparse CLI) and `decisions.md` explicitly defers its
  migration to Hydra's sweeper to *this* feature.

## Goal

Make parameter sweeps **first-class and reproducible**: a committed way to define a sweep, a
single polite submit path that fans out to Slurm via submitit, per-run W&B results that are
**unambiguously identifiable and grouped per sweep**, and retirement of the dead W&B-sweep
tooling — all without regressing data confidentiality.

## Guiding constraints

- **Confidentiality (CLAUDE.md).** Only aggregate metrics + data-free config reach W&B.
  Swept values are hyperparameters (safe), but `override_dirname`/run-name strings can
  include a **data path** if someone sweeps `data=...`; that is a path, not content, and
  paths are already logged — still, keep names/tags path-free where practical and never put
  data content in them.
- **No local GPU (and no training deps in the base env here).** As in `00_hydra`, only
  config composition / job-graph can be validated off-cluster (`--cfg job`, `--multirun
  ... --cfg job`, possibly the `local` launcher). A real sweep is a **user-run** cluster step.
- Keep the pinned `transformers` submodule and the conda install flow. Prefer **no new
  heavyweight deps** unless a smart sweeper is explicitly chosen (Q1).

## Proposed approach

### A. Sweep definition (ergonomics) — *D1/D2*
Two complementary paths; recommend supporting both:
- **Ad-hoc CLI (already works):** `bash egs/run.sh <exp> model.lora_r=8,16
  train.learning_rate=2e-4,2e-5` → cross-product. Document the override grammar
  (`choice` lists, `range(...)`, `glob(...)`).
- **Committed sweep presets (new):** a `conf/sweep/<name>.yaml` group that sets
  `hydra.sweeper.params:` (read by the basic sweeper), selected with
  `bash egs/run.sh <exp> +sweep=<name>`. This makes a sweep reproducible and reviewable,
  matching the "experiment = committed config" decision. (If a smart sweeper is chosen in
  Q1, this same group also carries its `hydra.sweeper` block.)

### B. Sweeper backend — *D1: basic grid*
Use Hydra's built-in basic sweeper (the default under `--multirun`). Grid / discrete
choices via the override grammar; no new deps and no change to `main()`'s signature.
*Optuna/Bayesian/pruning is out of scope — see Decisions D1 and Risk R1.*

### C. W&B identifiability + per-sweep grouping (the main code change) — *D3*
Localized change to `derive_run_identity()` / `init_wandb()` using `HydraConfig.get()`:
- **Per-sweep group.** When `HydraConfig.get().mode == RunMode.MULTIRUN`, derive a
  **sweep id** shared by all jobs of the launch (the launch timestamp / sweep-dir basename is
  identical across jobs — see R3) and set the W&B `group` to `<experiment>-sweep-<sweepid>`
  so each sweep's runs cluster together and separate from plain runs and other sweeps.
  Single (non-multirun) runs keep today's behavior.
- **Legible run name.** Fold `HydraConfig.get().job.override_dirname` (the compact
  `key=val,key=val` of this job's overrides) into the W&B run **name** and/or **tags**, so the
  UI shows `...lora_r=16,learning_rate=2e-4` instead of an opaque `...-3_ts`. **Sanitize +
  truncate** (reuse/extend the existing id regex) and **fall back to job.num** when
  `override_dirname` is empty (e.g. params swept via the preset).
- Keep `run_id` run-dir-derived (unique per job, stable across requeue) — unchanged.
- Confirm the swept values remain in `wandb.config` (they do) — item 4's durable record.

### D. Concurrency + cluster politeness — *D-cap*
- Add `array_parallelism: <N>` to `conf/hydra/launcher/slurm.yaml` so at most N sweep jobs
  run at once (submitit uses a Slurm **job array**; this caps simultaneous tasks).
- Document a guard against accidental giant grids (the legacy space was
  2·2·2·3·2·2·2 = **192** combos); recommend starting small / using the preset.

### E. Retire legacy W&B-sweep tooling — *D4*
- Delete (or clearly archive) `egs/mmtc/fr-en/sweep.yaml` + `sweep_run.sh` (dead code).
- Reproduce a sensible subset of their search space as the first committed
  `conf/sweep/<name>.yaml` so the documented intent survives the migration.

### F. Docs + validation
- README "Running experiments": add a **Sweeps** subsection (CLI grammar, `+sweep=` presets,
  concurrency cap, where outputs/W&B land, how runs are named/grouped).
- Update `CLAUDE.md` project-structure note (`conf/sweep/`) and append a short
  `decisions.md` entry (sweeper choice + W&B sweep-grouping scheme).
- **Off-cluster validation:** `--multirun ... --cfg job` shows the composed job matrix;
  verify distinct run dirs, distinct `run_id`, and a shared sweep group across jobs without a
  real run. **On-cluster smoke test (user-run):** a tiny 2-point sweep (e.g.
  `train.seed=37,42 train.max_steps=20`) confirming two Slurm jobs, two non-clobbering W&B
  runs in one sweep group, with legible names.

## Phased breakdown (→ becomes todo.md)
1. **Verify current behavior** — confirm fan-out, non-clobbering dirs + W&B ids via
   `--multirun --cfg job`; document the baseline.
2. **W&B legibility (C)** — per-sweep group + override-encoded name/tags + sanitization/fallback.
3. **Sweep presets (A/B)** — `conf/sweep/` group + `+sweep=` wiring; first preset (D-space).
4. **Concurrency (D)** — `array_parallelism` (D-cap) + grid-size guidance.
5. **Retire legacy tooling (E)** — remove `sweep*.{yaml,sh}`; port a subset to the preset.
6. **Docs + validation (F)** — README/CLAUDE.md/decisions.md; off-cluster checks + a user
   on-cluster smoke-test checklist.

## Risks
- **R1 — Smart-search scope jump.** Matching the old `bayes` + `hyperband` behavior needs the
  Optuna plugin **and** `main()` returning an objective (+ pruning hooks). Much larger than a
  grid. Decide in Q1; recommend grid now, Optuna later.
- **R2 — Grid explosion / partition flooding.** Cross-products grow fast (192 in the legacy
  space). Without `array_parallelism` a sweep can swamp the queue or hit QOS limits, and burn
  large GPU-hours. Needs a cap + docs (D).
- **R3 — Sweep-id assumption.** The per-sweep group relies on `${now:}` / sweep-dir being
  identical across all jobs of one multirun. True for a single submitit launch (resolved once
  at app start), but **verify**; if it ever differs per job, sweep grouping fragments. Prefer
  deriving the id from `HydraConfig.sweep.dir` (sweep-wide) over re-reading `now:`.
- **R4 — W&B name collisions / illegal chars.** `override_dirname` can contain `/`, `=`,
  long values, or a data path. The run *name* isn't sanitized today (only the id is). Must
  sanitize + truncate, and fall back to job.num when empty, or runs get ugly/duplicate names.
- **R5 — Confidentiality regression via names.** A `data=...` sweep could put a path into the
  name/tags. Paths are allowed, but guard against dumping anything data-bearing; keep the
  override string to hyperparameters where feasible.
- **R6 — Requeue vs. fresh re-run semantics.** `resume="allow"` + run-dir-derived id means a
  Slurm requeue correctly resumes one run, but re-launching the *same* sweep later creates new
  dirs/ids (new runs). Confirm that's the desired semantics (it matches `00_hydra`).
- **R7 — Losing documented intent on deletion.** Removing `sweep.yaml` discards the only
  recorded search space + metric/goal (`eval/bleu`, maximize). Capture an equivalent preset
  before deleting (E).
- **R8 — Off-cluster validation only.** No local GPU and no training deps in the base env; a
  real multi-job sweep can't be run here — only config/job-graph composition. On-cluster smoke
  test is a user step (as in `00_hydra`).
- **R9 — Structured-config strictness.** Sweeping an unknown key is rejected by the schema
  (good), but `+key=...` *appends*; document which knobs are sweepable and that group sweeps
  (`model=a,b`) are supported.
- **R10 — Disk growth.** N trials × checkpoints (`save_total_limit`, `*_b` adapter dumps,
  per-step `*.pred`) under `outputs/` can balloon. Note a disk budget / cleanup in docs.

## Open questions

All resolved — folded into Decisions D1–D4, D-cap, and D-space above.
