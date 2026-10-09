# Decisions

Important, non-obvious decisions and their context. Update infrequently.

---

## Experiments are config + a single generic launcher (feature 00_hydra)

**Decision:** An experiment is defined by a config file in `conf/experiment/<name>.yaml`
(all tunable parameters) and launched with one generic script, `egs/run.sh <name>` — not
by a per-experiment shell script.

**Context:** Pre-Hydra, each experiment was its own shell script under `egs/<task>/<pair>/`
(e.g. `sft1.sh`) that hard-coded parameters and invoked the trainer. After the Hydra
migration, parameters live in the experiment YAML, so a per-experiment `.sh` would only
duplicate the experiment name and boilerplate. We therefore removed `egs/mmtc/fr-en/sft1.sh`
and added `egs/run.sh`, which activates the env (`install/path.sh`) and submits to Slurm via
the submitit launcher (`-m +experiment=<name> hydra/launcher=slurm`). Run it from a login
node — do **not** `sbatch` it; submitit generates and submits the job.

Trade-off accepted: slightly less "browse `egs/` to see experiments" discoverability than a
file-per-experiment, in exchange for a single source of truth (the YAML) and no shell
boilerplate. This matches the `egs/run.sh` direction suggested in `overview.md` §5.5.

Scope: applies to the `sft_translation.py` workflow (feature 00_hydra). The other entry
points and the legacy W&B sweep tooling (`egs/mmtc/fr-en/sweep*.{yaml,sh}`) are not yet
migrated.

---

## Key Hydra technical choices (feature 00_hydra)

Non-obvious implementation decisions, for future readers/maintainers:

- **`hydra.job.chdir=false`.** The code targets the Hydra run dir explicitly (via
  `HydraConfig.runtime.output_dir`) and resolves data paths from `${oc.env:HFMT_ROOT}`, so it
  is CWD-independent and we don't let Hydra change the working directory.
- **Heavy imports are lazy** (inside `main()`/helpers; `transformers.EarlyStoppingCallback`
  guarded at module top). This keeps `--cfg job` and config composition runnable on a CPU-only
  box without torch/bitsandbytes installed — the basis for all pre-cluster validation.
- **We own the W&B run** (`init_wandb`, main process only), then the HF Trainer's WandbCallback
  reuses it. `run_id` is derived from the run-dir basename so it is unique per launch but stable
  across a Slurm requeue (`resume="allow"` reattaches). Config logged to `wandb.config` is
  data-free by construction (paths + hyperparams only); `WANDB_LOG_MODEL=false`.
- **Slurm via `hydra-submitit-launcher`**, not hand-written sbatch. The compute-node env is
  bootstrapped through the launcher's `setup:` (`source install/path.sh`), which is the only
  hook guaranteed to run before Python on the node.

Deferred (for the future sweep request): external/proprietary data root (`conf/data` still
points in-repo via `HFMT_ROOT`), `.gitignore`/`path.sh` data hardening, and migrating the
W&B sweep tooling to Hydra's sweeper. Considering a move to `uv` for env management.
*(The W&B-sweep → Hydra-sweeper migration is now done — see below.)*

---

## Parameter sweeps use Hydra's basic grid sweeper (feature 02_hydra_sweep)

**Decision:** Sweeps are plain Hydra `--multirun` grids over config knobs, fanned out to Slurm
by the existing submitit launcher. Reusable grids are committed under `conf/sweep/<name>.yaml`
(selected with `+sweep=<name>`); ad-hoc comma-list overrides also work. The legacy W&B-sweep
tooling (`egs/mmtc/fr-en/sweep*.{yaml,sh}`) is removed.

**Context:** The fan-out already existed — `egs/run.sh` runs `--multirun` with
`hydra/launcher=slurm`, and each job already got a unique output dir + W&B run id. So this
feature added ergonomics and legibility, not fan-out:

- **Basic grid, not Optuna.** Chose Hydra's built-in sweeper: zero new deps and no need for
  `main()` to return an objective. Bayesian/pruning search (the old `sweep.yaml`'s
  bayes+hyperband) is explicitly out of scope; it would need the Optuna plugin + objective
  reporting. Grid is enough for "experiment over multiple parameters."
- **W&B identifiability.** In multirun, `derive_run_identity()` groups a sweep's runs under
  `<base-group>-sweep-<timestamp>` and names/tags each run by its swept values. The sweep id is
  the **shared timestamp in the run-dir basename** (unique per launch, shared across its jobs),
  *not* `hydra.sweep.dir` — which is just `outputs/<experiment>` and identical across all sweeps
  of an experiment. Confidentiality: `data`/path-bearing overrides are stripped from names/tags.
- **Concurrency cap.** `array_parallelism: 4` in the Slurm launcher (the node-tier size) bounds
  how many array tasks run at once; grids are cross-products, so job counts multiply fast.

---

## Run outputs are aligned with W&B names (feature 03_output_wandb_align)

**Decision:** The output directory layout mirrors the W&B group→run hierarchy, and the run-dir
basename **is** the W&B run name, so a W&B run maps 1:1 to its files on disk. A sweep lands at
`outputs/<experiment>/sweep-<ts>/<run_label>/` and a plain run at `outputs/<experiment>/<ts>/`;
the trained adapter is saved inside the run dir as `model/` (was the `outdir + "_b"` sibling).

**Context:** Previously the run dir (`<job.num>_<ts>`) bore no resemblance to the W&B run name
(the swept-param string), so you couldn't get from a W&B run to its `eval.pred.trg` to score it.

- **Single source of truth.** `build_run_label()` turns Hydra's `override_dirname` into a
  strict, shell-safe (`[A-Za-z0-9._-]`), descriptive label (group prefix stripped, leaf keys
  kept verbatim — `lora_r`, `lora_target`, … — pairs joined by `__`). It is exposed to
  `hydra.sweep.subdir` via a custom OmegaConf resolver (`hfmt_runlabel`, registered at import),
  and `derive_run_identity()` reads the same string back off the path for the W&B name. One
  function feeds both, so dir and W&B name cannot drift.
- **Sweep id moved up to `hydra.sweep.dir`** (`sweep-<ts>`), **evolving the 02 decision.** In 02
  the shared timestamp lived in the subdir basename; here it moves to the sweep root so all
  jobs of a launch share one `sweep-<ts>/` parent (resolved once on the login node, stable
  across a requeue). `derive_run_identity` now derives the W&B group from that parent dir
  (`<base-group>-sweep-<ts>`) and the run name/tags from the run-dir basename — superseding 02's
  `job.partition("_")` parsing and retiring `_sweep_param_tokens()`.
- **Confidentiality is centralized** in `build_run_label()`: it drops `data`/`data.*` keys and
  any `/`-bearing value, so no data path enters the dir name (hence the W&B run name). The label
  being the sole confidentiality chokepoint is deliberate — audit it if the filter changes.
- **Descriptive, not abbreviated** (developer preference): keys stay verbatim and `lora` is kept
  in LoRA params to avoid confusion. Note values are resolved numbers, so `2e-4` → `0.0002`.
- **Forward-only.** Existing `outputs/` dirs from before this change are not migrated.
