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
