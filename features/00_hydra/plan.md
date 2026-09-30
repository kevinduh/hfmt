# Plan — 00_hydra

**Status:** finalized — all decisions locked; see `todo.md` for the actionable breakdown.
See `request.md`.

## Decisions (locked)
- **D1 — Scope:** this pass covers **`sft_translation.py` only**; the other three entry
  points are a later request.
- **D2 — Refactor style:** replace argparse with a **full `@hydra.main`** entry + typed
  structured config.
- **D3 — Env:** **update install scripts + version pins and validate config composition on
  CPU only.** GPU deps and real runs are validated on the cluster by the user (no local GPU).
- **D4 — Params:** move **all tweakable knobs** into config, including today's hardcoded
  magic numbers.
- **D5 — W&B sweep tooling:** leave `sweep.yaml`/`sweep_run.sh` in place, untouched, this
  pass; move the stubbed `--wandb_group`/`--wandb_tags` into `conf/wandb/`. Hydra's own
  sweeper adopts the run-linking in the future sweep request.
- **D6 — Output + naming:** run dirs under a git-ignored top-level
  `outputs/${experiment}/${run_id}/`; W&B `group=<experiment>`,
  `name=<experiment>-<hydra job id/timestamp>`, `WANDB_RUN_ID` pinned to the same slug so
  Slurm requeues map to one run; `run_dir` logged to `wandb.config`.

## Goal

Introduce [Hydra](https://hydra.cc) so experiments are driven by composable config
files instead of hardcoded values + argparse + hand-written recipe shell scripts. Runs
must launch on **Slurm**, log cleanly to **W&B** with an unambiguous run ↔ output ↔ config
link, and keep the door open for parameter **sweeps** (built later, not now).

## Guiding constraints

- **Data confidentiality (CLAUDE.md):** only aggregate metrics + *config* may reach W&B.
  Config may contain data **paths** but never data **content**. This holds throughout.
- No GPU in the dev environment: GPU-dependent deps (`torch+cu128`, `bitsandbytes`) and any
  real training run can only be validated on the target Slurm cluster. Locally we can
  validate config composition and job-graph/dry-run only.
- Keep the pinned `transformers` submodule and the existing conda-based install flow.

## Proposed approach

### 1. Scope (first pass)
Make **`sft_translation.py`** the reference Hydra implementation (it is the active fr-en
workflow). Establish the config tree + Slurm launcher + W&B linking there, then port the
other three entry points (`inf_translation`, `train_seq2seq`, `decode_summarization`) once
the pattern is proven. *(Confirm — see Open Questions Q1.)*

### 2. Config layout (`conf/`)
Config groups so every tweakable knob lives in a file, composed via a defaults list:

```
conf/
├── config.yaml            # defaults list + hydra settings (output dir, chdir)
├── model/                 # checkpoint id, quantization, lora target/r/alpha/dropout
├── data/                  # train/dev/test paths (the YAML manifest), instruction prefix
├── train/                 # lr, scheduler, steps, warmup, batch, weight_decay, seed, ...
├── decode/                # max_length, max_new_tokens, num_beams, eval batch size
├── wandb/                 # project, entity, group, tags, run-naming
├── experiment/            # @package _global_ presets == today's recipe scripts
│                          #   e.g. experiment/mmtc_fr-en_sft1.yaml
└── hydra/launcher/        # local.yaml + slurm.yaml (submitit)
```

Each current `egs/**/sft1.sh` becomes one `experiment/*.yaml` preset. Overrides on the
command line (`train.learning_rate=2e-4`) or in presets replace the shell-var editing.

**This surfaces the hardcoded magic numbers** (`max_length=128`, `max_new_tokens=128`,
inference `batch_size=16`, `lora_dropout=0.1`) into `decode/`/`model/`, since "isolate all
tweakable params" requires it. *(Scope confirm — Q4.)*

### 3. Entry point refactor
Replace the argparse block in `sft_translation.py` with a `@hydra.main` entry that receives
a typed config (structured config / dataclasses for validation). Body logic is unchanged;
values are read from `cfg` instead of `args`/globals. *(Confirm approach — Q2.)*

### 4. Slurm integration
Use **`hydra-submitit-launcher`** (`hydra/launcher: slurm`) so `python train.py -m ...`
submits via `sbatch`. Slurm resources (partition `gpu`, `--gres=gpu:a100:1`, time, mem,
cpus) live in `conf/hydra/launcher/slurm.yaml`, mirroring the `sbatch` line in `sft1.sh`.

**Critical:** the compute node needs `conda activate hfmt` + `module load cuda/...` before
Python runs. Fold `install/path.sh` into the launcher's `setup:` commands so every job
bootstraps its env. Without this, jobs fail on the node (see Risk R2).

### 5. W&B ↔ output ↔ config linking
- Use Hydra's per-run output dir as the experiment output dir (checkpoints, preds, logs).
- Derive a deterministic W&B run **name** and **group** from the experiment name + Hydra
  job id; set `WANDB_RUN_ID` so re-runs and Slurm requeues map to the same run.
- Hydra already writes the fully-resolved config to `<run_dir>/.hydra/`; additionally log
  the (data-content-free) resolved config to W&B `config` for at-a-glance linkage.
- Reconcile the stubbed `--wandb_group`/`--wandb_tags` in `sweep_run.sh`: these move into
  `conf/wandb/` (the argparse flags never existed — see Risk R5).

### 6. Dependencies
Add `hydra-core`, `omegaconf`, `hydra-submitit-launcher` (pulls `submitit`), optionally
`hydra-colorlog` to `install/install_hf_custom.sh`; record tested versions in
`check_versions.py`. Consider adding a real `requirements.txt`/`environment.yaml` (none
exists today) so the dep set is declared in one place. GPU deps are unchanged. *(Build env
here now vs. update scripts only — Q3.)*

## Phased breakdown (→ becomes todo.md once narrowed)
1. Dependencies + env (install scripts, version pins, optional requirements file).
2. `conf/` skeleton + structured-config schema; compose/validate on CPU (`--cfg job`).
3. Port `sft_translation.py` to `@hydra.main`; wire output dir + W&B naming.
4. Surface remaining hardcoded knobs into config groups.
5. Slurm launcher config + `setup:` env bootstrap; dry-run job graph locally.
6. One `experiment/` preset reproducing today's `mmtc/fr-en/sft1.sh`.
7. Docs (README + CLAUDE.md); on-cluster validation checklist (GPU run must be user-run).
8. (Later request) Port remaining entry points; Hydra sweeper.

## Risks
- **R1 — No local GPU.** Cannot validate torch-cuda/bitsandbytes or a real run here; only
  config composition + dry runs. On-cluster smoke test is a user-run step.
- **R2 — Env bootstrap on compute node.** submitit runs Python on the node; `conda
  activate` + `module load` must happen there via `setup:`. Easy to miss → all jobs fail.
- **R3 — Hydra changes CWD.** By default Hydra chdirs into the run dir; the code uses
  relative paths (`HFMT_ROOT`, `egs/...`). Must set `hydra.job.chdir` and/or make paths
  absolute, or existing path assumptions break.
- **R4 — Scope creep.** "Isolate *all* tweakable params" pulls in magic numbers scattered
  across the code, which is edits beyond argparse. Bound it to the reference script first.
- **R5 — Existing sweep tooling drift.** `sweep_run.sh` passes flags the code lacks;
  `sweep.yaml` is a *W&B* sweep. Hydra has its own sweeper. Decide whether W&B-sweep
  tooling stays or is superseded — affects the W&B-linking design now (Q5).
- **R6 — Config vs. run dir collisions.** Naming scheme must guarantee unique dirs and
  stable W&B ids across requeues (Slurm preemption on `--time=48:00:00` runs).
- **R7 — Confidentiality regression.** The resolved config logged to W&B must carry only
  paths, never data content; guard when adding config logging.

## Open questions

All resolved — folded into Decisions D5 (W&B sweep tooling) and D6 (output + naming) above.
