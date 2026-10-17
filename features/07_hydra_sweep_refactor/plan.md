# Plan — 07_hydra_sweep_refactor

## Goal

Replace the toggleable inline sweep (`sweep: {enabled, params}` + `hfmt/sweep_probe.py`) with
**two kinds of experiment yaml**, using only vanilla Hydra:

- **single-run experiment**: no grid; `bash egs/run.sh <exp>.yaml` → exactly 1 job.
- **sweep experiment**: grid in Hydra's native `hydra.sweeper.params`; invoking it always
  runs the full cross product.

Success = fewer lines and fewer moving parts between `run.sh` and `main()`: the job-start path becomes
`run.sh → python -m hfmt.sft_translation --multirun` with no intermediate compose step.

## Decisions (from request + developer answers)

| Topic | Decision |
|---|---|
| Example single-run file | `egs/mmtc/fr-en/sft_lora.yaml` (from today's `sft1_lora.yaml`) |
| Example sweep file | `egs/mmtc/fr-en/sft_lora.sweep.yaml`: `train.learning_rate: 1e-4,2e-4` × `lora.lora_r: 8,16` = **4 jobs** |
| Other fr-en experiments | delete: `sft1.yaml`, `sft1_lora.yaml` (replaced), `sft1_fullft.yaml`, `sft1_fullft_modelsweep.yaml`, `sft1_lora_modelsweep.yaml`, `sft1_lora_model_lr_sweep.yaml`, `sft1_refine1.yaml` |
| `conf/hydra/launcher/slurm_fullft.yaml` | delete along with the fullft experiments |
| Experiment names | `mmtc_fr-en_sft_lora` / `mmtc_fr-en_sft_lora_sweep` (provisional; developer may rename later) |
| Telling the two kinds apart | no enforced convention; the file contents decide (the `.sweep` name is just the example) |
| Grid info before submit | rely on the submitit launcher's existing `#idx : overrides` log; no new code (§D) |
| Dataset fast-fail | keep, as a one-line shell check in `run.sh` (§E) |
| Untracked `egs/experiment/`, `data/`, `sft1_lora.yaml.old`, `slurm_h100/v100.yaml` | ignore; leave untouched |

Assumption: "`sft_loral.sweep.yaml`" in the answers was a typo for `sft_lora.sweep.yaml`.

## Verified behaviour (Hydra 1.3.7, scratch app laid out like the repo)

- A grid in `hydra.sweeper.params` **inside a `# @package _global_` experiment selected via
  `+group=option`** is honoured under `--multirun`; a file without one → 1 job.
- Config-group axes work (`basemodel: a,b`) despite the file's own `override /basemodel`.
- CLI comma-lists add axes; a single-value CLI override of a swept key pins that axis.
- A **dotted option name** works: `+mmtc/fr-en=sft_lora.sweep` composed and launched 4 jobs.
- Sweeper-generated overrides land in `hydra.overrides.task` → `override_dirname`, which is what
  `_label_pairs` / `build_run_label` read, so run labels and `is_sweep` should be unaffected.
- The submitit launcher already logs `Submitit 'slurm' sweep output dir : …` and one
  `#<idx> : <overrides>` line per job **immediately before** `executor.map_array` submits
  (`hydra_submitit_launcher/submitit_launcher.py:120-145`).
- Only *comma-list* values on `hydra.*` keys are rejected from the CLI; single values are fine
  (corrects the wording in `decisions.md:132-134`).

## Proposed design

### A. `egs/run.sh`
Remove the probe block (`run.sh:76-95`: `probe_overrides` loop + `sweep_axes=$(…)`) and the
`${sweep_axes}` arg. What remains: arg check → env → split path → experiment-file check →
dataset-template check (§E) → launch. Rewrite the header comment to describe the two kinds
of experiment and drop all `sweep.enabled` examples. Expected: ~110 → ~70 lines.

### B. Python: remove the sweep machinery
- Delete `hfmt/sweep_probe.py` (114 lines).
- `hfmt/hydra_config.py`: delete `SweepConfig` and the `sweep` field on `Config`; refresh the
  module docstring (also fix the stale `egs/experiment/` mentions).
- `hfmt/sft_translation.py` `_label_pairs` (`:311`): drop the `"sweep"` / `"sweep."` filters,
  keep `experiment` + `hydra*`. Update the `build_run_label` docstring to match.
- `conf/config.yaml`: drop `sweep` and `sweep.enabled` from
  `hydra.job.config.override_dirname.exclude_keys`; refresh comments (stale
  `+experiment=` / `egs/experiment/` mentions, "plain run" wording).

Note: after this, passing `sweep.enabled=true` on the CLI fails schema validation
("Key 'sweep' not in 'Config'"). That is the intended loud failure for old habits.

### C. Experiments in `egs/mmtc/fr-en/`
`sft_lora.yaml`: content of today's `sft1_lora.yaml` with `experiment: mmtc_fr-en_sft_lora`,
`wandb.group: mmtc_fr-en_sft_lora`, header comment updated.

`sft_lora.sweep.yaml`: same components/data/wandb, `experiment: mmtc_fr-en_sft_lora_sweep`, plus:
```yaml
hydra:
  sweeper:
    params:
      train.learning_rate: 1e-4,2e-4
      lora.lora_r: 8,16
```
Each sweep file is self-contained: no inheritance from the single-run file, so it can be
read without opening a second file. The cost is some duplicated lines (see R4).

### D. Grid info before submitting (decided: existing launcher log)
**Rely on the launcher's existing log**, i.e. zero new code. The submitit launcher already
prints the sweep dir and every job's overrides right before submission, e.g.
```
Submitit 'slurm' sweep output dir : egs/mmtc/fr-en/outputs/mmtc_fr-en_sft_lora_sweep/sweep-<ts>
	#0 : train.learning_rate=1e-4 lora.lora_r=8 +mmtc/fr-en=sft_lora.sweep ...
	#1 : ...
```
The task is only to confirm it appears on the console under `run.sh`, and to document it.

Rejected alternatives: D2, a `sed` echo of the yaml's grid in `run.sh` (YAML parsing in shell, blind to CLI
axes); D3, a `--dry-run` via `--cfg hydra` (a second Python start-up, which is what this feature removes).

### E. Login-node fast-fail for a missing dataset template
Replace the probe's check with a shell test in `run.sh`:
`[ -f "${dataset_dir}/dataset.yaml" ] || { echo …; exit 1; }`. It checks only the **default**
filename. An experiment that sets `data.dataset_file: other.yaml` would bypass the shell check,
but `main()` still validates it authoritatively (`resolve_dataset_template`). See R3.

### E2. Delete the `slurm_fullft` launcher
Delete `conf/hydra/launcher/slurm_fullft.yaml`, and remove its mentions in `egs/run.sh` (the launcher
comment, "or a preset like slurm_fullft"), `conf/config.yaml` (defaults comment), and `README.md`.
`decisions.md` keeps its historical mention but gets a note in the feature-07 entry. A full
fine-tune remains possible via `train.finetune_mode=full` with launcher resources overridden on
the CLI (`hydra.launcher.mem_gb=… hydra.launcher.timeout_min=…`). Copy the values from
`slurm_fullft.yaml` into the README snippet before deleting it.

### F. Docs, comments, tests
- `README.md` "Sweeping parameters" (≈`:91-126`) and the experiment bullet (`:47`): rewrite
  around the two kinds of file; keep ad-hoc CLI sweeps, the cross-product warning, and the
  output/W&B naming text; replace `sft1*` examples (≈10 refs).
- `CLAUDE.md`: the parameter-sweeps bullet (`:60-67`), the configuration bullet (`:55-58`, stale
  `egs/experiment/` + `mmtc_fr-en_sft1` usage), and the tree comment (`:39`).
- `decisions.md`: a new feature-07 entry (why the toggle was dropped) and a note that it
  supersedes the "Sweep toggle via a compose probe" decision, including the CLI-rejection correction.
- `tools/run_on_free_gpu.sh`: usage examples (`:8,14`) use `mmtc_fr-en_sft1 sweep.enabled=true`
  → `mmtc/fr-en/sft_lora.sweep.yaml`. Comments only; the wrapper forwards args unchanged.
- Launcher yaml usage comments (`conf/hydra/launcher/local.yaml:2`, `slurm.yaml:2-4`) → current
  `run.sh` form. **`slurm.yaml` has your uncommitted edits**, so only the comment lines are touched.
- `hfmt/test_run_label.py`: the `sweep=coarse,sweep.enabled=true` drop test (`:86-87`) → keep
  the `experiment`/`hydra*` cases, remove the sweep tokens; refresh `mmtc_fr-en_sft1*` fixture
  names only if they hurt clarity (they are opaque strings; optional).

### What stays unchanged
Ad-hoc CLI sweeps on any experiment; `--multirun` always; `hydra.sweep.dir/subdir` + `hfmt_runlabel`;
`derive_run_identity` / W&B grouping; output layout under `egs/<dataset>/<pair>/outputs/`;
confidentiality rules (labels hold only hyperparameters / config-file names).

## Risks

- **R1: run-label / W&B identity regression.** Verified in a scratch app, not yet in the repo
  with `_label_pairs` + `derive_run_identity`. A plain run must still be named by experiment
  (no `-sweep-<ts>` group); a sweep job must get `learning_rate-…__lora_r-…` and the sweep group.
  *Mitigation:* a local `hydra/launcher=local` smoke run (tiny `train.max_steps`) of both files on a GPU
  node before a real Slurm submit, and inspect the output dirs and W&B names.
- **R2: label order.** The sweeper puts config-file axes *before* CLI overrides (demo:
  `model=small,lr=0.01,+mmtc/fr-en=grid`). That matches today: `run.sh` passes the probe's axes
  before `"$@"`, and the selector is dropped from labels by the `/`-in-key rule. So no change is expected;
  listed only because it is confirmed by reading the code, not by a repo run (covered by R1's smoke test).
- **R3: weaker fast-fail.** The shell check only covers `dataset.yaml`. Low impact today (every
  experiment uses the default), but it's a silent gap if someone sets `dataset_file`.
- **R4: duplication between single-run and sweep files.** Data, wandb, and component choices are
  repeated, so they can drift. Accepted for readability; alternatively the sweep file could list
  the single-run file in its `defaults:` (`- /mmtc/fr-en/sft_lora@_global_`). That's fewer lines but
  one more indirection, which works against the goal.
- **R5: losing examples.** Deleting `sft1_fullft*` and `slurm_fullft.yaml` removes the only in-repo
  example of `train.finetune_mode=full` with its larger launcher resources, and of a config-group
  axis (`basemodel: a,b`). The README will need to carry those as snippets, including the fullft
  `mem_gb`/`timeout_min` values.
- **R6: broken muscle memory and external scripts.** Any personal scripts or notes using
  `sft1.yaml` or `sweep.enabled=true` break. The schema error makes this loud, not silent.
- **R7: git history / old outputs.** Existing `egs/mmtc/fr-en/outputs/mmtc_fr-en_sft1*` dirs
  (gitignored) stay on disk under the old experiment names; nothing reads them, so no action is needed.

## Open questions

None outstanding. All five were resolved by the developer and folded into the Decisions table
(grid values OK; keep the shell fast-fail; launcher log is enough; delete `slurm_fullft`;
experiment names OK for now).

## Validation

1. `pytest hfmt/test_run_label.py hfmt/test_dataset_inflation.py` passes.
2. `grep -rn "sweep_probe\|sweep\.enabled\|SweepConfig\|sweep\.params"` over tracked files
   (excluding `features/0[0-6]`) returns nothing.
3. Config-only: `python -m hfmt.sft_translation +mmtc/fr-en=sft_lora data.dataset_dir=… --cfg job`
   composes cleanly, and `--cfg hydra -p hydra.sweeper.params` on the sweep file shows the 2 axes.
4. Smoke (GPU node, `hydra/launcher=local`, tiny `train.max_steps`): the single-run file → 1 job dir
   named `mmtc_fr-en_sft_lora`; the sweep file → 4 job dirs named
   `learning_rate-…__lora_r-…`, with W&B names and groups as in README.
5. Real Slurm launch of the sweep file from a login node: the launcher prints the 4 `#idx` lines and a
   4-task job array is submitted.
6. `bash egs/run.sh mmtc/fr-en/sft_lora.yaml sweep.enabled=true` fails fast with a schema error.
7. Line-count delta reported (`git diff --stat`): expect a net reduction (~-200 lines incl. probe + yamls).

## Phases (todo.md groups)

- **A: experiments.** Create `sft_lora.yaml` + `sft_lora.sweep.yaml`; delete the 7 old yamls and
  `slurm_fullft.yaml`.
- **B: launcher path.** `run.sh` probe removal + shell fast-fail + header; delete `sweep_probe.py`.
- **C: schema + label code.** `SweepConfig` removal, `_label_pairs` filters, `exclude_keys`, tests.
- **D: docs.** README, CLAUDE.md, decisions.md, run_on_free_gpu.sh, launcher comments.
- **E: validation.** Steps 1–7 above.
