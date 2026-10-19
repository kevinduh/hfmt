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

---

## Config is atomistic components + experiments in egs/ (feature 04_config_structure_change)

**Decision:** `conf/` holds only atomistic, mix-and-match components — `basemodel`, `lora`,
`data`, `train`, `decode` — and nothing else. An *experiment* composes them (plus inline `wandb`
and `sweep` blocks) and lives under `egs/experiment/<name>.yaml`, discovered via
`hydra.searchpath`. The old `model` group is split into `basemodel` (checkpoint + quantization +
load `dtype`) and `lora` (adapter, with an `enabled` toggle); `conf/wandb/`, `conf/experiment/`
and `conf/sweep/` are removed. LoRA can be turned off (`lora.enabled=false`, requires
`basemodel.load_in_4bit=false`) to fully fine-tune the base model.

**Context:** Feature 00 put everything in `conf/`, including experiments and (feature 02) sweep
presets; the `model` group conflated the base model with the adaptation method. This feature
makes the base model and the adaptation independent components, moves experiments next to the
other recipes under `egs/`, and folds wandb/sweep into the experiment so one file fully describes
a run.

- **`hydra.searchpath` over `--config-dir`.** `conf/config.yaml` adds
  `file://${oc.env:HFMT_ROOT}/egs` to the search path, so `+experiment=<name>` resolves from
  `egs/experiment/` under *any* launcher (`run.sh`, direct `python -m`, the compose API) — not
  just when `run.sh` passes a flag. Verified on the pinned Hydra (1.3.7).
- **Sweep toggle via a compose probe, not `hydra.sweeper.params` on the CLI.** *(Superseded by
  feature 07: two kinds of experiment, no toggle/probe — see below.)* Hydra has no
  native on/off for the basic sweeper and *rejects* `hydra.sweeper.params.*` as a CLI override
  ("Sweeping over Hydra's configuration is not supported"). So the grid lives in a typed
  `sweep: {enabled, params}` block and `egs/run.sh` runs `hfmt/sweep_probe.py` (compose-only, no
  `transformers`) which, when `sweep.enabled`, emits bare `<axis>=<csv>` overrides the sweeper
  fans out. We always `--multirun` (submitit only runs in multirun); disabled → no axes → one
  job. The probe honors `sweep.enabled` from the CLI; adding an *axis* from the CLI uses a bare
  override (`basemodel=a,b`), not `sweep.params.*` (a comma value breaks single-run compose).
- **Sweep over whole component files; label by file name.** Axes may be group choices
  (`basemodel: "qwen2.5-7b,qwen2.5-1.5b"`), so `build_run_label` labels such a job
  `basemodel-qwen2.5-7b` — the selected YAML file, not the checkpoint value inside. A bare
  `data=<name>` config name is now allowed in labels (a name, not a path); dotted `data.<leaf>`
  and any `/`-bearing value are still dropped (confidentiality unchanged).
- **Full fine-tune support.** `sft_translation.py` makes quantization conditional on
  `basemodel.load_in_4bit` and the adapter conditional on `lora.enabled`, with a fail-fast
  guardrail when LoRA is off but the base is 4-bit. Full-FT inference needs no new code — load
  `<run_dir>/model` as `inf_translation.py --checkpoint` with no `--peft`.
- **Refinement sweep is now an experiment.** The old `conf/sweep/mmtc_fr-en_refine1.yaml` (which
  both fixed values and swept others) became `egs/experiment/mmtc_fr-en_sft1_refine1.yaml`,
  since "a refined sweep" is just another experiment in the new model.
- **Plain runs are named by the experiment, not `job0`.** `run.sh` always uses `--multirun`
  (submitit needs it), so a single non-swept run reached the sweep naming path and got the
  `job0` fallback label and a `-sweep-<ts>` W&B group. Now `main` distinguishes a real sweep
  job (its `override_dirname` carries informative tokens) from a plain run: a plain run falls
  back to the `<experiment>` name for the run-dir leaf and W&B run name (so disk and W&B still
  match), keeps the base W&B group without the `-sweep-<ts>` suffix, and drops the `sweep` tag.
  `${experiment}` is threaded into the `hfmt_runlabel` resolver for the dir leaf (verified it
  interpolates inside `hydra.sweep.subdir`). Real sweeps are unchanged.
- **Launcher is a selectable component, not a forced flag.** The Slurm settings (GPU type,
  walltime, memory, concurrency) live in `conf/hydra/launcher/` profiles — `slurm` (QLoRA
  baseline) and `slurm_fullft` (extends `slurm`, bumps RAM/walltime for full fine-tunes) — and
  each experiment *selects* one in its defaults (`- override /hydra/launcher: <profile>`), so the
  experiment stays a flat list of choices with the complex presets in their own files.
  `egs/run.sh` no longer hardcodes `hydra/launcher=slurm`. **Why not a primary-config default:**
  Hydra ignores an experiment's launcher group-override once the primary config sets one (verified),
  so defaulting it in `conf/config.yaml` would silently block per-experiment selection. The trade-off
  is a footgun — an experiment that selects no launcher falls back to the local `basic` launcher —
  so every committed experiment selects a slurm profile. Per-run tweaks still work on the CLI
  (`hydra.launcher.timeout_min=120`, `hydra/launcher=local`).

---

## W&B run names prefix the experiment (feature 05_wandb_run_name_prefix)

**Decision:** A sweep job's W&B run *name* is `<experiment>-<label>` (e.g.
`mmtc_fr-en_sft1-basemodel-qwen2.5-0.5b`), not the bare swept-param `<label>`. The run
*directory* leaf still keeps the bare label, so name and dir basename now differ by exactly
the `<experiment>-` prefix. This **loosens the feature 03 invariant** that the run-dir basename
*is* the W&B run name — on disk it still is; in W&B it is prefixed.

**Context:** Sweep run names collided across experiments. Any two experiments that sweep the
same axis (the `fullft`/`lora` model-sweeps both sweep `basemodel`) emitted byte-identical names
like `basemodel-qwen2.5-0.5b`, indistinguishable in W&B's flat run list even though their
*groups* differed. Plain runs were already prefixed (`<experiment>-<label>` since feature 04);
the sweep branch was the lone holdout.

- **One-line change, consistent with plain runs.** Only `derive_run_identity`'s `is_sweep`
  branch changed (`run_name = cfg.wandb.name or f"{exp}-{label}"`), mirroring the plain-run
  branch. The `cfg.wandb.name` manual override still wins.
- **`run_id` and grouping are untouched.** `run_id` already embedded the experiment, so
  requeue/`resume` behavior is unchanged; the group stays `<base-group>-sweep-<ts>`.
- **Confidentiality unchanged.** The experiment name is config, not data; the label is still the
  sole data-path chokepoint (`build_run_label`). No data content enters the name.
- **Trade-off.** Names get longer and the prefix is redundant inside a single group's view, but
  the flat cross-experiment list — the view that was broken — is now unambiguous.

---

## Dataset/lang directory hierarchy + inline data (feature 06_hydra_config_directory_hierarchy)

**Decision:** Experiments move from the flat `egs/experiment/<name>.yaml` into
`egs/<dataset>/<src>-<trg>/<name>.yaml`, each beside a per-lang-pair `dataset.yaml` **template**.
The `conf/data` config group is removed: an experiment now declares its data **inline**
(`data.data_root`/`src`/`trg`/`instruction`) and names the template (`data.dataset_file`, default
`dataset.yaml`); the template's `$DATA_ROOT`/`$SRC`/`$TRG` placeholders are **inflated in code**
into absolute `train`/`dev`/`test` paths. Run outputs move under `egs/<dataset>/<src>-<trg>/outputs/`.

**Context:** A flat `egs/experiment/` got cluttered across datasets and language pairs, and the
data config was a separate group file disconnected from the experiment that used it. Embedding
the dataset/lang in the directory and the data knobs in the experiment makes each recipe
self-contained and self-describing.

- **Nested-group addressing, unchanged searchpath (verified by a compose spike).** `egs/run.sh`
  takes the experiment's `egs`-relative `.yaml` path, splits it into the directory (a nested
  Hydra config group) and basename (the option), and selects `+<dataset>/<src>-<trg>=<name>`
  against the existing `hydra.searchpath: [file://$HFMT_ROOT/egs]`. No per-run searchpath
  rewrite; `run.sh` stays `dirname`/`basename` and depth-agnostic. **Why not keep a fixed
  `experiment` group:** that would force an `egs/experiment/<dataset>/<lang>/` path, not the
  requested `egs/<dataset>/<lang>/`. The one cost is that the selector key now contains `/`, so
  `_label_pairs` drops `/`-bearing keys (a one-line, symmetric extension of the existing
  `/`-in-value rule) to keep the structural selector out of run-dir/W&B names and the `is_sweep`
  check. `run.sh` injects `data.dataset_dir` (the experiment's own dir) and
  `output_root=egs/<group>/outputs`.
- **Template inflated in code, not via OmegaConf interpolation.** `dataset.yaml` stays a plain
  shell-`$VAR` template (not a Hydra config). Inflation parses the YAML **first**, then
  substitutes only in string *values* — so a stray `$`/`${...}` in a template comment can't break
  it (found and fixed during migration). Inflated paths are written back into `cfg.data` so the
  logged/resolved config carries real paths (paths are W&B-safe; only data *content* is forbidden).
- **`test` split is optional.** When the `test` block is absent or its source file(s) don't exist
  on disk, the end-of-run held-out inference pass is skipped (no error). `train`/`dev` stay
  required (`dev` drives `eval_loss`/early-stopping). The `evalset` terminology is retired in
  favour of `cfg.data.test.src`.
- **Validation fails fast.** The named template must exist in the experiment's directory
  (`dataset_file` a bare basename ending in `.yaml`); this is checked authoritatively at the start
  of `main()` and again as a cheap login-node fast-fail in `sweep_probe.py` (before the Slurm
  submit), so a missing/mistyped template aborts early rather than deep in data loading.
- **Confidentiality.** Outputs now live inside the tracked `egs/` tree, so `egs/**/outputs/` is
  gitignored (they hold `*.pred`/`eval.pred.trg` dumps). Only paths + hyperparameters + the
  instruction string reach W&B, never data content.
- **Scope.** Only the Hydra SFT workflow (`sft_translation.py`) is migrated; the other
  (unmigrated) entry points — seq2seq, inference, summarization — keep their plain-manifest/shell
  recipes.

## Two kinds of experiment instead of a sweep toggle (feature 07_hydra_sweep_refactor)

**Decision:** An experiment is either a **single-run** file (no grid → one job) or a **sweep** file
whose grid lives in Hydra's native `hydra.sweeper.params` and always runs in full (example pair:
`egs/mmtc/fr-en/sft_lora.yaml` / `sft_lora.sweep.yaml`). The feature-04 toggle — the typed
`sweep: {enabled, params}` block, `hfmt/sweep_probe.py`, and `sweep.enabled=true` — is removed.
This **supersedes** the feature-04 entry "Sweep toggle via a compose probe".

- **Why.** The probe existed only to give an on/off switch: `run.sh` must always `--multirun`
  (submitit), and Hydra can't switch off a config-stored grid from the CLI. With separate files the
  switch is unnecessary, so the launch path is now just `run.sh → python -m hfmt.sft_translation
  --multirun` with no intermediate compose. Goal: fewer lines and an easier-to-follow job start.
- **Verified on Hydra 1.3.7.** `hydra.sweeper.params` in a `# @package _global_` experiment selected
  via `+<dataset>/<pair>=<name>` is honoured (including config-group axes despite the file's own
  `override /basemodel`); a dotted option name (`sft_lora.sweep`) works; CLI comma-lists add axes and
  a single-value CLI override pins a swept axis; sweeper-generated overrides land in
  `override_dirname`, so run labels / W&B identity are unchanged.
- **Correction to the feature-04 rationale.** Hydra rejects only *comma-list* values on `hydra.*`
  keys from the CLI ("Sweeping over Hydra's configuration is not supported");
  `hydra.sweeper.params.<k>=<v>` with a single value is accepted. What the CLI really can't do is
  remove the grid: `~hydra.sweeper.params` errors and `hydra.sweeper.params={}` is ignored.
- **Trade-offs accepted.** Single-run and sweep files duplicate their data/wandb/component lines
  (chosen over inheritance for readability). The login-node dataset fast-fail moved from the probe
  to a one-line shell check in `run.sh` that only covers the default `dataset.yaml` name; `main()`
  still validates `data.dataset_file` authoritatively. Grid info before submission comes from the
  submitit launcher's existing per-job `#<idx> : <overrides>` log — no new code.
- **Pruned examples.** The fr-en experiments were reduced to the `sft_lora` pair; the
  `slurm_fullft` launcher was deleted. A full fine-tune is `train.finetune_mode=full` plus CLI
  launcher overrides (`hydra.launcher.mem_gb=64 hydra.launcher.timeout_min=720`), documented in the
  README.

## Hydra redone from scratch: sbatch launcher, run_name folders, saved configs (feature 08_hydra_redo)

**Decision:** On branch `hydra3` (from `main`), the SFT workflow's Hydra integration was rebuilt
in one feature, using the lessons of features 00–07 (kept above as history, on branch `hydra2`).
The entries above are **superseded** where they conflict: submitit and `--multirun` (00, 02),
run labels from `override_dirname` and the `hfmt_runlabel` resolver (03, 05), the
`hydra.searchpath` + nested-group selector and the sweep probe (04), and the `dataset.yaml`
template filled in by code (06). Feature 07's two kinds of experiment are kept.

- **Launch: `egs/sft.sh` → `hfmt/grid.py` → `sbatch --array` of `egs/job.sh`; no submitit.** The
  submitit launcher forced `--multirun` for every run, which caused the plain-run/sweep special
  cases, a login process that blocks until all jobs finish, at least three compositions per job,
  and `main` being pickled by value from `__main__` (so module code didn't re-run on the node,
  which hydra2's comments misdescribed). Now `grid.py` expands the grid on the login node with
  Hydra's own `BasicSweeper.split_arguments` (same rules as before: file axes + command-line
  comma lists, a single value pins), and each job is a plain single Hydra run. `LOCAL=1` runs
  the same jobs locally. Given up: submitit's results collection and resubmit-on-preemption.
  `grid.py` uses Hydra internals (`OverridesParser`, `BasicSweeper`); `hfmt/test_grid.py`
  covers it.
- **Every job's config is composed before submitting and saved fully resolved** as
  `<run folder>/run_config.yaml` (with `git_rev`). Typos and bad grid values fail on the login
  node; queued jobs run exactly the validated config even if the yaml sources are edited; a
  finished job can be re-run from its folder. Named `run_config.yaml`, not `config.yaml`: with
  `--config-dir <job dir> --config-name config`, Hydra silently loads `conf/config.yaml`
  instead (tested), so `sft.sh` also refuses experiments named `config.yaml`.
- **Run identity is just `run_name`** (required in the experiment yaml). Folders are
  `<output_root>/<run_name>` for one job and `<run_name>-<idx>` for a grid (suffix by job count),
  with `output_root` defaulting to `<experiment folder>/outputs`; the W&B run name is the folder
  name and the group is `run_name`. No timestamps: re-using a folder (e.g. continuing chosen grid
  jobs longer) only warns; changing `run_name` when a grid's shape changes is the user's
  responsibility. Hyperparameters are compared via W&B config and the saved configs, not names.
  This removed all of hydra2's label code, whose weakness was that Hydra can't tell a swept axis
  from any other override (a one-job run with an override was treated as a sweep).
- **Experiment file is Hydra's primary config** (`--config-dir <its folder> --config-name
  <stem>`); its defaults start with `- config` and `- dataset`. No searchpath, `@package
  _global_` or `+group=` selector. `dataset.yaml` (one per folder, not sweepable) is a Hydra
  config with `${data.*}` paths, replacing the template code and `dataset_dir`/`dataset_file`.
- **Every schema field is read by code.** hydra2 validated fields that nothing read
  (`lora.enabled`, `basemodel.dtype`, `bnb_4bit_compute_dtype`), so setting them silently did
  nothing. Fine-tuning modes (`train.finetune_mode: qlora|lora|full`) were ported from hydra2 as
  is; the `slurm:` block and `output_root` are read by `grid.py`.
- **W&B console capture is off.** Found while checking W&B calls: by default W&B uploads the
  run's stdout/stderr as `output.log`, which contained example text (a training example,
  decoded training batches) on hydra2 runs, and would on `main`'s `report_to="wandb"` runs too.
  `WANDB_CONSOLE=off` + `Settings(console="off")`; noted in CLAUDE.md.
- **Scope.** Only `sft_translation.py` uses Hydra; `inf_translation.py` and the other recipes keep
  their command-line arguments. Hydra inference from a saved job config is the planned next
  feature (09). Resume-from-checkpoint is deferred.

---

## Types-only schema, peft group, local data location, Hydra inference (feature 09_hydra_detailed_reorg)

**Decision:** Each config key has exactly one home, and the code is split so training and
inference share model handling.

- **The schema holds types only, no default values.** Every value field in `hfmt/hydra_config.py`
  is `MISSING`; optional blocks and keys are `Optional[...] = None` (absent). Every default lives in
  exactly one yaml: `conf/config.yaml` (defaults list; project-wide `slurm`, `wandb`, `inf_name`),
  `conf/<group>/<option>.yaml` (`train`, `peft`, `decode`), or the experiment folder. 08 gave every
  field a default *and* repeated most of them in `conf/`, so there were two sources of truth, and a
  yaml quietly overrode a misleading dataclass default. Now a key that no yaml sets fails in
  `grid.py`. Cost: default values are read in `conf/`, not in the schema.
- **`peft` is both the group and the key.** `peft/{qlora_default,lora_default,full_finetune}` set
  `peft.mode` plus optional `peft.lora` / `peft.quant` blocks (a full fine-tune has neither);
  `check_config` rejects a mode that disagrees with the blocks. This replaces the `basemodel` and
  `lora` groups and `train.finetune_mode`. The alternative (keep old key paths, `@package
  _global_` option files writing into three nodes) was rejected because the group name would no
  longer match a key. `checkpoint` and `instruction` became top-level experiment keys;
  `load_in_4bit` was kept, in `peft.quant`. Sweep/override keys changed (`lora.lora_r` →
  `peft.lora.lora_r`); 08 saved configs no longer validate and were deleted, not migrated.
- **Data location is a per-folder, untracked `local_setting.sh`** (`export HFMT_DATA_ROOT=...`),
  sourced by `egs/launch.sh sft` after unsetting the variable; `dataset.yaml` reads
  `${oc.env:HFMT_DATA_ROOT}` and also holds `src`/`trg` (fixed per folder, the same on every
  machine, so tracked). A single machine-wide data variable and an untracked Hydra paths group were
  considered; the per-folder file keeps tracked yamls generic and switches automatically with the
  experiment folder. Only needed at launch: saved configs hold resolved paths. `data_root/src/trg`
  are optional, so a dataset yaml may list absolute paths instead.
- **Inference runs on a trained run folder's saved config**, one folder per call:
  `egs/launch.sh inf <run folder>/run_config.yaml [overrides]` → `grid.py --inf` checks it and
  saves `<run folder>/inf/<inf_name>/inf_config.yaml` (with `model_dir`), never touching
  `run_config.yaml`. Accepting experiments and sweeps too (mapping them back to run folders) was
  planned and dropped as too much code for a shell loop's worth of convenience. `inf_translation.py`
  moved to Hydra in place; its old argparse CLI (used by the zh-en and synth_lrl recipes) is gone.
  No W&B in inference. One launcher, `egs/launch.sh <sft|inf>`, replaces `egs/sft.sh`; job scripts
  are `egs/sft_job.sh` (was `egs/job.sh`) and `egs/inf_job.sh`.
- **Inference reproduces training's test predictions exactly** (checked on full, lora and qlora:
  0/2000 lines differ). `hfmt/mt_model.py` holds model loading, prompting, decoding and scoring for
  both. Two causes of mismatch were found and fixed: (1) training leaves Accelerate's bf16 autocast
  wrapper on `model.forward` (it patches the model object in place and the Trainer never removes
  it), so full fine-tune test decoding ran under autocast; `sft_translation.py` now calls
  `accelerator.unwrap_model(model, keep_fp32_wrapper=False)` after training, before test decoding
  (dev decoding during training stays bf16). PEFT models were unaffected because
  `PeftModel.generate` bypasses the patched forward. (2) `prepare_model_for_kbit_training` casts a
  qlora base's non-quantized parameters to fp32; inference now applies it too
  (`use_gradient_checkpointing=False`). The model is saved before test decoding so a crash there
  doesn't lose it.

## Base-model inference, per-file test sets, merging, dev-decoding interval (feature 10_inference_more_options)

**Decision:** Inference, merging and training share one launcher, and the yaml you give it decides
what happens.

- **The yaml kind chooses the model for `launch.sh inf`.** A trained run's `run_config.yaml`
  decodes its `model/` into `<run folder>/inf/<inf_name>/`; an experiment yaml decodes its
  `checkpoint` as downloaded (`model_dir` unset) into `<output_root>/<run_name>/base_ckpt_inf/<inf_name>/`.
  A `peft=none` option was considered: more explicit in saved configs, but it needs its own
  precision rule and training would have to reject it. Instead the experiment's `peft` option sets
  the load precision. The separate `base_ckpt_inf/` folder means zero-shot output can't collide with
  a trained run's `inf/`, so nothing is refused. Sweep experiments and comma lists are refused.
- **Each test file is its own test set.** `data.test` is a `TestSet` (`trg` optional; a `null`
  entry = no reference). `mt_model.decode_test_sets` writes `test<i>.pred.trg` (i from 1, in list
  order, even for one file) and `scores.json` (paths + scores), and both training and inference
  call it, so inference still reproduces training exactly. Train/dev stay concatenated. Named
  test sets in `dataset.yaml` were considered; numbering kept the schema and the
  `'data.test.src=[...]'` override simple. Training now scores its test sets too and puts the
  numbers (no paths) in the W&B summary.
- **Merging goes through Slurm: `launch.sh merge <run>/run_config.yaml`** writes
  `<run>/model_merged/` (bf16 weights, tokenizer, `merge_info.yaml`). QLoRA merges onto the 4-bit
  base dequantized to bf16 (`dequantize()`), i.e. the weights the adapter was trained against,
  with a warning about the 4-bit rounding; merging into the original bf16 base gives better base
  weights but not the evaluated model. Measured: merged vs adapter within ~0.3 BLEU, 10-12% of
  lines differ (bf16 rounding). A second round is just an experiment with
  `checkpoint: <run>/model_merged`; nothing else changed. Training also saves the tokenizer in
  `model/`, so a full fine-tune's `model/` is a checkpoint as is.
- **One job script, `egs/job.sh <sft|inf|merge>`,** replaces `sft_job.sh`/`inf_job.sh` (and a third
  merge script): they differed only in the Python entry point.
- **`train.inference_callback_interval`** (N): the MT callback decodes dev + BLEU every N
  evaluations and at the last step (0 = never); early stopping runs every evaluation. With a
  BLEU-type `metric_for_best_model`, the Trainer checks that metric at *every* evaluation and raises
  if it's missing. So on skipped evaluations the callback runs early stopping first (it finds no
  value and leaves its patience counter alone), then sets a never-best placeholder (±inf) for the
  Trainer's best-checkpoint check. The placeholder never reaches W&B or `log_history`, because
  the Trainer logs eval metrics before callbacks run. Refusing BLEU + N > 1 was the alternative.
- **Saved configs aren't kept backward compatible.** A new required key (here
  `train.inference_callback_interval`) makes older `run_config.yaml` files fail `inf`/`merge`
  checks; the developer chose this over `Optional = None` keys or filling defaults on load.
- `grid.py` was split into `main_sft` / `main_inf` / `main_merge` with step comments, and
  `launch.sh` was commented block by block, since both had become hard to follow.

## LoRA target modules as explicit lists, short keys, model inspection (feature 11_hydra_lora)

**Decision:** LoRA layers are named, not chosen from presets, and a script shows the names.

- **`peft.lora.target_modules` is a required list of exact module names** (no `null`, no
  `all-linear`). It replaces `lora_target` (`qv | all-linear | attention | mlp`), which
  `sft_translation.py` mapped to names. That mapping was Qwen/Llama-specific and wrong even there:
  `attention` listed `out_proj`, but Qwen's attention output is `o_proj`, so it silently trained only
  q/k/v; `mlp` left out `gate_proj`. PEFT's built-in per-architecture default was rejected as the
  fallback: it exists for qwen2 but not for cohere2 (tiny-aya). Names aren't checked at launch;
  PEFT fails at job start on a name that matches nothing, and training logs what it wrapped.
- **Short keys:** `peft.lora.{r,alpha,dropout,target_modules}` (were `lora_r`, `lora_alpha`,
  `lora_dropout`, `lora_target`); the block is already `peft.lora`. No backward compatibility:
  older run configs and W&B columns keep the old names.
- **`tools/inspect_model.py <checkpoint>`** builds the model on the meta device (no weights, login
  node, seconds) and prints module names, LoRA candidates, parameters, config, chat-template
  overhead and a training memory estimate. The estimate uses fixed, printed equations (weights by
  mode; gradients 4T and AdamW 8T for T trainable parameters; checkpointed activations; logits
  14·b·s·V; CUDA context separately), not fitted; on four A100 runs it was within ±15% of peak
  allocated memory. Peak *reserved* memory, logged before, was 2–4× allocated and grows to fill
  larger GPUs, so training now also logs peak allocated. For large-vocabulary models (tiny-aya,
  262k) the logits term dominates.
