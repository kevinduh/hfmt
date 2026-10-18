# 08_hydra_redo — plan (DRAFT, for developer editing)

Status: **draft v1**. All open questions (Q1–Q13) are decided; see "Design" and "Open
questions".

## Request

Redo the Hydra integration for the SFT workflow **from scratch** on branch `hydra3`, which
starts from `main` (a clean slate before any Hydra work). Use the lessons learned on the
`hydra2` branch (features 00–07) and the code walkthrough / review discussion that followed
feature 07. The goal is to arrive at a simpler design up front instead of accreting it feature by
feature.

## Starting point (`main` @ 9a40c12 == `hydra3`)

- `hfmt/sft_translation.py`: 380 lines, argparse (`-t train.yaml -e evalset -c checkpoint -o outdir`,
  `--learning_rate`, `--qlora_r`, ...). QLoRA only. `report_to="wandb"`; logs aggregate
  `eval/bleu` etc. via an `EarlyStopping_MT_Callback`. Training data is read from a yaml manifest
  of `train`/`dev` → `src`/`trg` file lists; each side's files are loaded in order and the two
  sides joined line by line (`get_data`, lines 207–214).
- Recipes are shell scripts that `sbatch` themselves, e.g. `egs/mmtc/zh-en/sft1.sh` (hand-written
  `sbatch -p gpu --gres=gpu:h200:1 ...` comment, hyperparameters in a `cmdarg` string).
- No `conf/`, no `features/`, no `decisions.md`, no tests. `.gitignore` does **not** ignore
  `outputs/` or `egs/**/outputs/`.
- Env: hydra-core 1.3.7, omegaconf 2.3.1, Python 3.13.13 (already in the `hfmt` conda env).
  hydra-submitit-launcher 1.2.0 / submitit 1.5.4 are installed but won't be used (Q1).
- Working tree carries **untracked leftovers from hydra2** (`conf/hydra/launcher/slurm_{h100,v100}.yaml`,
  `egs/experiment/`, `egs/mmtc/fr-en/` incl. `outputs/` with prediction dumps, `data/`).
  These are not part of this branch's history. See Q11.

## Lessons from hydra2

hydra2 (`git diff main hydra2`: 23 files under `hfmt/ conf/ egs/ tools/`, **+1732 / −160**;
`sft_translation.py` 380 → ~750 lines, `hydra_config.py` 197, `run.sh` 92, ~400 lines of tests
mostly for run labels and dataset inflation). Feature history in the appendix.

### Keep
- **Composition from small mix-and-match components** (`basemodel`, `lora`, `train`, `decode`).
  An experiment is a short defaults list plus a few inline values. That's idiomatic Hydra, easy
  to read, and makes it easy to see how two experiments differ.
- **Typed dataclass schema (ConfigStore).** It rejects unknown keys and wrong types before a job
  is submitted.
- **Native sweep grids (feature 07).** A sweep experiment carries its grid in
  `hydra.sweeper.params` and always runs all of it; command-line comma lists add an axis, a
  single value pins one. No probe, no on/off toggle.
- **Thin launcher script** that only turns a yaml path into commands.
- **Confidentiality designed in.** Nothing data-bearing in run names, tags or W&B; config holds
  only hyperparameters, data *paths* and the instruction string.
- **Practical Slurm bits**: concurrency cap, a short time limit for backfill, GPU type
  overridable from the command line, compute-node env setup (`source install/path.sh`).

### Problems to avoid (from the post-07 review)

**1. Run naming built from `override_dirname` was the biggest source of complexity, and it's
brittle.** *(Resolved by Q2.)*
- The machinery: the `hfmt_runlabel` resolver, `_label_pairs`/`build_run_label`, `exclude_keys`,
  `is_sweep`, the three-way split in `derive_run_identity`, falling back to the experiment name,
  and dropping `/`-bearing keys. All of it exists to keep one rule: the folder name equals the
  W&B run name, and both describe the hyperparameters.
- The weakness: Hydra can't tell a swept axis from any other command-line override. So
  `bash egs/run.sh .../sft_lora.yaml lora.lora_r=16` (one job) counts as `is_sweep=True`: it gets
  a `-sweep-<ts>` group and a `sweep` tag. Smoke-run settings also leak into names
  (`...__max_steps-20` in the feature-07 E5 run).
- Labels come out sorted by key (Hydra sorts `override_dirname`), not in grid order.
- Better: let W&B do what it's good at. It already stores the full config, so you can filter and
  group by `lora.lora_r` directly.

**2. Always `--multirun` just to reach Slurm.** *(Resolved by Q1.)*
- The submitit launcher only works in multirun. So every run, even a single one, goes through
  the sweeper, gets a `sweep-<ts>` folder, and needs the plain-run vs sweep special cases.
- Further costs:
  - The login process blocks until every job finishes (run it in tmux).
  - Composition runs at least three times (login compose, per-job validation, per-job on the
    node).
  - `main` is copied over by value from `__main__`. On the node, `sft_translation.py`'s top
    level does **not** re-run: only `hfmt.hydra_config` is re-imported. The `base_config` schema
    and the OmegaConf resolvers arrive via Hydra's `Singleton.get_state()`. hydra2's comments got
    this wrong, and its "Entered main() −8.2s after process start" log was meaningless, because
    the start time travelled from the login process.
- This is the standard cost of `hydra-submitit-launcher`, and it's the root cause of lesson 1.

**3. Schema fields that are validated but ignored.** *(Rule for 08.)*
- The typed schema implies "if it's in the config, it's used". In hydra2 these were all
  accepted, type-checked, and silently ignored:
  - `lora.enabled`: never read; the docstring still called it the full-fine-tune switch.
  - `basemodel.dtype`: the load dtype was derived from `finetune_mode` and the hardware.
  - `basemodel.bnb_4bit_compute_dtype`: ignored; computed from bf16 support (a `todo` in code).
  - `basemodel.load_in_4bit`: only read in `qlora` mode.
- `lora.enabled=false` silently doing nothing is worse than an error.
- Rule for 08: **every schema field is read by code** (training code, or the launcher for the
  `slurm:` block and `output_root`). Derive quantization from `train.finetune_mode` rather than
  exposing `load_in_4bit` as a second knob that must agree.

**4. Two ways of filling in values for data paths.** *(Resolved by Q3.)*
- `dataset.yaml` used `$DATA_ROOT/$SRC/$TRG` placeholders filled in by `populate_data_paths`.
  On top of that, `run.sh` injected `data.dataset_dir` and the schema needed `dataset_file`; a
  ~180-line test covered the inflation.

**5. Selecting experiments as nested groups (`+mmtc/fr-en=sft_lora`) is clever but unusual.**
*(Resolved by Q4.)*
- It keeps experiments next to their data, which is good. But it needs the `hydra.searchpath`
  extension, `# @package _global_`, and special handling in the label code for keys containing
  `/`.

**6. Upkeep risk.** Hydra 1.3 gets little maintenance nowadays, and we're on Python 3.13. Keep
the Hydra dependency thin so it's easy to swap out later. Dropping the submitit plugin (Q1)
helps.

**Smaller lessons**
- Doc and comment drift was constant across 8 features: CLAUDE.md, README, yaml headers and
  docstrings described removed behaviour. This branch's CLAUDE.md already describes hydra2's
  feature-04 layout, which doesn't exist here (Q12).
- Test env: `pytest` is not installed in the `hfmt` env; tests ran standalone.
- `--cfg job` couldn't be combined with `--multirun`. Under the new design every job is a plain
  single run, so `--cfg job` always works.
- Outputs containing prediction dumps landed inside `egs/` and needed a `.gitignore` rule.

## Design

Aim: one obvious path from command to job, and a small Hydra footprint (≤ 400 added lines of
code+config, versus hydra2's ~1700; see Q10).

### Layout
```
conf/                          # base config + components only; never experiments
├── config.yaml                # defaults: base_config + default components; no hydra.searchpath
├── basemodel/  lora/  train/  decode/
egs/
├── sft.sh                     # launcher (renamed from hydra2's run.sh)
├── job.sh                     # one job on a compute node (or locally)
└── mmtc/fr-en/
    ├── sft_lora.yaml          # single-run experiment
    ├── sft_lora.sweep.yaml    # sweep experiment
    ├── dataset.yaml           # data file lists, loaded by Hydra
    └── outputs/               # run folders (git-ignored)
hfmt/
├── sft_translation.py         # @hydra.main
├── hydra_config.py            # typed schema
└── grid.py                    # login-node grid expansion + config saving
```

### Experiment files (Q4)
- Experiments live in `egs/<dataset>/<pair>/` and are launched by path:
  `bash egs/sft.sh egs/mmtc/fr-en/sft_lora.yaml [overrides]`. The path may be relative to the
  repo root, relative to the current dir, or absolute (`realpath`).
- The experiment file is Hydra's **starting (primary) config**: `sft.sh` passes
  `--config-dir <experiment's folder> --config-name <file stem>`. `--config-dir` only tells
  Hydra where the starting file is, for this run; nothing else in `egs/` is searched. No
  `hydra.searchpath`, no `# @package _global_`, no `+<group>=` selector. Prototyped on Hydra
  1.3.7.
- The file's `defaults:` begins with `- config` (pulls in `conf/config.yaml`), then `- dataset`
  (the `dataset.yaml` next to it), then `override <group>: <option>` lines, then `_self_`:
  ```yaml
  # egs/mmtc/fr-en/sft_lora.yaml
  defaults:
    - config
    - dataset
    - override basemodel: qwen2.5-1.5b
    - override train: qlora_default
    - _self_
  run_name: sft_lora
  data:
    data_root: ${oc.env:HFMT_ROOT}/data/fr-en
    src: fr
    trg: en
    instruction: "Translate French to English"
  train:
    seed: 37
  wandb:
    tags: [sft, fr-en, lora]
  ```
- Two kinds of experiment (feature 07, Q7): a single-run file has no grid; a `.sweep.yaml` has
  `hydra.sweeper.params` and always runs its full grid. The suffix is only a naming
  convention: `grid.py` just counts jobs. Command-line comma lists add an axis on either kind;
  a single command-line value pins an axis.
- **A sweep file is standalone**: it repeats the single-run file's defaults, data and W&B lines
  rather than extending it (Q7). Example:
  ```yaml
  # egs/mmtc/fr-en/sft_lora.sweep.yaml
  defaults:
    - config
    - dataset
    - override basemodel: qwen2.5-1.5b
    - override train: qlora_default
    - _self_
  run_name: sft_lora_sweep
  data:
    data_root: ${oc.env:HFMT_ROOT}/data/fr-en
    src: fr
    trg: en
    instruction: "Translate French to English"
  train:
    seed: 37
  wandb:
    tags: [sft, fr-en, lora]
  hydra:
    sweeper:
      params:
        train.learning_rate: 1e-4,2e-4
        lora.lora_r: 8,16
  ```
- Grid axes: leaf values (`train.learning_rate: 1e-4,2e-4`) or whole component files
  (`basemodel: qwen2.5-0.5b,qwen2.5-1.5b`; prototyped). The dataset is **not** an axis (see
  Data).

### Data (Q3)
- `dataset.yaml` is a Hydra config, loaded through the experiment's `- dataset` default (found
  because it sits in the `--config-dir` folder). Its paths use OmegaConf `${...}` references to
  the experiment's `data.*` values:
  ```yaml
  # egs/mmtc/fr-en/dataset.yaml
  # @package data
  train:
    src:
      - ${data.data_root}/train.${data.src}-${data.trg}.${data.src}
      - ${data.data_root}/extra.${data.src}-${data.trg}.${data.src}
    trg:
      - ${data.data_root}/train.${data.src}-${data.trg}.${data.trg}
      - ${data.data_root}/extra.${data.src}-${data.trg}.${data.trg}
  dev:
    src: [ "${data.data_root}/valid.${data.src}-${data.trg}.${data.src}" ]
    trg: [ "${data.data_root}/valid.${data.src}-${data.trg}.${data.trg}" ]
  test:
    src: [ "${data.data_root}/test.${data.src}-${data.trg}.${data.src}" ]
    trg: [ "${data.data_root}/test.${data.src}-${data.trg}.${data.trg}" ]
  ```
  Prototyped: composes to absolute paths, type-checked as `List[str]` by the schema.
- **Multiple files per split**: each side's list is loaded in order into one long column, and
  the two sides are joined line by line into one dataset, as `main` does today.
- **One `dataset.yaml` per experiment folder** (Q7): simplest; it can't be overridden from the
  command line or swept (`- dataset` is a single file, not a component group). A different file
  in the same folder is still possible by editing the experiment's defaults (`- dataset_v2`).
  Moving to a `dataset/<name>.yaml` group later (prototyped: then `dataset=big` works on the
  command line and as an axis) is an easy additive change.
- A missing `dataset.yaml` fails at compose time ("Could not find 'dataset'"), on the login
  node, before submitting.
- No `data_yaml`/`dataset_dir`/`dataset_file` keys, no template-filling code, no `exp_dir`.
- Lists are replaced, not merged: overriding `data.train.src` (experiment or command line)
  replaces the whole list.

### Launch (Q1)
- **`egs/sft.sh` calls `sbatch` directly; no submitit.** Steps:
  1. Resolve the experiment path; split it into folder and file stem.
  2. Run `hfmt/grid.py` on the login node (below).
  3. Print the grid (`idx: overrides`) and any "folder exists" warnings.
  4. `sbatch --array=0-$((n-1))%<array_parallelism> <flags from slurm:> egs/job.sh <job list>`
     and return right away. With `LOCAL=1`, run the jobs sequentially instead (below).
- **`hfmt/grid.py`** (login node):
  - Composes the config once from the non-comma overrides, reads `hydra.sweeper.params`, and
    reuses Hydra's `BasicSweeper.split_arguments` for the cross product, so the grid rules are
    Hydra's own. A single-run file gives 1 job, a `.sweep.yaml` its full grid. Prototyped: sweep
    file → 4 jobs; `r=32 seed=1,2` pins `r` and adds a `seed` axis.
  - **Builds every job's config before submitting**, so a bad grid value or typo fails on the
    login node.
  - Decides each job's output folder (see "Output folders") and **saves** the job's config there
    as **`run_config.yaml`**, **fully resolved** (absolute paths, no `${...}` left; easier to debug;
    portability isn't needed: single user, one cluster). A `defaults: [base_config, _self_]`
    header keeps schema validation when the file is reused. The `hydra:` section (and so the
    sweep grid) is not saved. Prototyped: running from the saved file works, overrides apply
    on top, and a typo is rejected (`Key 'sed' not in 'Cfg'`).
  - **Why not `config.yaml`:** tested: with `--config-dir <job_dir> --config-name config`, Hydra
    silently loads `conf/config.yaml` instead of the job's file (same name, `conf/` wins). So the
    saved file is `run_config.yaml`, and `sft.sh` refuses an experiment file named `config.yaml`.
  - Writes the git commit next to each saved config (`<job_dir>/git_rev`), so a later re-run on
    changed code is detectable.
  - Writes `<output_root>/<run_name>.grid.txt` (one `idx: overrides` line per job).
  - Prints the `slurm:` block as sbatch flags.
- **`egs/job.sh`** (~6 lines): `source install/path.sh`, picks its job folder from
  `$SLURM_ARRAY_TASK_ID`, then runs
  `python -m hfmt.sft_translation --config-dir <job_dir> --config-name run_config hydra.run.dir=<job_dir>`.
  That's a plain single Hydra run of exactly the config validated at submit time; edits to the
  yaml sources while jobs are queued don't affect them.
- **`slurm:` block** in the config (partition, gres, time, mem_gb, cpus, array_parallelism).
  Defaults in `conf/config.yaml`; overridable per experiment and from the command line
  (`slurm.gres=gpu:l40s:1`). Read by `grid.py`/`sft.sh`, not by training code.
- **Local, without Slurm:**
  - A whole experiment or grid: `LOCAL=1 bash egs/sft.sh egs/mmtc/fr-en/sft_lora.sweep.yaml`.
    Loops over the jobs sequentially, calling `SLURM_ARRAY_TASK_ID=$i bash egs/job.sh ...`.
    Same folders and W&B naming as Slurm; `slurm:` is ignored. Parallel multi-GPU local runs
    are deferred.
  - One configuration, directly: `python -m hfmt.sft_translation --config-dir egs/mmtc/fr-en
    --config-name sft_lora [overrides]`. For a sweep file this runs the file's base values
    (verified: a single run ignores `hydra.sweeper.params`). Output goes to Hydra's default
    folder unless `hydra.run.dir=...` is passed (Q13).
- Given up with submitit: the blocking wait and results collection, and its checkpoint/resubmit
  (which restarted training from scratch anyway; `sbatch --requeue` covers the same).

### Output folders and run identity (Q2)
- The experiment yaml sets **`run_name`**, which is required (`MISSING` in the schema).
- **`output_root`**: the folder that holds run folders. If neither the experiment nor the
  command line sets it, `grid.py` sets it to `<experiment's folder>/outputs`, e.g.
  `egs/mmtc/fr-en/outputs`. It's saved resolved in each job's config.
- Job folders are decided in **one place, `grid.py`** (login node, before submitting), and handed
  to each job as `hydra.run.dir=<job folder>` by `job.sh`. Every run launched through `sft.sh`
  (Slurm, `LOCAL=1`, re-runs from a saved config) follows this rule:
  - single run: `<output_root>/<run_name>/`, e.g. `egs/mmtc/fr-en/outputs/sft_lora/`
  - grid job: `<output_root>/<run_name>-<idx>/`, e.g. `egs/mmtc/fr-en/outputs/sft_lora_sweep-3/`;
    `idx` is `0..n-1`, zero-padded only when the grid has more than 10 jobs.
  - The suffix depends on the **job count n**, not the kind of file (Q7): n = 1 → no suffix,
    n > 1 → `-<idx>`. E.g. `sft_lora.yaml lora.lora_r=8,16` → `sft_lora-0`, `sft_lora-1`; a sweep
    file with every axis pinned → `sft_lora_sweep`.
  - An index is only a position in the grid: changing a grid's shape (pinning or adding an
    axis) makes indexes point to different configurations than an earlier launch with the same
    `run_name`. `grid.py` only warns that the folder exists; **changing `run_name` when changing
    a grid's shape is the user's responsibility** (Q7).
- **Direct Python runs** (by hand, without `sft.sh`, for debugging) are not covered by the
  rule: Hydra uses its built-in default folder (`outputs/<date>/<time>/` in the current dir)
  unless `hydra.run.dir=...` is passed. `conf/config.yaml` sets no `hydra.run.dir` (Q13).
- No timestamps. An existing folder → **warning, then proceed**. Re-using a folder is an intended
  workflow (short grid first, then continue chosen jobs longer). Resume-from-checkpoint is
  **deferred to a later feature**; until then a re-launch into an existing folder restarts
  training.
- **W&B**: run name = the job folder's basename (`sft_lora`, `sft_lora_sweep-3`); group =
  `run_name`; run id left to W&B. Name clashes in W&B are acceptable for now. Hyperparameters
  are compared via W&B config columns and each job's saved `run_config.yaml`.
- Removes all of hydra2's naming code: `hfmt_runlabel`, `build_run_label`/`_label_pairs`,
  `exclude_keys`, `is_sweep`, `derive_run_identity`'s three cases, and `test_run_label.py`.

### Re-running from a saved config
- `bash egs/sft.sh egs/mmtc/fr-en/outputs/sft_lora_sweep-1/run_config.yaml train.seed=1 run_name=sft_lora-seed1`
  → a single job in `egs/mmtc/fr-en/outputs/sft_lora-seed1/`, since the saved config already
  holds the resolved `output_root`.
- **Without a new `run_name`, a saved config re-runs in its own folder** (the "continue"
  workflow, once resume exists). Rule in `grid.py`: when the given file is a `run_config.yaml`
  and the command line doesn't set `run_name`, the job folder is the folder the file is in.
  Needed because a grid job's saved `run_name` is the grid's (`sft_lora_sweep`), not the
  folder's (`sft_lora_sweep-1`), so the normal rule would pick a different folder.
- Or directly: `python -m hfmt.sft_translation --config-dir <job_dir> --config-name run_config
  [overrides] hydra.run.dir=<folder>`.
- Changes to the data files themselves between runs are out of scope.

### Fine-tuning modes (Q5)
- Port from `hydra2:hfmt/sft_translation.py` **as is** (the "Load Base Model" and "Apply
  LoRA/QLoRA if needed" blocks, ~lines 550–615): `train.finetune_mode: qlora | lora | full`.
  - `full`: unquantized, float32, no adapters.
  - `lora`: unquantized, bf16/fp16, LoRA adapters.
  - `qlora`: 4-bit (bitsandbytes) + LoRA adapters, as on `main`.
  - Compute dtype bf16 vs fp16 chosen from GPU support (`torch.cuda.is_bf16_supported`); the
    trainer's `bf16`/`fp16` flags follow it.
- Only these blocks are ported; the rest of hydra2's `sft_translation.py` (naming code,
  `derive_run_identity`, `populate_data_paths`, submitit-related comments) is what this
  redo replaces.
- Schema fields follow the ported code (lesson 3): `train.finetune_mode`; `lora.{lora_r,
  lora_alpha, lora_dropout, lora_target}` (read in `lora`/`qlora`); `basemodel.{checkpoint,
  load_in_4bit, bnb_4bit_use_double_quant, bnb_4bit_quant_type}` (the `load_in_4bit`/`bnb_*`
  fields are read only in `qlora`). Left out because hydra2 never read them: `lora.enabled`,
  `basemodel.dtype`, `basemodel.bnb_4bit_compute_dtype`, `basemodel.pretrain`.
- Not ported: hydra2's `logging.info("Trainer fp16:", trainer.args.fp16)` lines (wrong call
  format; caused the `--- Logging error ---` in the feature-07 E5 node logs). Use `%s` format.
- Inference for `full`-mode models: `inf_translation.py` help text is ported (Q6 #9). Note it
  always loads a 4-bit base when given an adapter, which mismatches `lora`-mode jobs; fixed in
  09 (Q8).

### Ported from hydra2 (non-Hydra) (Q6)
In `sft_translation.py`:
1. Test set from `cfg.data.test` (via `dataset.yaml`); end-of-run test decoding writes
   `<job_dir>/eval.pred.trg`; skipped when there's no test set. Replaces
   `main`'s required `-e evalset`.
2. Decoding settings in config: `decode` component (`max_length`, `max_new_tokens`, `num_beams`,
   `do_sample`, `batch_size`), used by dev decoding and test decoding. `main` hard-codes them.
3. Early-stopping / best-model settings in config (`train`: `early_stopping_patience`,
   `early_stopping_threshold`, `metric_for_best_model`, `greater_is_better`,
   `load_best_model_at_end`). `main` hard-codes patience 100 / threshold 0.05.
4. Model saved to `<job_dir>/model` (`main` saves to a sibling folder `<outdir>_b`).
5. Own `wandb.init` (see W&B below), without hydra2's `id=` / `resume="allow"` (run ids are
   random, Q2c).
6. `hfmt/__init__.py` (empty) so the code runs as `python -m hfmt.sft_translation` and
   `grid.py` can import `hfmt.hydra_config`.
7. Heavy imports (torch, transformers, peft, trl, datasets) inside `main()`, so `--cfg job` and
   schema imports are fast on a login node without touching the GPU stack.

Not ported: #8, hydra2's `_startup_log` timing lines.

Separate files:
9. `hfmt/inf_translation.py`: hydra2's help-text change explaining the two load modes
   (adapter: `-c <base> -p <job_dir>/model`; full fine-tune: `-c <job_dir>/model`, no `-p`).
11. `tools/gpu_free.sh` (Justin Rokisky): ported as is.
12. `tools/run_on_free_gpu.sh` (Justin Rokisky): ported with two changes: it calls
    `egs/sft.sh` and appends `slurm.gres=gpu:<type>:1` instead of `hydra.launcher.gres=...`.

Deferred: #10, `egs/mmtc/fr-en/inf.sh` (assumes the old layout; rewritten in 09, Q8).

### W&B
- Own the run in `main()`; log the resolved (data-free) config and aggregate metrics only;
  `WANDB_LOG_MODEL=false`. Re-check every W&B call against CLAUDE.md.

## Open questions (all decided)

Q5. **Fine-tuning modes.** ✅ **Decided: port hydra2's implementation as is** (see "Fine-tuning
modes" in Design).

Q6. **Non-Hydra changes on hydra2.** ✅ **Decided** (see "Ported from hydra2 (non-Hydra)" in
Design): port #1–7, #9, #11–12; drop #8 (startup timing logs); defer #10 (`inf.sh`) to Q8.

Q7. **Sweep semantics.** ✅ **Decided:** feature 07's model as is; folder suffix by job count
(n = 1 → none); no config-diff in the "folder exists" warning (user's responsibility to change
`run_name`); sweep files are standalone (no extending the single-run file); one `dataset.yaml`
per folder, not sweepable.

Q8. **Scope.** ✅ **Decided (c):** only `sft_translation.py` moves to Hydra in 08.
`inf_translation.py` keeps its command-line arguments (plus Q6 #9's help text); the LRL and
zh-en inference recipes are untouched. Hydra inference from a saved job config is planned as
the next feature, 09 (see "Out of scope / deferred").

Q9. **Feature breakdown.** ✅ **Decided:** one feature 08, with phases A–E (see "Phases"),
each validated before moving on. Feature 09 stays Hydra inference.

Q10. **Simplicity budget.** ✅ **Decided:** ≤ 400 added lines as a soft acceptance criterion,
measured as `git diff --shortstat main -- hfmt conf egs/sft.sh egs/job.sh` excluding tests
(`hfmt/test_*.py`). Not counted: tests, docs, `features/`, `tools/`, example experiments under
`egs/<dataset>/`. Ported code (fine-tuning modes, Q6 items) counts. Rough estimate: schema ~120,
`conf/` ~60, `grid.py` ~60, `sft.sh` ~40, `job.sh` ~10, net `sft_translation.py` change
~+60–100. If the final count exceeds 400, the commit message says why.

Q11. **Untracked hydra2 leftovers.** ✅ **Decided: ignore them.** Claude doesn't read, move,
build on or commit `conf/hydra/launcher/slurm_{h100,v100}.yaml`, `egs/experiment/`,
`egs/mmtc/fr-en/sft1_lora.yaml.old`, the old `egs/mmtc/fr-en/outputs/` or `data/`. Notes:
- Phase A creates tracked files in the existing `egs/mmtc/fr-en/` folder (`sft_lora.yaml`,
  `dataset.yaml`); no name clashes with the leftovers.
- The example's `data.data_root` is `${oc.env:HFMT_ROOT}/data/fr-en`, i.e. it *reads* the
  untracked `data/` folder at run time, as in hydra2.
- `conf/` already exists untracked (the two launcher files). The new tracked `conf/` files sit
  beside them; `git add` must name the new files explicitly rather than `git add conf/`.

Q12. **Docs.** ✅ **Decided:**
- **CLAUDE.md**: edited in phase D to describe the 08 design (config, experiments, `egs/sft.sh`,
  sweeps, output folders, saved configs).
- **`decisions.md`**: copied from hydra2 as is (`git show hydra2:decisions.md`, features 00–07,
  kept as history), then a new 08 entry appended summarizing the key decisions and why. The
  entry notes which hydra2 decisions 08 supersedes (submitit, run labels, searchpath/nested
  groups, template inflation, sweep toggle) rather than editing the old entries.
- **README**: a short section on running SFT experiments (`egs/sft.sh`, single vs sweep,
  `LOCAL=1`, output folders, re-run from a saved config).

Q13. **Output folders.** ✅ **Decided (a):** single runs go to `<output_root>/<run_name>/`, grid
jobs to `<output_root>/<run_name>-<idx>/`, decided only by `grid.py` for runs launched through
`sft.sh`. `output_root` defaults to `<experiment's folder>/outputs`. Direct Python runs use
Hydra's default folder unless `hydra.run.dir=...` is passed; no `hydra.run.dir` in
`conf/config.yaml`.

## Risks

- R1. **Re-learning hydra2's pitfalls.** Mitigation: the lessons above are acceptance criteria
  (no dead schema fields; no name-parsing of `override_dirname`; one composition path).
- R2. **Dropping submitit** loses its checkpoint/resubmit and results collection. Concurrency
  cap and requeue are covered by `sbatch --array=...%N` and `--requeue`; failures are checked
  with `sacct`/logs.
- R3. **`grid.py` relies on Hydra internals** (`BasicSweeper.split_arguments`, `OverridesParser`).
  These are stable in 1.3 but not public API; pin hydra-core and cover `grid.py` with a small
  test.
- R4. **Confidentiality.** A fresh W&B integration must be re-checked against CLAUDE.md. `main`
  already uses `report_to="wandb"`; make sure no sample text or tables reach W&B, and that
  `egs/**/outputs/` is git-ignored *before* the first run (prediction dumps are data-derived).
- R5. **Behaviour parity.** Hyperparameter defaults must match `main`'s argparse defaults or the
  recipe values on purpose. E.g. `main` defaults to `lr 2e-5`, `max_steps 10000`, `linear`
  schedule; `sft1.sh` uses `2e-4`, `5000`, `reduce_lr_on_plateau`. Decide which become the
  component defaults.
- R6. **Hydra 1.3 maintenance** on Python 3.13 (lesson 6).
- R7. **No timestamps → shared folders.** A second launch of the same `run_name` reuses its
  folders and only warns. Risks: mixed logs/predictions from two launches in one folder, and a
  restart from scratch until checkpoint resume is implemented.
- R8. **Index reuse.** Re-launching a `run_name` with a different grid shape puts different
  configurations into existing `-<idx>` folders with only a "folder exists" warning. Accepted;
  documented as the user's responsibility (Q7).

## Out of scope / deferred

- Resume from checkpoint, and keeping config history when a folder is re-used (later feature).
- **Feature 09 (next): Hydra inference from a saved job config** (Q8):
  `python -m hfmt.inf_translation --config-dir <job_dir> --config-name run_config [overrides]`,
  e.g. another test set (`'data.test.src=[...]'`) or `decode.num_beams=5`. One shared
  `load_model(cfg)` for training and inference so the fine-tuning mode always matches (fixes the
  4-bit-on-`lora` mismatch); outputs under the job folder; Slurm submission; `egs/mmtc/fr-en/inf.sh`
  rewritten. 08 needs no changes for this: saved configs plus `<job_dir>/model` hold everything.
- Migrating the seq2seq, summarization, LRL and zh-en recipes, and the existing
  `inf_translation.py` command-line interface they use.
- Optuna or other non-grid sweepers.
- Parallel multi-GPU local runs.

## Phases

One feature (Q9); one commit per phase (`08_hydra_redo-<phase>: ...`). Each phase ends with its
own check before the next starts.

| Phase | Content | Done when |
|---|---|---|
| A | Schema (`hfmt/hydra_config.py`); `conf/config.yaml` + `basemodel/ lora/ train/ decode/` components; `sft_translation.py` on `@hydra.main` with data from `cfg.data`; fine-tuning modes ported from hydra2 (Q5); Q6 items 1–4, 6–7, 9; `.gitignore` (`egs/**/outputs/`); `egs/mmtc/fr-en/sft_lora.yaml` + `dataset.yaml` | `--cfg job` on a login node prints the expected config; a direct 20-step run on a GPU node finishes, with predictions and `model/` in the run folder |
| B | `hfmt/grid.py` (grid, build + resolved save of every job's config, `git_rev`, `.grid.txt`, folder-exists warning); `egs/sft.sh` (path resolution, sbatch array, `LOCAL=1`); `egs/job.sh`; `slurm:` block; `egs/mmtc/fr-en/sft_lora.sweep.yaml` | `LOCAL=1` single run; Slurm single run + 4-job grid (20 steps) land in the right folders; re-run from a saved config works; a typo in a grid value fails before submitting |
| C | W&B (own the run; name = job folder basename, group = `run_name`; Q6 item 5); GPU tools (Q6 items 11–12) | W&B shows the right names/groups and only config + aggregate metrics, no text (confidentiality check, CLAUDE.md); `tools/run_on_free_gpu.sh` submits via `sft.sh` |
| D | Docs: CLAUDE.md; `decisions.md` copied from hydra2 + 08 entry appended; README section on running SFT experiments; tests for `grid.py` | Docs match the code; tests pass (standalone, since `pytest` isn't installed); line budget checked (Q10) |
| E | A full-length run (single + grid) | Results comparable to `main`'s recipe (behaviour parity, R5) |

## Appendix: hydra2 feature history

| # | Feature | In 08 |
|---|---|---|
| 00 | Hydra + typed schema + submitit Slurm launcher, `run.sh` | Hydra + schema kept; submitit dropped (Q1) |
| 01 | W&B integration (own the run, data-free config) | kept |
| 02 | Basic grid sweeper, `array_parallelism` cap | grid rules kept via `grid.py`; cap via `sbatch --array %N` |
| 03 | Output dir == W&B run name (`hfmt_runlabel` resolver) | replaced by `run_name` (Q2) |
| 04 | Atomistic components; experiments in `egs/`; sweep toggle via probe; full fine-tune | components kept; probe gone; fine-tuning modes ported as is (Q5) |
| 05 | W&B run name prefixed by experiment | gone (Q2) |
| 06 | `egs/<dataset>/<pair>/` hierarchy; `dataset.yaml` template filled in by code | hierarchy kept; `dataset.yaml` loaded by Hydra (Q3) |
| 07 | Two kinds of experiment (single-run vs `.sweep.yaml` with native `hydra.sweeper.params`) | kept |
