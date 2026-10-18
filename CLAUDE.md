# CLAUDE.md

Guidance for working in **HFMT** ("HuggingFace Machine Translation"), a small collection
of research scripts for machine translation and cross-lingual NLP. It is a thin, opinionated
layer over 🤗 Transformers, `trl`, `peft`, and `bitsandbytes`, targeting an academic
GPU-cluster workflow (SGE/`qsub`, SLURM/`sbatch`) with experiments organized as reproducible
recipes. Upstream: `kevinduh/hfmt`. See `overview.md` for the full architecture writeup.

## Data confidentiality (must read)

**The dataset is proprietary. Dataset content MUST NOT be pushed to Weights & Biases.**
This applies to source/target sentences, prompts built from them, and model
hypotheses/translations — anything derived from the data.

- **W&B is the hard boundary.** It uploads to an external service, so only
  **aggregate, non-reversible metrics** (BLEU/CHRF/TER/ROUGE scores, loss, step counts)
  may be logged there. Never send sample text, tables, or artifacts containing examples
  to W&B.
- **Local files and logs are fine.** Writing example text to disk (prediction dumps like
  `*.pred`, `eval.pred.trg`) or to local `logging`/`print` output (e.g. "Decoded
  predictions…", batch-inspection dumps) is allowed — those stay on the machine. Keep
  them out of version control.
- **Config may be logged to W&B** because it holds only hyperparameters, data *paths*, and
  the instruction string — never data content. The Hydra SFT path logs the resolved config
  to `wandb.config`; keep it data-free and leave `WANDB_LOG_MODEL=false` (no model artifacts).
- **Keep W&B console capture off** (`WANDB_CONSOLE=off`, `wandb.Settings(console="off")`): by
  default W&B uploads the run's stdout/stderr as `output.log`, and our local logs contain
  example text. Any new W&B entry point must turn it off too.
- When adding or changing code, check every W&B call against this rule before running it,
  and make sure nothing data-bearing is routed to W&B (directly or via `report_to`).

## Project structure

```
hfmt/
├── hfmt/                    # Python entry-point scripts (not yet a package)
│   ├── train_seq2seq.py     # encoder–decoder MT (T5, Marian) via Seq2SeqTrainer
│   ├── sft_translation.py   # decoder-only MT SFT (qlora/lora/full) — Hydra-driven (@hydra.main)
│   ├── inf_translation.py   # decoder-only MT inference on a trained run folder — Hydra-driven
│   ├── mt_model.py          # model loading, prompting, decoding, scoring shared by SFT + inference
│   ├── hydra_config.py      # typed Hydra config schema (ConfigStore): types only, no defaults
│   ├── grid.py              # login-node grid expansion + per-job saved configs (egs/launch.sh)
│   └── decode_summarization.py  # zero/few-shot summarization
├── conf/                    # SFT base config.yaml + groups train/ peft/ decode/; never experiments
├── analysis/run_sacrebleu.py    # standalone BLEU/CHRF/TER + vocab-overlap scorer
├── install/                 # conda bootstrap; pins an exact Transformers commit
├── egs/                     # recipes; launch.sh <sft|inf> <yaml> launches the Hydra SFT workflow
│   ├── launch.sh            # launcher (sbatch array, LOCAL=1 or DRYRUN=1) for training and inference
│   ├── sft_job.sh  inf_job.sh   # one training job / one inference job
│   ├── data/                # example bitext, YAML/JSONL manifests
│   ├── mmtc/<src>-<trg>/    # SFT experiments, dataset.yaml, local_setting.sh (untracked);
│   │                        #   outputs/<run_name>[-<idx>]/ (gitignored)
│   ├── translation/  summarization/  synth_lrl/<lang>-eng/
├── tools/                   # gpu_free.sh, run_on_free_gpu.sh (pick the freest GPU type)
├── outputs/                 # direct (non-launch.sh) Hydra run outputs (gitignored)
├── transformers/            # git submodule (pinned HF Transformers)
└── wandb/                   # local W&B logs (gitignored)
```

## Key conventions

- **Configuration (Hydra)** — for the `sft_translation.py`/`inf_translation.py` workflow
  (features 08, 09). **The schema (`hfmt/hydra_config.py`) holds types only, no default values;
  every default lives in exactly one yaml:** `conf/config.yaml` (defaults list + project-wide
  `slurm`, `wandb`, `inf_name`), `conf/<group>/<option>.yaml` for the groups `train`, `peft`
  (`qlora_default` | `lora_default` | `full_finetune`: `peft.mode`, `peft.lora`, `peft.quant`) and
  `decode`, or the experiment folder. A key no yaml sets fails before launch. Every schema field
  must be read by code (or, for the optional `data.data_root/src/trg`, by `dataset.yaml`).
  An experiment is a yaml in `egs/<dataset>/<src>-<trg>/` whose defaults start with `- config`
  and `- dataset` (the `dataset.yaml` next to it: `data_root`, `src`, `trg`, split file lists),
  then `override <group>: <option>` lines; it sets `run_name`, `checkpoint` and `instruction`.
  `data_root` comes from `$HFMT_DATA_ROOT`, set in the folder's untracked `local_setting.sh`
  (copy `local_setting.sh.example`); a dataset yaml may instead list absolute paths. Launch with
  `bash egs/launch.sh sft egs/mmtc/fr-en/sft_lora.yaml [overrides]` from a login node; e.g.
  `peft.lora.lora_r=16`, `peft=lora_default`, `slurm.gres=gpu:l40s:1`. Avoid `+key`/`~key`
  overrides: they add or delete keys and so bypass the schema. See README "Running SFT
  experiments" and `decisions.md`.
- **Launch path (no submitit)** — `hfmt/grid.py` expands the grid, composes and checks every
  job's config, and saves it fully resolved as `<run folder>/run_config.yaml` (+ `git_rev`);
  `egs/launch.sh sft` then submits one Slurm array of `egs/sft_job.sh` (each job a plain single
  Hydra run of its saved config) and returns. `LOCAL=1` runs the jobs locally, `DRYRUN=1` prints the
  sbatch command. Run folders: `<output_root>/<run_name>` (1 job) or `<run_name>-<idx>` (grid);
  `output_root` defaults to `<experiment folder>/outputs`. No timestamps: an existing folder
  only warns. W&B run name = run folder name, group = `run_name`. Re-run a saved job with
  `bash egs/launch.sh sft <run folder>/run_config.yaml [overrides]` (same folder unless `run_name=`).
- **Inference** — `bash egs/launch.sh inf <run folder>/run_config.yaml [overrides]`, one trained
  run folder per call (loop in the shell for a sweep). `grid.py --inf` checks the saved config +
  overrides (e.g. `decode.num_beams=5`, `'data.test.src=[...]'`, `inf_name=beam5`) and saves
  `<run folder>/inf/<inf_name>/inf_config.yaml`; `egs/inf_job.sh` writes `pred.trg` and
  `scores.json` there. `run_config.yaml` is never changed; no W&B. Inference reproduces training's
  `eval.pred.trg` exactly: both use `mt_model.py`, and training unwraps Accelerate's autocast
  wrapper before its test decoding (and qlora bases get the same fp32 preparation).
- **Parameter sweeps (two kinds of experiment)** — a single-run file has no grid (1 job); a
  `.sweep.yaml` carries its grid in `hydra.sweeper.params` and always runs all of it. Axes are
  leaf values or component files (`peft: lora_default,qlora_default`). Command-line comma
  lists add an axis on either kind; a single value pins one. `<run_name>.grid.txt` maps idx →
  values; change `run_name` when changing a grid's shape. `slurm.array_parallelism` caps
  concurrency.
- **Recipe-driven** (`egs/<task>/<lang-pair>/`) — the *other* (unmigrated) entry points are
  still plain shell scripts that source `install/path.sh`, set shell vars, and call an
  `hfmt/*.py` script. `HFMT_ROOT` keeps paths portable. Those calling `sft_translation.py` or
  `inf_translation.py` (`egs/mmtc/zh-en/`, `egs/synth_lrl/*/inf*.sh`) use the old argparse CLIs
  and are stale.
- **Pinned Transformers submodule** for reproducibility — the HF training APIs change often.
- **Data** = sentence-aligned parallel text (`.src`/`.trg`), listed per split (SFT: in the
  experiment's `dataset.yaml`; other recipes: a YAML manifest) and loaded via
  `datasets.load_dataset("text", ...)`. Summarization uses JSONL (`text`/`summary`).
- **Prompting**: seq2seq uses an instruction prefix; causal LM uses `apply_chat_template`
  (with a plain-concat fallback).
- **Tracking/metrics**: W&B (`WANDB_PROJECT="hfmt"`); `sacrebleu` (BLEU/CHRF/TER), ROUGE.
- Known friction points (duplication, no shared package, module-level globals, magic numbers,
  no CI) are catalogued in `overview.md` §4–5.

## Tools and commands

- Run scripts with `python`/`python3`; install deps via the conda bootstrap in `install/`.
- Format with `black`, lint with `pylint`/`flake8`, test with `pytest` (where configured).
- Follow PEP 8, write docstrings, use type hints, keep functions focused.

## Experiment analysis

When asked to analyze an experiment, archive the analysis under `analysis/` in a directory
named for the current date (`analysis/<YYYY-MM-DD>/`; if one already exists for today, add a
short suffix so distinct analyses don't clobber each other, e.g. `analysis/2026-10-08_mmtc_fr-en_sft1/`).
Into that directory:

- **Copy** the `wandb_export.json` (or other source export) being analyzed, so the inputs are
  preserved alongside the findings.
- **Write** a markdown file (e.g. `summary.md`) summarizing the results: what the runs were, the
  sweep axes, a results table, and the key findings/recommendations.

Keep these the same data-confidentiality rules as everywhere else — summaries hold only
aggregate metrics and config (hyperparameters, data *paths*, instruction string), never dataset
content or model hypotheses.

## Development workflow

Feature work lives under a `features/` directory. Each feature gets its own subdirectory
named `IDX_FEATURENAME`, where `IDX` is a zero-padded ordinal that orders features as they
are developed and `FEATURENAME` is a short, high-level descriptor (e.g. `03_shared_data_lib`).

Each feature directory contains three files:

- **`request.md`** — written entirely by the developer. A few sentences describing the
  feature to build. Claude does not generate this.
- **`plan.md`** — developed collaboratively. The developer asks Claude to draft a plan; Claude
  proposes an implementation approach and **actively surfaces risks and open questions**, then
  the two narrow it down to an actionable plan. Risk identification is the priority at this stage.
- **`todo.md`** — generated entirely by Claude when asked, from the finalized `plan.md`.
  Granular, checkbox-style steps covering every aspect of the plan — think of it as a set of
  well-scoped GitLab issues. Items are **semantically grouped and ordered to capture
  dependencies** between them. Check items off as they are completed.

There is also a single high-level **`decisions.md`** (project root). Update it infrequently
and only to record important decisions and the context behind them — not routine progress.

## Commit messages

**Never run `git commit` yourself.** Claude must not execute `git commit` (or any
command that creates a commit) under any circumstances — not even when the developer
says "commit this" or "generate a commit". Instead, stage changes if helpful and
present the finished commit message for the developer to run themselves; the developer
always performs the actual commit. The same applies to `git push`.

Tie each commit to the feature and the `todo.md` phase it advances. Format the subject as
`<feature_dir>-<phase>: <summary>`, where `<feature_dir>` is the `features/` directory name
and `<phase>` is the todo group letter being worked on:

- **Subject line** — `<feature_dir>-<phase>: <clear, specific summary>`, e.g.
  `02_hydra_sweep-C: add committed sweep presets via +sweep= group`.
- **Body** — a very short, concise explanation of what else changed (one short paragraph or a
  couple of bullets). Keep it to the essentials; detail lives in `plan.md`/`todo.md`.

Example:

```
02_hydra_sweep-C: add committed sweep presets via +sweep= group

Add conf/sweep/ group with the first coarse mmtc fr-en preset (24-job grid).
Update the feature's todo.md/validation.md to mark group C done.
```

## Reminders

- Read relevant files before changing them; follow existing structure and patterns.
- Run tests after modifications; report failures honestly.
- Ask for clarification when feature requirements are unclear.
