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
- When adding or changing code, check every W&B call against this rule before running it,
  and make sure nothing data-bearing is routed to W&B (directly or via `report_to`).

## Project structure

```
hfmt/
├── hfmt/                    # Python entry-point scripts (not yet a package)
│   ├── train_seq2seq.py     # encoder–decoder MT (T5, Marian) via Seq2SeqTrainer
│   ├── sft_translation.py   # decoder-only MT via QLoRA SFT — Hydra-driven (@hydra.main)
│   ├── hydra_config.py      # typed Hydra config schema (ConfigStore) for the SFT workflow
│   ├── inf_translation.py   # inference for decoder-only MT (optional PeftModel)
│   └── decode_summarization.py  # zero/few-shot summarization
├── conf/                    # Hydra config for the SFT workflow (model/data/train/decode/wandb/experiment/sweep/launcher)
├── analysis/run_sacrebleu.py    # standalone BLEU/CHRF/TER + vocab-overlap scorer
├── install/                 # conda bootstrap; pins an exact Transformers commit
├── egs/                     # recipes; run.sh <experiment> launches the Hydra SFT workflow
│   ├── data/                # example bitext, YAML/JSONL manifests
│   ├── translation/  summarization/  mmtc/<lang-pair>/  synth_lrl/<lang>-eng/
├── outputs/                 # Hydra run outputs (gitignored)
├── transformers/            # git submodule (pinned HF Transformers)
└── wandb/                   # local W&B logs (gitignored)
```

## Key conventions

- **Configuration (Hydra)** — for the `sft_translation.py` QLoRA SFT workflow: all tunable
  params live in `conf/` (typed schema in `hfmt/hydra_config.py`); an experiment is a preset
  `conf/experiment/<name>.yaml`; launch with `bash egs/run.sh <name>` (submits to Slurm via
  the submitit launcher; run from a login node, don't `sbatch` it). Override on the CLI, e.g.
  `bash egs/run.sh mmtc_fr-en_sft1 model.lora_r=16`. See README and `decisions.md`.
- **Parameter sweeps (Hydra basic grid)** — `egs/run.sh` runs in `--multirun`, so comma-list
  overrides fan out into one Slurm job per combination via submitit
  (`bash egs/run.sh mmtc_fr-en_sft1 model.lora_r=8,16,32`). Reusable grids are committed as
  `conf/sweep/<name>.yaml` and selected with `+sweep=<name>`. Sweep runs share a per-sweep W&B
  group and are named/tagged by their swept values (data paths/content never go in labels);
  `hydra.launcher.array_parallelism` caps concurrency. This replaces the old W&B-sweep tooling
  (`wandb sweep`/`wandb agent`). See README "Sweeping parameters".
- **Recipe-driven** (`egs/<task>/<lang-pair>/`) — the *other* (unmigrated) entry points are
  still plain shell scripts that source `install/path.sh`, set shell vars, and call an
  `hfmt/*.py` script. `HFMT_ROOT` keeps paths portable.
- **Pinned Transformers submodule** for reproducibility — the HF training APIs change often.
- **Data** = sentence-aligned parallel text (`.src`/`.trg`), listed in a YAML manifest and
  loaded via `datasets.load_dataset("text", ...)`. Summarization uses JSONL (`text`/`summary`).
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
