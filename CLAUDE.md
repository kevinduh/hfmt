# CLAUDE.md

Guidance for working in **HFMT** ("HuggingFace Machine Translation"), a small collection
of research scripts for machine translation and cross-lingual NLP. It is a thin, opinionated
layer over 🤗 Transformers, `trl`, `peft`, and `bitsandbytes`, targeting an academic
GPU-cluster workflow (SGE/`qsub`, SLURM/`sbatch`) with experiments organized as reproducible
recipes. Upstream: `kevinduh/hfmt`. See `overview.md` for the full architecture writeup.

## Project structure

```
hfmt/
├── hfmt/                    # Python entry-point scripts (not yet a package)
│   ├── train_seq2seq.py     # encoder–decoder MT (T5, Marian) via Seq2SeqTrainer
│   ├── sft_translation.py   # decoder-only MT via QLoRA SFT (trl.SFTTrainer, 4-bit)
│   ├── inf_translation.py   # inference for decoder-only MT (optional PeftModel)
│   └── decode_summarization.py  # zero/few-shot summarization
├── analysis/run_sacrebleu.py    # standalone BLEU/CHRF/TER + vocab-overlap scorer
├── install/                 # conda bootstrap; pins an exact Transformers commit
├── egs/                     # Kaldi-style recipes, one shell script per experiment
│   ├── data/                # example bitext, YAML/JSONL manifests
│   ├── translation/  summarization/  mmtc/<lang-pair>/  synth_lrl/<lang>-eng/
├── transformers/            # git submodule (pinned HF Transformers)
└── wandb/                   # local W&B logs (gitignored)
```

## Key conventions

- **Recipe-driven** (`egs/<task>/<lang-pair>/`): each experiment is a shell script that
  sources `install/path.sh`, sets checkpoint/data/hyperparameters as shell vars, and calls
  an `hfmt/*.py` script. The script *is* the experiment log; `HFMT_ROOT` keeps paths portable.
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

## Reminders

- Read relevant files before changing them; follow existing structure and patterns.
- Run tests after modifications; report failures honestly.
- Ask for clarification when feature requirements are unclear.
