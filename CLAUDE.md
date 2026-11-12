# CLAUDE.md

Guidance for working in **HFMT** ("HuggingFace Machine Translation"), a small collection
of research scripts for machine translation and cross-lingual NLP. It is a thin, opinionated
layer over 🤗 Transformers, `trl`, `peft`, and `bitsandbytes`, targeting an academic
GPU-cluster workflow (SGE/`qsub`, SLURM/`sbatch`) with experiments organized as reproducible
recipes. Upstream: `kevinduh/hfmt`.

## Data confidentiality (must read)

**The dataset is proprietary. Dataset content MUST NOT be pushed to Weights & Biases.**
This applies to source/target sentences, prompts built from them, and model
hypotheses/translations — anything derived from the data.

- **W&B is the hard boundary.** It uploads to an external service, so only
  **aggregate, non-reversible metrics** (BLEU/CHRF/TER/ROUGE scores, loss, step counts)
  may be logged there. Never send sample text, tables, or artifacts containing examples
  to W&B.
- **Local files and logs are fine.** Writing example text to disk (prediction dumps like
  `*.pred`, `test<i>.pred.trg`) or to local `logging`/`print` output (e.g. "Decoded
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
│   ├── inf_translation.py   # decoder-only MT inference (trained run or downloaded checkpoint) — Hydra-driven
│   ├── merge_model.py       # merge a run's lora/qlora adapter into model_merged/ (launch.sh merge)
│   ├── mt_model.py          # model loading, prompting (CHAT_TEMPLATE_KWARGS), decoding, scoring shared by SFT + inference
│   ├── hydra_config.py      # typed Hydra config schema (ConfigStore): types only, no defaults
│   ├── grid.py              # login-node checks + job folders for egs/launch.sh (sft grid, inf, merge)
│   ├── decode_summarization.py  # zero/few-shot summarization
│   └── test_grid.py  test_mt_model.py  test_sft_translation.py   # standalone tests (see "Tools and commands")
├── conf/                    # SFT base config.yaml + groups train/ peft/ decode/; never experiments
├── analysis/run_sacrebleu.py    # standalone BLEU/CHRF/TER + vocab-overlap scorer
├── install/                 # conda bootstrap; pins an exact Transformers commit and hydra-core
├── egs/                     # recipes; launch.sh <sft|inf|merge> <yaml> launches the Hydra SFT workflow
│   ├── launch.sh            # launcher (sbatch array, LOCAL=1 or DRYRUN=1) for training, inference, merging
│   ├── job.sh               # one job of a launch: job.sh <sft|sft_resume|inf|merge> <folders...>
│   ├── data/                # example bitext, YAML/JSONL manifests
│   ├── mmtc/<src>-<trg>/    # SFT experiments, dataset.yaml, local_setting.sh (untracked);
│   │                        #   outputs/<run_name>[-<idx>]/ (gitignored)
│   ├── translation/  summarization/   # shell recipes for the two command-line entry points
├── tools/                   # gpu_free.sh, run_on_free_gpu.sh (pick the freest GPU type),
│                            #   inspect_model.py <checkpoint> (module names, LoRA candidates, memory estimate, prompt overhead),
│                            #   run_status.py <outputs dir> (training + inference states and scores, fix commands,
│                            #     failed checks: known-bad log warnings, model/ dev-score mismatch, deletable checkpoints,
│                            #     experiment yamls not launched yet; prints only, unless --delete-checkpoints) + its test
├── outputs/                 # direct (non-launch.sh) Hydra run outputs (gitignored)
├── transformers/            # git submodule (pinned HF Transformers)
└── wandb/                   # local W&B logs (gitignored)
```

## Key conventions

- **Configuration (Hydra)** — for the `sft_translation.py`/`inf_translation.py` workflow
  (features 08-11). **The schema (`hfmt/hydra_config.py`) holds types only, no default values;
  defaults live in yaml only:** `conf/config.yaml` (defaults list + project-wide `slurm`, `wandb`,
  `inf_name`), `conf/<group>/<option>.yaml` for the groups `train` (schedules; `shortrun` is the
  default), `peft` (`peft.mode` plus `peft.lora.{r,alpha,dropout,target_modules}`
  and `peft.quant`; `target_modules` is a list of exact module names, found with
  `tools/inspect_model.py <checkpoint>`) and `decode`, or the experiment folder. Don't enumerate
  the options in docs: each option file's header comment says what it is for (`ls conf/*/`), and
  README "Running SFT experiments" is the one place that lists them. A key no yaml sets fails
  before launch. Every schema field must be read by code (or, for the optional
  `data.data_root/src/trg`, by `dataset.yaml`).
  An experiment is a yaml in `egs/<dataset>/<src>-<trg>/` whose defaults start with `- config`
  and `- dataset` (the `dataset.yaml` next to it: `data_root`, `src`, `trg`, split file lists),
  then `override <group>: <option>` lines; it sets `run_name`, `checkpoint` and `instruction`.
  `data_root` comes from `$HFMT_DATA_ROOT`, set in the folder's untracked `local_setting.sh`
  (copy `local_setting.sh.example`); a dataset yaml may instead list absolute paths. Launch with
  `bash egs/launch.sh sft egs/mmtc/fr-en/sft_lora.yaml [overrides]` from a login node; e.g.
  `peft.lora.r=16`, `peft=lora_default`, `train=midrun`, `slurm.gres=gpu:l40s:1`. Avoid
  `+key`/`~key` overrides: they add or delete keys and so bypass the schema. See README "Running
  SFT experiments" (how to use it) and `decisions.md` (why it is this way).
- **Launch path (no submitit)** — `hfmt/grid.py` expands the grid, composes and checks every
  job's config, and saves it fully resolved as `<run folder>/run_config.yaml` (+ `git_rev`);
  `egs/launch.sh sft` then submits one Slurm array of `egs/job.sh sft` (each job a plain single
  Hydra run of its saved config) and returns. `LOCAL=1` runs the jobs locally, `DRYRUN=1` prints the
  sbatch command. Run folders: `<output_root>/<run_name>` (1 job) or `<run_name>-<idx>` (grid);
  `output_root` defaults to `<experiment folder>/outputs`. No timestamps: an existing folder
  only warns. W&B run name = run folder name, group = `run_name`. Tasks starting together can race
  for GPUs (exclusive-process mode), so `job.sh` waits for its GPU before starting;
  every sbatch has `--requeue --open-mode=append`, and a requeued `sft` task resumes from its last
  checkpoint (Slurm requeues only after a node failure or preemption, never after a failed exit).
- **Run folder housekeeping** — finished runs may be moved into `<outputs>/complete/`
  (`tools/run_status.py` skips that folder unless it is passed itself). `run_status.py` lists the runs whose
  `checkpoint-*` folders may go (test scores written, training job not in squeue; `model/` alone is saved
  before test decoding; `complete/` included) and `--delete-checkpoints` deletes them after asking, so
  unfinished runs stay resumable. `trainer_state.json` and `model/training_info.yaml` hold what the
  checkpoints knew.
- **Resume (feature 12)** — `bash egs/launch.sh sft <run folder>/run_config.yaml [slurm.* overrides]`
  continues a cancelled run from its last checkpoint, in the same folder with the same config; only
  `slurm.*` may change. `grid.py` `main_resume` refuses finished, checkpoint-less and cut-off runs and
  prints `KIND=sft_resume`; `egs/job.sh sft_resume` sets `HFMT_SFT_RESUME=1`. To change a run, launch
  its experiment yaml with a new `run_name`. What is and isn't restored: README "Resume".
- **Inference** — `bash egs/launch.sh inf <yaml> [overrides]`, one model per call. A trained run's
  `run_config.yaml` decodes its `model/` into `<run folder>/inf/<inf_name>/`; an experiment yaml decodes
  its `checkpoint` zero-shot into `<output_root>/<run_name>/base_ckpt_inf/<inf_name>/`. `grid.py --inf`
  saves `inf_config.yaml` there; `run_config.yaml` is never changed; no W&B. `mt_model.decode_test_sets`
  writes `test<i>.pred.trg` + `scores.json` for both training's end-of-run decoding and inference, and
  the two are identical (training unwraps Accelerate's autocast wrapper first). Outputs, overrides and
  score keys: README "Inference".
- **Merging / second SFT round** — `bash egs/launch.sh merge <run folder>/run_config.yaml` writes
  `<run folder>/model_merged/` for lora/qlora runs (qlora: onto the dequantized 4-bit base), usable as
  another experiment's `checkpoint`. A full fine-tune's `model/` is a checkpoint already. README
  "Merging and a second round of SFT".
- **Dev decoding and model saving** — the MT callback decodes dev + BLEU every
  `train.inference_callback_interval` evaluations (0 = never) and adds the scores to `log_history`;
  early stopping runs at every evaluation. At the end training saves `<run folder>/trainer_state.json`
  and `model/training_info.yaml` (the step `model/` holds, best metric and step, early stop, and
  `model_dev_check`: a fresh dev decode of the saved `model/` against the callback's BLEU at that
  step, within `DEV_SCORE_TOLERANCE`; a mismatch is logged, not fatal). For a full fine-tune the best
  checkpoint (the last one on resume) is loaded by `mt_model.load_checkpoint_weights`, because the
  Trainer's own reload loads nothing for Qwen3.5 and only warns (every full run before 2026-11-05 kept
  its last step; `decisions.md` 2026-11-05). `tools/run_status.py` reports mismatches and known-bad log
  warnings (`BAD_WARNINGS`). Details: README "Training".
- **Saved configs aren't backward compatible** — a run saved before a new required key was added
  fails `launch.sh inf`/`merge`'s checks (pass the key on the command line for `inf`). A saved config
  with a key the schema has since dropped (`grid.py` `RETIRED_KEYS`, e.g. `prompt_format`) is refused
  with the `sed` command that removes the line.
- **Parameter sweeps (two kinds of experiment)** — a single-run file has no grid (1 job); a
  `.sweep.yaml` carries its grid in `hydra.sweeper.params` and always runs all of it. Axes are
  leaf values or component files (`peft: lora_default,qlora_default`). Command-line comma
  lists add an axis on either kind; a single value pins one. `<run_name>.grid.txt` maps idx →
  values; change `run_name` when changing a grid's shape. `slurm.array_parallelism` caps
  concurrency.
- **Recipe-driven** (`egs/<task>/<lang-pair>/`) — the two command-line entry points
  (`train_seq2seq.py`, `decode_summarization.py`) are driven by plain shell scripts in
  `egs/translation/` and `egs/summarization/` that source `install/path.sh`, set shell vars, and call
  the script with arguments. They need `HFMT_ROOT` exported (`egs/launch.sh` derives it itself).
- **Pinned Transformers submodule** for reproducibility — the HF training APIs change often.
- **Data** = sentence-aligned parallel text (`.src`/`.trg`), listed per split (SFT: in the
  experiment's `dataset.yaml`; other recipes: a YAML manifest) and loaded via
  `datasets.load_dataset("text", ...)`. Summarization uses JSONL (`text`/`summary`).
- **Prompting**: seq2seq uses an instruction prefix. Causal-LM prompts are one user turn,
  `mt_model.format_input_prompt` (`instruction`, a newline, the source), rendered with the checkpoint's
  `apply_chat_template`. Every render (training via TRL's per-example `chat_template_kwargs` column, dev
  and test decoding, inference, `tools/inspect_model.py`) passes `mt_model.CHAT_TEMPLATE_KWARGS`
  (`enable_thinking=False`): Qwen3.5-4B's template thinks by default (0.8B/2B don't), and templates
  without the variable (Gemma) ignore it. The generation prompt must be a token prefix of prompt + reply:
  TRL's "Mismatch between tokenized prompt" warning in a training log means a checkpoint's template
  breaks that (`tools/run_status.py` flags it). The `SFTTrainer` gets `processing_class=tokenizer` so
  TRL doesn't load the VL processor Qwen3.5/Gemma 3 ship and switch to its multimodal path.
- **Tracking/metrics**: W&B (`WANDB_PROJECT="hfmt"`); `sacrebleu` (BLEU/CHRF/TER), ROUGE.

## Tools and commands

- Run scripts with `python`/`python3`; install deps via the conda bootstrap in `install/`
  (`install_hf_default.sh`; it pins Transformers and `hydra-core`).
- Tests are standalone scripts (pytest isn't in the hfmt env; they also run under pytest). After a
  change to `grid.py`, the schema, `conf/`, the experiment yamls or `run_status.py`, run the two that
  need no torch: `HFMT_ROOT=$(pwd) python hfmt/test_grid.py` and `python tools/test_run_status.py`.
  `python hfmt/test_mt_model.py` and `python hfmt/test_sft_translation.py` need torch (cluster env).
- Format with `black`, lint with `pylint`/`flake8`.
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

## Suspected bugs: quick tests before conclusions

A problem found by reading code, configs or model files is **suspected** until a run shows it. Before a
write-up, a plan change or a fix:

- **Say what the evidence shows and what it doesn't**, and which observation would confirm or refute it.
  Call it "suspected" or "untested" until then.
- **Write the smallest test that would show it** and give the developer the command to run it. This
  sandbox has no GPU or torch, so GPU tests run on the cluster.
  - A standalone script in the feature folder plus an sbatch wrapper that sources `install/path.sh` and
    waits for its GPU as `egs/job.sh` does.
    Call hfmt's own functions (`mt_model.load_base_model`, `inference_on_eval_data`) so the test runs the
    real code path, and include a control: a model or setting expected to behave.
  - Or a short throwaway training run: an existing experiment yaml with overrides (`checkpoint=`,
    `run_name=<topic>test_<model>`, `train.max_steps=50`, `train.eval_steps=25`, `wandb.enabled=false`).
    Check it composes (`DRYRUN=1`) and delete its run folder afterwards: it is not a result.
- **Inputs are made-up sentences, never dataset content**, and quick tests send nothing to W&B.
- **Read the output before concluding**: scores (TER far above 100 means run-on or chatty output),
  hypothesis lengths against the references, decode times, raw tokens.
- A cheap change that does nothing when the problem is absent may be proposed before a test, labelled
  as a safeguard, not a bug fix.

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
