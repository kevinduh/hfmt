# HFMT — Repository Overview

**HFMT** ("HuggingFace Machine Translation") is a small collection of research scripts
for **machine translation and cross-lingual NLP**, built as a thin, opinionated layer on
top of [🤗 Transformers](https://github.com/huggingface/transformers), `trl`, `peft`, and
`bitsandbytes`. It targets an academic GPU-cluster workflow (SGE/`qsub` and SLURM/`sbatch`)
where experiments are organized as reproducible "recipes."

The repository is upstream at `kevinduh/hfmt`.

---

## 1. What the project does

Four independent entry-point scripts cover the supported tasks:

| Script | Task | Core stack | Notes |
| --- | --- | --- | --- |
| `hfmt/train_seq2seq.py` | Encoder–decoder MT (T5, Marian/OPUS-MT) | `AutoModelForSeq2SeqLM`, `Seq2SeqTrainer` | Fine-tune a pretrained checkpoint *or* train from scratch from its config; BLEU/CHRF during eval; decodes an eval set at the end. |
| `hfmt/sft_translation.py` | Decoder-only MT via **QLoRA SFT** | `AutoModelForCausalLM`, `trl.SFTTrainer`, 4-bit `bitsandbytes`, `peft.LoraConfig` | Chat-template prompting, custom early-stopping callback that decodes the dev set and logs BLEU/CHRF/TER to W&B. |
| `hfmt/inf_translation.py` | Inference for decoder-only MT | `AutoModelForCausalLM`, optional `PeftModel` | Loads a base model (optionally with a LoRA adapter) and decodes an eval set. |
| `hfmt/decode_summarization.py` | Zero/few-shot summarization | `transformers.pipeline("text-generation")` | Prompt library (`sysprompts`), ROUGE scoring, `batch_size=1` and batched code paths. |

Supporting code:

- `analysis/run_sacrebleu.py` — standalone BLEU / CHRF / TER scorer plus a vocabulary-overlap
  report between hypothesis and reference (useful for diagnosing copy/undertranslation).
- `install/` — conda environment bootstrap that pins an exact Transformers commit.
- `egs/` — Kaldi-style example "recipes" (see below).

---

## 2. Architecture & key decisions

### 2.1 Recipe-driven design (`egs/`, "examples")
The project follows a **Kaldi/ESPnet-style recipe layout**. Each experiment lives in
`egs/<task>/<lang-pair>/` as a small shell script that:

1. `source ${HFMT_ROOT}/install/path.sh` to activate the conda env and load cluster modules,
2. sets the checkpoint, instruction prefix, data paths, and hyperparameters as shell vars,
3. calls the relevant `hfmt/*.py` script.

Directories seen: `egs/translation/` (T5, Marian), `egs/summarization/`, `egs/mmtc/`
(multilingual MT competition, zh-en with Qwen/Aya), `egs/synth_lrl/` (synthetic
low-resource languages: Ewe, Turkmen, Shan with tiny-Aya).

**Why it works:** every run is a self-documenting, version-controllable artifact — the
script *is* the experiment log. The `HFMT_ROOT` environment variable makes paths portable.

### 2.2 Pinned Transformers as a git submodule
`transformers/` is a git submodule (`.gitmodules`). `install/install_hf_default.sh` checks
out a **specific commit** (`HF_COMMIT`, currently v5.8.0) and installs it editable
(`pip install -e .`). `install/check_versions.py` records the exact tested versions of every
major dependency.

**Why:** the HF training APIs (`SFTConfig`, `Seq2SeqTrainingArguments`, chat templates)
change frequently and break silently; pinning a commit guarantees reproducibility and lets
users patch Transformers locally if needed.

### 2.3 Data as plain aligned text + YAML manifest
Training data is sentence-aligned parallel text (one sentence per line, `.src`/`.trg`).
A YAML manifest (`egs/data/example.template.yaml`) lists `train`/`dev` `src`/`trg` file
lists, which are loaded via `datasets.load_dataset("text", ...)`, renamed to `src`/`trg`,
and column-concatenated. Summarization instead uses JSONL with `text`/`summary` fields.

**Why:** matches how bitext is distributed in the MT community; no custom data format to learn.

### 2.4 Two prompting regimes
- Seq2Seq: a plain **instruction prefix** prepended to the source (`"translate German to English:"`).
- CausalLM: a **chat template** (`tokenizer.apply_chat_template`) with a `user`/`assistant`
  message structure; `inf_translation.py` falls back to a plain concatenated prompt when the
  tokenizer has no chat template.

### 2.5 Experiment tracking & metrics
- **Weights & Biases** (`WANDB_PROJECT="hfmt"`) is wired into all training runs; run names are
  derived from the output directory path.
- MT metrics use `sacrebleu` (BLEU with `flores200` tokenization, CHRF, TER); summarization uses ROUGE.
- QLoRA training uses a bespoke `EarlyStopping_MT_Callback` that actually **decodes** the dev
  set each eval (rather than relying on token-level loss) to get real translation metrics.

### 2.6 Cluster-first execution
Scripts carry `qsub`/`sbatch` invocation lines in comments. `install/path.sh` `module load`s
CUDA/cuDNN/NCCL. The design assumes a shared filesystem with absolute data paths under
`/exp/...`.

---

## 3. Repository layout

```
hfmt/
├── README.md                     # install + usage walkthrough
├── .gitmodules                   # pins huggingface/transformers submodule
├── hfmt/                         # the actual Python entry points
│   ├── train_seq2seq.py
│   ├── sft_translation.py
│   ├── inf_translation.py
│   └── decode_summarization.py
├── analysis/
│   └── run_sacrebleu.py          # standalone scoring + vocab overlap
├── install/
│   ├── install_hf_default.sh     # pins HF commit, calls custom installer
│   ├── install_hf_custom.sh      # creates conda env, installs deps
│   ├── path.sh                   # per-cluster env activation (edit for your site)
│   └── check_versions.py         # prints installed vs. tested versions
├── egs/                          # recipes (one shell script per experiment)
│   ├── data/                     # example bitext, YAML/JSONL manifests
│   ├── translation/              # T5, Marian
│   ├── summarization/
│   ├── mmtc/zh-en/               # Qwen / Aya QLoRA + inference
│   └── synth_lrl/{ewe,tuk,shn}-eng/
├── transformers/                 # git submodule (pinned HF Transformers)
└── wandb/                        # local W&B run logs (gitignored)
```

---

## 4. Observations that affect extensibility

These are the main friction points a future contributor will hit. None are bugs in the
"broken" sense (the code runs), but they cap how easily the repo grows.

1. **Heavy code duplication across the four scripts.** `format_input_prompt`,
   `inference_on_eval_data`, `preprocess_fn`, `get_data`, and the ~25-line argparse block are
   copy-pasted between `sft_translation.py` and `inf_translation.py` (and partly
   `train_seq2seq.py`). A change to prompting or decoding must be made in several places.

2. **No shared package.** `hfmt/` is a folder of scripts, not an importable package (no
   `__init__.py`, no `pyproject.toml`/`setup.py` for HFMT itself). Everything is invoked by
   file path via `HFMT_ROOT`, so logic can't be unit-tested or reused.

3. **Module-level globals** (`instruction_prefix`, `experiment_id`) are set inside `main()`
   with `global` and read by nested helpers. This couples functions to import-time state and
   blocks reuse.

4. **`inf_translation.py` carries unused training args** (`--max_steps`, `--learning_rate`,
   `--weight_decay`, …) copied from the trainer scripts — noise that suggests the arg parsing
   should be shared and task-scoped.

5. **Magic numbers scattered inline:** `max_length=128`, inference `batch_size=16`,
   `max_new_tokens=128`, `char_limit=10000`, LoRA `lora_dropout=0.1`. These are the knobs most
   likely to need tuning per language/model but aren't configurable.

6. **`qlora_target` mapping is brittle** (`sft_translation.py`): the `argparse` `choices` only
   allow `qv`/`all-linear`, yet the `if/elif` chain also handles `attention`/`mlp` (unreachable)
   and the error message claims the valid values are `'attention' or 'all'`. The comment
   `# TODO: fix, this is brittle` acknowledges this.

7. **Prompt templates are hardcoded** in `decode_summarization.py` (`sysprompts` dict) rather
   than loaded from files, so adding/versioning prompts means editing source.

8. **No automated tests or CI**, despite `CLAUDE.md` prescribing pytest and coverage. The only
   verification is a manual "smoke" run (`egs/models/t5small.smoke/`).

9. **Minor data-template bug:** in `egs/data/example.template.yaml` the `dev.trg` list points
   to `example.de-en.dev.de` (the German source) instead of `...dev.en`. Harmless for the
   throwaway example, but it will mislead anyone copying the template.

---

## 5. Recommended paths to greater extensibility

Ordered roughly by payoff-to-effort.

### 5.1 Extract a shared `hfmt` library
Turn `hfmt/` into a real package and pull the duplicated logic into modules, e.g.:

```
hfmt/
├── __init__.py
├── data.py          # get_data(), YAML manifest loading, preprocess_fn
├── prompting.py     # format_input_prompt(), chat-template vs. plain fallback
├── decoding.py      # inference_on_eval_data() (single source of truth)
├── metrics.py       # BLEU/CHRF/TER/ROUGE wrappers + compute_metrics
├── callbacks.py     # EarlyStopping_MT_Callback
└── cli/             # thin entry points: train_seq2seq, sft, infer, summarize
```

Add a `pyproject.toml` with `[project.scripts]` console entry points so recipes call
`hfmt-train-seq2seq ...` instead of `python $HFMT_ROOT/hfmt/....py`. This removes the
`HFMT_ROOT` path coupling and makes the code importable and testable.

### 5.2 Replace duplicated argparse with typed, shared config
Use a `@dataclass` group per concern (`DataArgs`, `ModelArgs`, `LoraArgs`, `DecodeArgs`) and
HF's `HfArgumentParser`, which also lets a run be driven by a single JSON/YAML config file.
Task-specific scripts then compose only the arg groups they actually use — fixing observation
#4 and centralizing the magic numbers of #5.

### 5.3 Introduce a small task/model registry
A dict mapping `task -> (model_loader, preprocess, decode, metric)` would let new tasks
(the README's "todo: cascading of models") or new model families be added by registering a
handler rather than writing a new top-level script. This is the natural home for the
seq2seq-vs-causal branching that is currently split across files.

### 5.4 Move prompts and hyperparameter presets into config files
Store `sysprompts` and the recurring hyperparameter blocks (the `cmdarg="..."` lines repeated
in every `egs/*.sh`) as YAML/JSON under `egs/config/`. Recipes become "pick a preset +
override," and prompts can be versioned and A/B-tested without code edits.

### 5.5 Decouple the cluster scheduler
The `qsub`/`sbatch`/`module load` specifics live in `path.sh` and script comments. A tiny
`egs/run.sh` wrapper (or a `--launcher {local,sge,slurm}` flag) that generates the submission
command would make recipes runnable on a laptop, a different cluster, or CI unchanged.

### 5.6 Add tests + CI
The `t5small.smoke` checkpoint shows a smoke test already exists informally. Formalize it:
a pytest that runs `train_seq2seq` for ~20 steps on the bundled `example.de-en` data and
asserts the run completes and emits metrics. Add unit tests for `data.py`/`prompting.py`
(pure functions once extracted). Wire a GitHub Actions job that runs `check_versions.py` and
the fast unit tests.

### 5.7 Fix the small correctness items
Correct `qlora_target` (observation #6) and the `example.template.yaml` dev/trg path
(observation #9) so new users aren't misled.

---

## 6. Quick start (from the README)

```bash
git clone https://github.com/kevinduh/hfmt.git
cd hfmt/
/bin/bash install/install_hf_default.sh      # creates conda env "hfmt", pins HF commit

export HFMT_ROOT=`pwd`
sed "s|__HFMT_ROOT__|$HFMT_ROOT|g" egs/data/example.template.yaml > egs/data/example.yaml

# submit a small T5 fine-tune (edit checkpoint/hyperparams in the script)
qsub -S /bin/bash -V -cwd -j y -q gpu.q@@v100 -l gpu=1,h_rt=24:00:00,num_proc=8,mem_free=25G \
     egs/translation/train_seq2seq_t5.sh
```

Score outputs afterward with:

```bash
python analysis/run_sacrebleu.py --ref <ref.trg> --hyp <eval.pred.trg> --do_ter
```
