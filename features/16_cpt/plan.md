# 16_cpt — plan (v10, final)

Status: **v10, final** (2026-11-18). Every question is decided; Q14 and Q15 take their recommendations.
Next: `todo.md`.

Developer decisions so far (2026-11-18):
- **CPT and SFT are separate experiment yamls** (option B). A CPT yaml has a `- cpt: <option>` line; an SFT
  yaml is the same as today. This replaces the shared-yaml design of v2–v4, so `cpt.train`, nested
  `<run_name>/cpt` folders, stripping `cpt` from SFT configs and the cross-override guards are all gone.
- Class name `CPTConfig`.
- Answers given under the shared-yaml design, rechecked under B in "Questions": Q1 shared code, Q4 space
  separator, Q5 sampling ratio, Q6 CPT never decodes (efficiency matters), Q7 tokenizer `eos_token`, Q8
  `HFMT_SFT_RESUME`, Q9 `cpt.packing_strategy`.
- **`conf/train/` stays SFT-only** (Q5/Q6/Q12): a train yaml needs no knowledge of CPT. Everything CPT,
  including the mixing ratio, is under `cpt:`. When `cpt` is set, CPT forces its own values for the
  train keys that only make sense for SFT and ignores what the train option says, e.g. decoding in
  `train/longrun.yaml`. There is no `conf/train/cpt.yaml`.
- Q3: one `dataset.yaml` serves both SFT and CPT experiments; each script reads the `data` lists it
  needs and ignores the rest.
- Q13: an explicit `launch.sh cpt <cpt yaml>`; a mismatch is refused both ways.
- §5: the examples go in a **new, committed `egs/synlrl/`** with `tuk-eng`, `shn-eng` and `ewe-eng`. Each
  pair has an `sft_lora.yaml` (the example SFT run, as in `egs/synth_lrl`) and a `cpt_lora.yaml` like it.
  `egs/synlrl/dataset.yaml` adds `monotext: [${oc.env:HFMT_DATA_ROOT}/synth1.${data.src}]`. The original
  `egs/synth_lrl/` stays untracked and untouched, and will be deleted by hand later. D5: fix
  `run_status.py` for `- ../dataset` yamls.
- E1: the `egs/synlrl` yamls use the tracked `conf/peft/lora_tiny_aya_all_linear.yaml` as is. E2: the
  proposed `conf/cpt/default.yaml`. E3: include `sft_lora_after_cpt.yaml` per pair.

## Request (summary)

- New script `hfmt/cpt_monotext_bitext.py`, built like `hfmt/sft_translation.py`, that does continued
  pre-training (plain next-token loss, no chat template) with `SFTTrainer`, `packing=True` and
  `metric_for_best_model=eval_loss`.
- Data: the same `dataset.yaml` as SFT, plus an optional `data.monotext` list of text files. There are three
  options: monotext only, bitext only, or both. For bitext, src and trg are either **separate** examples or **concatenated**
  line by line.
- All new options are optional, so existing SFT runs and saved `run_config.yaml`s are untouched.
- `egs/launch.sh` and `egs/job.sh` launch it.
- Intended use: CPT, save the model, then SFT starting from it (`checkpoint:` = the CPT model).

## Starting point (verified by reading the code; branch `synlrl`, Transformers 5.8.0, TRL 1.3.0)

**Launch path assumes every training run is SFT:**
- `egs/job.sh` hardcodes `python -m hfmt.sft_translation` for `sft`/`sft_resume`. Its requeue check
  (`kind = sft` + `SLURM_RESTART_COUNT > 0` → `sft_resume`) also names `sft`.
- `hfmt/grid.py`: `main_sft` handles experiment yamls and, for a `run_config.yaml`, `main_resume`. That
  function prints `KIND=sft_resume`. `main_inf` and `main_merge` don't care which script trained the run.
- `tools/run_status.py` `fix_command` prints `egs/job.sh sft <run>` (failed/not started) and
  `launch.sh sft <run_config>` (interrupted). Its "finished" test is `model/training_info.yaml` or
  `scores.json`.

**Saved configs aren't backward compatible** (CLAUDE.md): a new *required* key breaks resume/inf/merge
of every existing run. So the new keys must be `Optional[...] = None`.

**Schema:** `data.train` and `data.dev` are required `Bitext`s, with src and trg required. The check is
`grid.py:_compose_checked` → `OmegaConf.missing_keys`. Only `data.test.trg` is optional.

**TRL 1.3.0, plain `text` datasets** (`trl/trainer/sft_trainer.py`):
- **EOS:** TRL appends `tokenizer.eos_token` to every `text` example unless it already ends with it
  (`add_eos`, ~l.1111). The token is a chat-turn token on some of our models:
  Tiny Aya `<|END_OF_TURN_TOKEN|>`, Gemma 3 `<eos>`. Qwen3.5 is not in the local HF cache; Qwen instruct
  tokenizers usually use `<|im_end|>` (to check). `SFTConfig.eos_token` overrides it.
- **BOS:** `_tokenize` calls the tokenizer with its own special-token settings. Gemma and Aya
  (`add_bos_token: true`) put BOS on every example; Qwen has no BOS.
- **Packing:** `packing=True` defaults to `packing_strategy="bfd"` (best-fit decreasing; an example
  longer than `max_length` is *truncated*). `bfd` forces **padding-free** mode (l.837): each batch is
  flattened into **one row** with `position_ids` restarting at 0 for every packed example, and no
  `attention_mask`. The sequence length is `SFTConfig.max_length`, default 1024. Our schema doesn't set it,
  so SFT runs use the default as well.
- With a non-flash attention implementation, TRL warns that padding-free/packing "may lead to
  cross-contamination between samples". `load_base_model` sets no `attn_implementation` (so sdpa), and
  neither `flash_attn` nor `fla` nor `causal_conv1d` is installed in the `hfmt` env.
- `eval_packing` defaults to `packing`, so dev loss is computed packed too.
- `completion_only_loss` has no effect on a `text` dataset, because there is no `completion_mask`.

**Transformers 5.8.0, packed batches without flash-attention:**
- `masking_utils._preprocess_mask_arguments` detects packed sequences from `position_ids` when
  `attention_mask is None` (`find_packed_sequence_indices`). It then builds a block-diagonal causal mask, so
  **regular attention layers (sdpa/eager) keep packed examples apart.**
- **Qwen3.5 is hybrid:** 3 of every 4 layers are Gated DeltaNet linear attention (`layer_types`,
  `full_attention_interval=4`). `Qwen3_5GatedDeltaNet.forward` passes no sequence boundaries:
  `causal_conv1d_fn(..., seq_idx=None)` (or plain `conv1d`), and `chunk_gated_delta_rule` gets no
  `cu_seqlens`. **Suspected (not yet run):** under padding-free packing, the linear-attention layers carry
  state from one packed example into the next across the whole flattened batch. The full-attention layers
  don't. Gemma 3 and Tiny Aya are pure attention and should be fine. See R1.

## Design

### 1. Config (schema + `conf/`)

All new fields are optional, so SFT configs and every saved `run_config.yaml` compose unchanged.

```python
@dataclass
class DataConfig:
    ...
    monotext: Optional[List[str]] = None       # CPT: monolingual text files, one example per non-empty line

@dataclass
class CPTConfig:
    """Continued pre-training data and packing (hfmt/cpt_monotext_bitext.py). Set only in CPT
    experiments; its presence makes a run a CPT run."""
    use_monotext: bool = MISSING                # train on data.monotext
    use_bitext: bool = MISSING                  # train on data.train (src and trg)
    bitext_format: str = MISSING                # separate | concat
    monotext_ratio: float = MISSING             # with both sources: fraction of training examples drawn from monotext
    max_length: int = MISSING                   # packed sequence length (SFTConfig.max_length)
    packing_strategy: str = MISSING             # bfd | wrapped (SFTConfig.packing_strategy; R1)

@dataclass
class Config:
    ...
    cpt: Optional[CPTConfig] = None
```

- **`cpt` marks a CPT run**, in the composed config and in the saved `run_config.yaml` alike. Only CPT
  experiments add it, so no stripping and no new required key are needed, and old saved configs are
  unaffected.
- **`conf/cpt/` group** (not in `conf/config.yaml`'s defaults): the CPT data/packing options. A CPT
  experiment adds it with a plain `- cpt: <option>` line. One option to start, `conf/cpt/default.yaml`
  (bitext only, separate, `monotext_ratio: 0.5`, `max_length: 1024`, `packing_strategy: bfd`); more as
  needed. Header comments as usual.
- **The CPT schedule is any `conf/train/` option**, read for its generic keys (steps, learning rate,
  scheduler, batch size, eval/save intervals, early stopping, seed). A CPT yaml picks one with
  `override train: <option>` like an SFT yaml, and sets a smaller `train.train_batch_size` itself (R3).
- **Forced train values for CPT** (Q6, Q12). Five `TrainConfig` keys only make sense for SFT. When `cpt`
  is set, they are replaced, whatever the train option says:

  | key | forced value | why |
  |---|---|---|
  | `packing` | `true` | CPT packs (request) |
  | `completion_only_loss` | `false` | no prompt/completion in a `text` dataset |
  | `inference_callback_interval` | `0` | CPT never decodes |
  | `metric_for_best_model` | `eval_loss` | the only metric without decoding |
  | `greater_is_better` | `false` | goes with `eval_loss` |

  - One function in `hfmt/hydra_config.py` (no torch), `apply_cpt_overrides(cfg) -> changed`, holds this
    table. There is no extra on/off option under `cpt`: CPT never decodes (Q6), so a switch with one
    allowed value would add nothing.
  - **`grid.py` applies it before saving** the run config of a CPT launch, and prints each changed key
    (`grid: cpt: train.inference_callback_interval 1 -> 0 (CPT never decodes)`). The saved
    `run_config.yaml` then says what actually ran, and `run_status.py`, which reads
    `train.metric_for_best_model`, sees `eval_loss`.
  - **The CPT script applies it again at start-up** (it does nothing on a saved config) and logs the
    same lines, so a direct `python -m hfmt.cpt_monotext_bitext` run behaves the same.
  - The alternative, forcing only inside the training code, would leave a saved config that says
    `inference_callback_interval: 1`, `metric_for_best_model: eval_sp_bleu` for a run that did neither.
  - A command-line override of a forced key (e.g. `train.inference_callback_interval=1` on a CPT launch)
    is refused rather than silently replaced: it can only be a mistake.
- **`check_config` additions** (it sees the whole config, so no kind argument is needed):
  - with `cpt` set:
    - `bitext_format ∈ {separate, concat}`, `packing_strategy ∈ {bfd, wrapped}`;
    - at least one of `use_monotext`/`use_bitext`;
    - `use_monotext` needs a non-empty `data.monotext`;
    - `0 < monotext_ratio < 1` when both sources are on (ignored otherwise).

    The forced train keys aren't checked; they are set (above).
  - `data.monotext` without `cpt` is allowed (Q3: a shared `dataset.yaml` lists it); SFT ignores it and
    logs one line saying so.
- `instruction` stays required by the schema (Q15).
- **One `dataset.yaml` for SFT and CPT** (Q3). Each script reads the `data` lists it needs and ignores
  the rest:

  | `data` key | SFT | CPT |
  |---|---|---|
  | `train` (bitext) | training data | training data when `cpt.use_bitext` |
  | `monotext` | ignored (logged) | training data when `cpt.use_monotext` |
  | `dev` | dev loss + dev BLEU | dev loss |
  | `test` | test decoding | ignored (CPT never decodes) |

  `monotext` can also be set or overridden in an experiment yaml, like any `data` key. CPT and SFT read the
  same `train` bitext.

### 2. Data → training examples

One `text` column, built from the `cpt` options:

| source | `separate` | `concat` |
|---|---|---|
| bitext train (`use_bitext`) | each src line and each trg line is its own example | `src + " " + trg` per line pair (Q4) |
| monotext (`use_monotext`) | each non-empty line is an example | same |

- **Train with both sources: sampled by `cpt.monotext_ratio`** (Q5). The ratio sets the *mix*, not how
  much data is used: no source is cut to fit it, and how much is trained is set by `train.max_steps`. The two example sets (monotext
  lines; bitext examples after formatting) are mixed with `datasets.interleave_datasets([mono, bitext],
  probabilities=[r, 1-r], seed=train.seed, stopping_strategy="all_exhausted")`:
  - Each example is drawn from monotext with probability `r`, until both sets have been seen once. The
    smaller set is oversampled (repeated) to keep the ratio. The result is one finite dataset that the
    Trainer shuffles and loops over until `max_steps`.
  - The alternative, `first_exhausted`, stops when the smaller set runs out and so drops part of the
    larger one. That's not wanted when data is plentiful and the run is defined by `max_steps`.
  - The ratio counts **examples (lines), not tokens**. Monotext lines may be much longer than sentences, so
    the token share differs. The script logs both the example counts and the token share per source
    (from the tokenized lengths), so the actual mix is visible.
  - Size ≈ max(n_mono / r, n_bitext / (1-r)). Interleaving is an index mapping and cheap in Arrow;
    tokenizing the result is the cost (R8).
  - With one source only, `monotext_ratio` is ignored, as there is nothing to mix.
- **Dev (eval loss)** = `data.dev` bitext formatted with the same `bitext_format` (Q3), also when
  `use_bitext` is false. So every CPT run has a dev loss. No monotext dev set.
- Empty lines are dropped. Otherwise they become EOS-only examples.
- Bitext is read as in SFT (src and trg files joined line by line), so the files must be line-aligned in
  both formats.
- Formatting functions are module-level and get their options through `fn_kwargs`. This keeps the
  `datasets` cache fingerprints stable (R8) and makes them testable without torch.

### 3. `hfmt/cpt_monotext_bitext.py` and code sharing (Q1)

`sft_translation.main()` is ~280 lines, and most of it would be identical for CPT: W&B init with the
privacy settings, LoRA/QLoRA/full setup, the Trainer args, batch inspection, resume, the full-FT reload
through `load_checkpoint_weights`, saving `model/`, `trainer_state.json` and `training_info.yaml`. A copy
would duplicate code that has been debugged carefully (decisions.md 2026-11-05, 12_resume_sft).

- Move the shared blocks of `sft_translation.main()` into functions in a new `hfmt/train_common.py`
  (names indicative):
  - `init_wandb(cfg, outdir, resume)`
  - `build_model(cfg, device) -> (model, use_bf16)` (load + PEFT + parameter logging)
  - `trainer_args(cfg, outdir, use_bf16, **extra) -> SFTConfig` (`extra`: the per-script settings)
  - `inspect_first_batch(trainer, tokenizer)`
  - `train(trainer, model, cfg, outdir, resume)` (resume + full-FT reload)
  - `save_model(trainer, model, tokenizer, outdir) -> info` (unwrap, `model/`, tokenizer,
    `trainer_state.json`, `training_info`)
- The module-level helpers (`config_token_ids`, `wandb_privacy_settings`, `training_info`) move there too.
  `sft_translation.py` keeps what only SFT uses: the MT callback, `add_dev_scores_to_log_history`,
  `check_model_dev_scores` and test decoding. It still exports the three names that
  `test_sft_translation.py` imports.
- `sft_translation.main()` becomes data prep + these calls + its dev check and test decoding. Its
  behaviour must not change: the same SFTConfig values and the same log lines (R5).
- `cpt_monotext_bitext.main()` differs in:
  - data prep (§2);
  - `trainer_args(..., dataset_text_field="text", max_length=cfg.cpt.max_length,
    packing_strategy=cfg.cpt.packing_strategy, dataset_num_proc=cfg.slurm.cpus)`, after
    `apply_cpt_overrides` (so `packing`, `completion_only_loss` and the metric come out forced), and with
    the tokenizer's own `eos_token` (Q7);
  - the callback: transformers' plain `EarlyStoppingCallback` on `eval_loss`, with no decoding (Q6);
  - the model/ check (below);
  - no test decoding (Q6).

  Batch inspection shows the packed row and the EOS/BOS between examples.
- Resume: the same `HFMT_SFT_RESUME=1` variable (Q8).

**model/ check for CPT.** SFT's `check_model_dev_scores` compares a fresh dev decode with the callback's
BLEU at the step `model/` holds. CPT has no BLEU. For CPT:
- `trainer.evaluate()` runs on the final model;
- its `eval_loss` is compared with `log_history`'s `eval_loss` at `model_global_step` (small tolerance,
  e.g. 1e-3 relative; to be measured);
- it catches the same failure (the reload loaded nothing) at the cost of one dev evaluation;
- `training_info.yaml` gets the same `model_dev_check` field ("ok" / "unchecked: …" / "MISMATCH: …"), so
  `run_status.py` reads it unchanged.

**No test decoding** (Q6). A CPT run ends after saving `model/` and `training_info.yaml`. A zero-shot
BLEU of a CPT model is still available by hand with `launch.sh inf <cpt run>/run_config.yaml`, which is
unchanged.

### 4. Launch: `launch.sh`, `grid.py`, `job.sh`

- `bash egs/launch.sh cpt <experiment.yaml> [overrides]` (an explicit kind, Q13):
  - `grid.py --cpt` → `main_sft(..., kind="cpt")`: the same grid expansion, folders
    (`<output_root>/<run_name>[-<idx>]`), `.grid.txt`, saved `run_config.yaml` and job names as SFT;
  - it refuses a config without `cpt` ("not a CPT experiment; use launch.sh sft");
  - it prints `KIND=cpt`.
- `bash egs/launch.sh sft <experiment.yaml>` refuses a config with `cpt` ("a CPT experiment; use launch.sh
  cpt").
- Resume, `bash egs/launch.sh cpt <run folder>/run_config.yaml [slurm.*]`: through `main_resume`, which
  prints `KIND=cpt_resume` for a saved config with `cpt`. The launch kind must match the saved config
  (`launch.sh sft <cpt run_config>` is refused, and vice versa).
- `job.sh`: new kinds `cpt` and `cpt_resume` → `python -m hfmt.cpt_monotext_bitext`; the requeue rule
  becomes `sft|cpt` → `<kind>_resume`.
- `inf` and `merge` need no change: `model/` has the same format. `merge` turns a LoRA/QLoRA CPT run into
  `model_merged/`; a full CPT run's `model/` is a checkpoint directly.
- **Safeguard (Q14):** `grid.py` checks that a `checkpoint` that is an absolute path exists. An SFT yaml
  pointing at a CPT run not merged or finished yet would otherwise fail only on the GPU (HF reads it as a
  hub id).
- `tools/run_status.py`, for a run whose saved config has `cpt`:
  - `fix_command` prints `cpt` instead of `sft` (`launch.sh cpt …`, `sbatch … egs/job.sh cpt …`);
  - `has_test` is False (CPT never decodes), so the run is "finished" and its checkpoints deletable once
    `model/training_info.yaml` exists. These rules already exist for SFT runs without test sets.

  Discovery, queue matching, `max_steps` and launch counting are unchanged.

### 5. Experiments: new `egs/synlrl/` (tuk-eng, shn-eng, ewe-eng)

A new folder, committed in full; nothing in it is copied from runs. The untracked `egs/synth_lrl/` is not
touched: its yamls, run folders and `local_setting.sh` files stay as they are, and it will be deleted by
hand later. Runs launched from `egs/synlrl/` start with fresh `outputs/` folders.

```
egs/synlrl/
├── README.md                   # data files each pair needs (names only), the CPT -> merge -> SFT workflow
├── dataset.yaml                # shared by the pair folders (- ../dataset)
├── tuk-eng/
│   ├── local_setting.sh.example
│   ├── sft_lora.yaml           # example SFT run
│   ├── cpt_lora.yaml           # example CPT run
│   └── sft_lora_after_cpt.yaml # SFT from the CPT run's model (E3)
├── shn-eng/  (same files)
└── ewe-eng/  (same files)
```

**`dataset.yaml`:** the same lists as `egs/synth_lrl/dataset.yaml`, plus `monotext` (Q3). CPT and SFT both
use it:
```yaml
# @package data
# synlrl <src>-eng data: seed bitext, synthetic monotext, flores dev/test; <src>/<trg> set by each experiment.
train:
  src: [${oc.env:HFMT_DATA_ROOT}/seed1.${data.src}-${data.trg}.${data.src}]
  trg: [${oc.env:HFMT_DATA_ROOT}/seed1.${data.src}-${data.trg}.${data.trg}]
monotext:
  - ${oc.env:HFMT_DATA_ROOT}/synth1.${data.src}
dev:
  src: [${oc.env:HFMT_DATA_ROOT}/flores200.smalldev-256.${data.src}]
  trg: [${oc.env:HFMT_DATA_ROOT}/flores200.smalldev-256.${data.trg}]
test:
  src: [${oc.env:HFMT_DATA_ROOT}/flores200.dev.${data.src}]
  trg: [${oc.env:HFMT_DATA_ROOT}/flores200.dev.${data.trg}]
```
- SFT ignores `monotext`; CPT ignores `test`.
- `synth1.{tuk,shn,ewe}` aren't in `$HFMT_DATA_ROOT` yet (checked 2026-11-18). Until they are, a CPT launch
  composes but the job fails when it loads the file.

**`local_setting.sh.example`** (per pair, like mmtc's): `export HFMT_DATA_ROOT=/path/to/...` with a
one-line comment. Each user copies it to the gitignored `local_setting.sh`.

**`sft_lora.yaml`** (per pair): the same settings as `egs/synth_lrl/<pair>/sft_lora.yaml`, i.e. Tiny Aya
global, `override peft: lora_tiny_aya_all_linear` (E1), `decode.batch_size: 64`, tags `[synlrl, <pair>]`, with a one-line header.

**`cpt_lora.yaml`** (per pair), "similar to `sft_lora.yaml`":
```yaml
# This experiment is for CPT of Tiny Aya global on synlrl tuk-eng monotext and bitext.
defaults:
  - config
  - ../dataset
  - cpt: default
  - override peft: lora_tiny_aya_all_linear
  - _self_

run_name: cpt_lora
checkpoint: CohereLabs/tiny-aya-global
instruction: "Translate Turkmen to English"    # required by the schema; CPT doesn't read it (Q15)

data:
  src: tuk
  trg: eng

train:
  train_batch_size: 2              # packed rows of cpt.max_length tokens (R3); value from the 50-step test
  gradient_accumulation_steps: 8

wandb:
  tags: [synlrl, tuk-eng, cpt]
```
`conf/cpt/default.yaml` (E2) supplies the `cpt:` values; the experiment overrides them as needed.

**`sft_lora_after_cpt.yaml`** (per pair, E3): `sft_lora.yaml` with `run_name: sft_lora_after_cpt` and
`checkpoint: ${oc.env:HFMT_ROOT}/egs/synlrl/<pair>/outputs/cpt_lora/model_merged`.

**Workflow** (also in the README):
```
bash egs/launch.sh cpt   egs/synlrl/tuk-eng/cpt_lora.yaml                      # -> outputs/cpt_lora/
bash egs/launch.sh merge egs/synlrl/tuk-eng/outputs/cpt_lora/run_config.yaml   # lora -> model_merged/
bash egs/launch.sh sft   egs/synlrl/tuk-eng/sft_lora_after_cpt.yaml            # -> outputs/sft_lora_after_cpt/
bash egs/launch.sh sft   egs/synlrl/tuk-eng/sft_lora.yaml                      # baseline: SFT without CPT
```

Tiny Aya is pure attention, so R1 (the Qwen3.5 packing leak) doesn't apply. R2 does: its `eos_token` is
`<|END_OF_TURN_TOKEN|>`.

**`tools/run_status.py` fix (D5).** Today `find_experiments` only finds experiments:
- in folders that contain a `dataset.yaml`, and
- whose defaults start with exactly `config`, `dataset`.

The `egs/synlrl` pair folders have no `dataset.yaml` of their own, and their yamls say `- ../dataset`.
So none of them would appear in the "experiment yamls not launched yet" list (their run folders are still
reported). Change:
- **An experiment** is a yaml whose defaults start with `config` followed by a dataset yaml: a plain string
  whose last path component starts with `dataset` (`dataset`, `../dataset`, `dataset_small100.yaml`). It
  must also set `run_name`, as now.
- **An experiment folder** is any folder (outputs/ and run folders skipped, as now) holding such a yaml,
  not only folders with a `dataset.yaml`. The folder holding a root `outputs/` counts as before.
- `tools/test_run_status.py`: a pair folder with `- ../dataset` yamls and no `dataset.yaml` of its own is
  found; a yaml whose second default isn't a dataset still isn't.
- CLAUDE.md's experiment convention reads "defaults start with `- config` and a dataset yaml (`- dataset`,
  `- ../dataset`, …)", and the structure tree lists `egs/synlrl/`.

## Risks

- **R1 — Qwen3.5 packing leaks across examples (suspected, from code reading).** With `bfd`
  (padding-free), full-attention layers are isolated per example but the linear-attention layers are not.
  At inference those layers start each sentence from zero state, so training sees a different context than
  decoding. Options (Q9):
  - (a) `bfd` anyway: effect probably small, but untested;
  - (b) `packing_strategy="wrapped"`: classic concat-and-chunk, where every layer sees across examples,
    consistently, as in standard pretraining, with EOS as the separator; examples are split at chunk edges;
  - (c) install `flash-attn`/`fla`: that alone doesn't help, because transformers 5.8.0 still passes no
    `cu_seqlens`/`seq_idx` to the DeltaNet layers;
  - (d) `packing=False`: correct but wasteful; the request asks for packing.

  **Quick test to settle it** (CLAUDE.md "suspected bugs"):
  - script `features/16_cpt/test_packing_leak.py` + sbatch wrapper;
  - made-up sentences;
  - compute per-example loss on the same examples (1) one at a time and (2) as a packed padding-free batch
    built by TRL's own collator (`DataCollatorForLanguageModeling(padding_free=True)`);
  - for Qwen3.5-0.8B or 2B (hybrid) and Gemma 3 1B / Tiny Aya (control, expected equal);
  - a per-example loss difference beyond bf16 noise on Qwen3.5 only confirms R1.
- **R2 — EOS is a chat-turn token.** Raw-text CPT appends `<|END_OF_TURN_TOKEN|>` (Aya) / likely
  `<|im_end|>` (Qwen) after every sentence. The model learns "a sentence is followed by end-of-turn", which is
  close to what it sees in SFT replies, so this is probably harmless or even helpful. The alternative is the
  base `<|endoftext|>`-style token. Decided for v1 (Q7): keep the tokenizer's token and look at the first
  batch-inspection log.
- **R3 — Batch size means something else.** With packing, `per_device_train_batch_size` counts packed
  rows of `cpt.max_length` tokens. 16 × 1024 = 16k tokens per step, against SFT's 16 short pairs (a few
  hundred tokens). Padding-free flattens the batch into one 16k-token row. With sdpa and an explicit
  block mask (no flash kernel), memory for full-attention layers grows ~quadratically in that row length,
  and OOMs are likely at SFT batch sizes. A CPT experiment yaml should start small (e.g.
  `train.train_batch_size: 2`, `cpt.max_length: 1024`, raise `train.gradient_accumulation_steps`). Batch
  size isn't forced: it depends on model and GPU. Measure with a
  50-step run. `tools/inspect_model.py`'s memory estimate doesn't model packing.
- **R4 — `bfd` truncates long lines.** An example longer than `cpt.max_length` loses its tail.
  Sentences won't hit this; monotext with long paragraph lines could. Log how many examples exceed it
  (cheap: tokenized lengths).
- **R5 — Refactoring SFT.** Moving code out of `sft_translation.py` risks changing SFT behaviour (resume,
  full-FT reload, the model/ check). Mitigation: a pure move (no logic edits in the same step), the existing
  tests, a `DRYRUN=1` + `LOCAL=1` short SFT run compared with a pre-refactor run (same losses at a fixed
  seed), and a resume of a cancelled short run.
- **R6 — CPT on instruct models without a template.** It may weaken chat behaviour. The second SFT round
  is meant to restore it. CPT itself doesn't measure it (Q6); `launch.sh inf` on the CPT run does, on
  demand. This is a research risk, not a code risk.
- **R8 — Preprocessing time and disk on large data.** Everything before step 1 runs in the main process,
  single-threaded by default:
  - `load_dataset("text")` → our formatting `map` → interleaving → TRL's `add_eos` map → tokenize map →
    `bfd` packing;
  - each step writes an Arrow cache file (`HF_DATASETS_CACHE`, default `~/.cache/huggingface/datasets`),
    so disk use is several times the text size;
  - `datasets` reuses a cached `map` only when the function's fingerprint matches. Closures over `cfg`
    (like `preprocess_fn` today) may not hash stably, so a resume could redo all of it.

  Mitigations for v1:
  - pass `SFTConfig.dataset_num_proc` (= `slurm.cpus`, no new key) so maps run in parallel;
  - make the formatting functions module-level, taking their options as `fn_kwargs`, so the cache hashes
    stably;
  - log the time of each step;
  - measure on a realistically sized file in the 50-step test;
  - out of scope unless that test says otherwise: streaming (`IterableDataset`; TRL supports it, but `bfd`
    packing and `all_exhausted` interleaving change meaning), and pre-tokenizing once and sharing it across
    runs.
- **R7 — dev eval loss across settings isn't comparable.** Dev loss depends on `bitext_format` and
  packing, so compare runs only within one setting, and with test BLEU after SFT.
- **Confidentiality:** nothing new goes to W&B. Only loss/perplexity and the data-free config (monotext
  *paths*). The packed batch dump stays in the local log, as for SFT. The quick tests use made-up sentences
  and `wandb.enabled=false`.

## Tests

- `hfmt/test_grid.py` (no torch):
  - a CPT experiment composes; `launch.sh cpt` saves a `run_config.yaml` with `cpt` in
    `<output_root>/<run_name>`;
  - `launch.sh cpt` on an SFT yaml and `launch.sh sft` on a CPT yaml are refused;
  - forced keys: a CPT launch with `override train: longrun` saves `inference_callback_interval: 0`,
    `metric_for_best_model: eval_loss`, `greater_is_better: false`, `packing: true`,
    `completion_only_loss: false` and prints the changes; an SFT launch of the same schedule is untouched;
    a command-line override of a forced key on a CPT launch is refused;
  - `check_config`'s CPT rules: no source enabled, `use_monotext` without files, `monotext_ratio` outside (0, 1)
    with both sources, an unknown `bitext_format`/`packing_strategy`;
  - a `dataset.yaml` with `monotext` composes for an SFT experiment too;
  - resume: a CPT run prints `KIND=cpt_resume`, and a kind mismatch is refused;
  - old saved configs (no `cpt`) still compose, resume, infer and merge;
  - (Q14) an absolute `checkpoint` that doesn't exist is refused.
- `tools/test_run_status.py`: `fix_command` prints `cpt` for a CPT run; a CPT run with
  `model/training_info.yaml` is "finished" with deletable checkpoints.
- `hfmt/test_cpt_monotext_bitext.py` (needs `datasets`, not torch): separate/concat (space) and monotext
  example building, empty-line dropping, dev formatting, and the interleave: with r = 0.25 on two small
  made-up sets, every example of both appears and about 25% of the mixed examples are monotext.
- Cluster:
  - R1 leak test;
  - a 50-step CPT run per model family (`train.max_steps=50 train.eval_steps=25 wandb.enabled=false`).
    Check the batch dump (EOS/BOS between examples), memory, preprocessing time on a large file (R8) and
    the model/ check; then `launch.sh merge` and an SFT run from `model_merged/`;
  - a pre/post-refactor SFT comparison (R5);
  - delete the throwaway run folders afterwards.

## Docs

- README: a short "Continued pre-training" section after "Merging and a second round of SFT" (launch,
  data options, CPT → merge → SFT with a second yaml), and the new `cpt/` group (and `train/cpt`) in the
  option list.
- CLAUDE.md: the structure tree (`cpt_monotext_bitext.py`, `train_common.py`, `conf/cpt/`), one
  convention bullet, `launch.sh <sft|cpt|inf|merge>`.
- `decisions.md`: one entry covering:
  - separate CPT and SFT yamls, and why (option B over a shared yaml);
  - the `cpt` block as the run-kind marker;
  - CPT never decodes (efficiency);
  - the packing strategy chosen after R1;
  - the eval-loss model/ check.

## Questions (all revisited for separate yamls)

- **Q1 — Code sharing: unchanged, and simpler.** Shared code goes in `hfmt/train_common.py`. With CPT
  reading the top-level `train`, the shared functions take `cfg` as SFT's do today; the extra "which
  train block" parameter of v2–v4 is gone.
- **Q2 — How a saved run is marked as CPT: settled by B.** Only CPT experiments have `cpt`, so "config has
  `cpt`" ⇔ CPT run, in composed and saved configs alike. No stripping, no `objective:` key, and old saved
  configs are unaffected.
- **Q3 — Shared `dataset.yaml`: decided; easy.** `data.monotext` can be listed in the shared `dataset.yaml`.
  SFT ignores it (one log line), as CPT ignores `data.test`. The only change from v6 is dropping the check
  that refused `monotext` without `cpt`. Dev loss is on `data.dev` bitext in the run's `bitext_format`.
  CPT reads the same `data.train` bitext as SFT.
- **Q4 — `concat` separator: unchanged.** A space.
- **Q5 — Mixing: decided; the ratio is under `cpt:`.** `cpt.monotext_ratio` lives in `conf/cpt/<option>.yaml`
  and can be overridden in the CPT yaml (`cpt.monotext_ratio=0.3`); `conf/train/` knows nothing about it.
  `interleave_datasets(..., "all_exhausted")` takes its random seed from `train.seed`. That's the run's
  generic seed, which SFT uses too, not a CPT setting, so no train yaml has to change. A separate
  `cpt.seed` is possible if you want to vary the mix without changing the training seed.
- **Q6 — No decoding in CPT: decided; forced, not checked.** With `cpt` set, CPT forces the five SFT-only
  train keys (§1 table) and ignores what the train option says. Any `conf/train/` schedule works for CPT
  unchanged, including `longrun` (which decodes for SFT) and the BLEU-selected ones. There is no extra
  option: the presence of `cpt` is the switch. `grid.py` writes the forced values into the saved config
  and prints what it changed, so the saved config says what ran.
- **Q7 — EOS: unchanged.** The tokenizer's own `eos_token`, with no option. Check the first batch dump (R2).
- **Q8 — Resume variable: unchanged.** `HFMT_SFT_RESUME`.
- **Q9 — Packing strategy: unchanged.** `cpt.packing_strategy` (`bfd` | `wrapped`), `bfd` by default;
  revisit after the R1 test.
- **Q10 — `cpt.train` written out in full: no longer applies.** There is no `cpt.train`; replaced by Q12.
- **Q11 — CPT run folders and example yaml: folders settled, example changed.** CPT runs use the ordinary
  `<output_root>/<run_name>[-<idx>]`; the nested `<run_name>/cpt` layout and its naming changes are gone.
  The example is two new yamls (`cpt_lora.yaml`, and `sft_lora_after_cpt.yaml` whose `checkpoint` is the
  CPT run's `model_merged`). `sft_lora.yaml` is unchanged.
- **Q12 — Where the CPT schedule lives: decided.** In any `conf/train/` option, which stays SFT-only.
  CPT-specific behaviour comes from `cpt:` and the forced keys (Q6). There is no `conf/train/cpt.yaml`, and
  no `conf/cpt` option writes `train:` keys.
- **Q13 — `launch.sh cpt`: decided.** An explicit `launch.sh cpt <cpt yaml>`. `launch.sh sft` refuses a
  config with `cpt` and `launch.sh cpt` refuses one without; resume and `run_status.py` fix commands use
  the matching kind.
- **Q14 — Safeguard for a path `checkpoint`: new, optional, recommended.** `grid.py` refuses an absolute
  `checkpoint` path that doesn't exist. With a CPT → SFT yaml pointing at `.../cpt_lora/model_merged`,
  launching SFT before the merge would otherwise fail only on the GPU, where HF reads the path as a hub id.
  It's cheap, and it does nothing when the path exists.
- **Q15 — `instruction` in CPT yamls: new, minor.** It stays required by the schema, so a CPT yaml sets it
  although CPT doesn't read it. It is used if someone runs `launch.sh inf` on the CPT run. Recommended:
  leave the schema as is (making it optional would change SFT's checks).

### §5 decisions

- **E1:** the `egs/synlrl` yamls use the tracked `conf/peft/lora_tiny_aya_all_linear.yaml` as is (same
  target modules as synth_lrl's untracked `lora_all_linear`, but r=8 instead of 32, so the new runs aren't
  directly comparable with the old synth_lrl ones).
- **E2:** `conf/cpt/default.yaml` = `use_monotext: true`, `use_bitext: true`, `bitext_format: separate`,
  `monotext_ratio: 0.5`, `max_length: 1024`, `packing_strategy: bfd`. What `monotext_ratio: 0.5` means:
  1. **Mixing** (once, before training): `interleave_datasets(..., probabilities=[0.5, 0.5],
     stopping_strategy="all_exhausted")` builds one mixed dataset. Each example is drawn from monotext or
     bitext with probability 0.5 until *both* sources have been used up at least once. The smaller source
     is repeated to keep the 50/50 mix, so every example of both sources is in the mixed set, and the set
     holds about 2 × max(n_mono, n_bitext) examples.
  2. **Training:** the Trainer shuffles that set and draws batches from it. Each batch has *on average*
     half monotext and half bitext examples, not exactly half: the draw is random, and `bfd` packing groups
     examples by length, so one packed row may lean to one source.
  3. **How much is trained** is `train.max_steps` × batch size × gradient accumulation (in packed rows),
     compared with the size of the packed mixed set. Fewer steps than one pass leaves some examples unseen
     (random ones, not one source's); more steps loop over it again.
  4. The ratio counts examples (lines), not tokens. Long monotext lines shift the token share, so the
     script logs both counts and token shares per source.
- **E3:** each pair gets `sft_lora_after_cpt.yaml`.
