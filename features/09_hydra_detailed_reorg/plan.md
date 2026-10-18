# 09_hydra_detailed_reorg — plan (v3, final)

Status: **v3, final**. All decided: the Rule, every key-placement row, Q1–Q10, data location
(`local_setting.sh`). `todo.md` is generated from this version.

## Request (summary)

1. **Where each key lives.** Every key/field gets exactly one home: **A** the `base_config`
   schema (`hfmt/hydra_config.py`), **B** `conf/config.yaml`, **C** `conf/<group>/<option>.yaml`,
   **D** the experiment/data yamls in `egs/mmtc/fr-en/`, or **E** not defined. Goal: clear split
   between general defaults and experiment-specific keys, and less duplication. A vs B/C should be
   (nearly) mutually exclusive; D may still override B/C.
2. **New hierarchy.** A `peft` group (`full_finetune`, `lora_default`, `qlora_default`); the
   `basemodel` group is removed (the checkpoint is set by the experiment); `conf/hydra/launcher`
   deleted; `conf/` ends with only the `decode`, `train`, `peft` groups.
3. **Inference launcher.** A script like `egs/sft.sh` for `inf_translation.py` that takes an
   experiment yaml. It also exercises the reorganized config.

## Starting point (`hydra3` @ 15d5324, feature 08 done)

- `hydra_config.py` gives **every** field a default, and `conf/` repeats most of them. Only
  `run_name` and `data.data_root/src/trg` are `MISSING`. `conf/lora/default.yaml`,
  `conf/train/qlora_default.yaml`, `conf/decode/default.yaml` and the `slurm:` block in
  `conf/config.yaml` duplicate the dataclass defaults. The one real difference is
  `save_steps: ${train.eval_steps}` (yaml) vs `50` (dataclass).
- `basemodel` is a group with 4 options that set only `checkpoint`. The 4-bit fields sit in
  `basemodel` but are read only for `qlora`. `train.finetune_mode` selects `full | lora | qlora`,
  but the `lora` block is always present, even for `full`.
- `inf_translation.py` is still argparse. It always loads a **4-bit** base under an adapter,
  which is wrong for `lora`-mode runs, and it is used by 7 un-migrated recipes
  (`egs/mmtc/zh-en/inf.sh`, `egs/synth_lrl/*/inf{1,2}.sh`). Those recipes, and
  `egs/mmtc/zh-en/sft1.sh`, have been stale since 08 moved `sft_translation.py` to Hydra.
- `conf/hydra/launcher/` holds **untracked** hydra2 leftovers (`slurm_h100.yaml`, its `~`
  backup, `slurm_v100.yaml`) for the submitit launcher, which hydra3 doesn't use.

## Rule (decided)

> **A holds types only, with no default values. Every default lives in exactly one yaml.**

- A defines every key and its type. Each value field is `MISSING`. A whole optional block is
  `Optional[...] = None`, meaning "absent", which is not a default value.
- B holds settings shared by the whole project (slurm, wandb, Hydra settings) and the defaults list.
- C holds named, swappable bundles (one file = one complete option).
- D holds what makes this experiment this experiment.
- Because A has no values, a key that no yaml sets fails in `grid.py`'s missing-values check on
  the login node, instead of silently using a forgotten fallback.

This makes A vs B/C fully exclusive. The cost: reading A no longer shows default values. You
read them in `conf/`, which is where they're edited anyway.

## Key placement (all rows confirmed)

"Request" is what `request.md` says. "Proposal" is the same unless marked ⚠. Key paths shown are
the *current* ones. New paths follow Q1 (H2).

| Key (current path) | Request | Proposal | Notes |
|---|---|---|---|
| `run_name` | D | D (A: `MISSING`) | unchanged |
| `output_root` | E | ⚠ kept: A `Optional[str] = None`, set by no yaml (Q5 (a)) | launch setting read only by `grid.py`; set on the command line or in D to redirect output; `grid.py` saves the resolved path so renamed re-runs of a saved config land next to the original |
| `basemodel.checkpoint` | D, A `MISSING` | D, A `MISSING` | new path: top-level `checkpoint` |
| `basemodel.load_in_4bit` | C `peft/qlora_default` | C `peft/qlora_default` | new path `peft.quant.load_in_4bit` (Q2: kept) |
| `basemodel.bnb_4bit_use_double_quant` | C `peft/qlora_default` | C `peft/qlora_default` | |
| `basemodel.bnb_4bit_quant_type` | C `peft/qlora_default` | C `peft/qlora_default` | |
| `lora.lora_r`, `lora_alpha`, `lora_dropout`, `lora_target` | C `peft/{qlora,lora}_default` | C `peft/{qlora,lora}_default`; absent for `full_finetune` | a `full` run has no lora block (A: `Optional = None`) |
| `train.finetune_mode` | C, in every `peft/*` | C, in every `peft/*` | new path: `peft.mode` |
| `train.*` (other 23 fields) | C | C `conf/train/default.yaml` (renamed from `qlora_default.yaml`) | `save_steps: ${train.eval_steps}` stays there |
| `train.seed` | C | C (42), D overrides (37 in `sft_lora`) | as now |
| `data.data_root` | D | D: `dataset.yaml` sets `${oc.env:HFMT_DATA_ROOT}` (A: `Optional = None`; a dataset yaml with absolute paths may omit it) | value comes from the untracked `local_setting.sh` (see "Data location") |
| `data.src`, `data.trg` | D | D: `dataset.yaml` (A: `Optional = None`, like `data_root`) (Q10) | fixed per folder, the same on every machine, so tracked; only consumer is `dataset.yaml`'s file-name templates |
| `data.instruction` | D | D: top-level `instruction` in the experiment (A: `MISSING`; today A has `""`) (Q10) | every experiment states its prompt |
| `data.train`, `data.dev` | D (`dataset.yaml`) | D (A: `MISSING`) | |
| `data.test` | D (`dataset.yaml`) | D (A: `Optional = None`) | no test → no final decoding (unchanged) |
| `decode.*` | C, as is | C `conf/decode/default.yaml` (A: `MISSING`) | |
| `wandb.enabled`, `project`, `entity` | B (Q6) | B | project-wide settings like `slurm` |
| `wandb.tags` | D | D (B: `tags: []`) | |
| `slurm.*` | B | B (A: `MISSING`, today A repeats B) | |
| `hydra.job.chdir` | — | B, unchanged | Hydra setting, not in the schema |
| `inf_name` (new) | — | B: `test` (A: `MISSING`) | inference output subfolder (Q4b); read only by `grid.py --inf` |
| `model_dir` (new) | — | not set by any yaml (A: `Optional[str] = None`) | set by `grid.py --inf` to `<run folder>/model` in `inf_config.yaml`; read only by `inf_translation.py`, which fails if it is unset. Same pattern as `output_root` (Q5) |

## Hierarchy (Q1, decided: H2, top-level `checkpoint`): how the `peft` group maps onto keys

A group's files normally land under the group's own key (`peft/x.yaml` → `peft.*`). The request
moves `finetune_mode`, the 4-bit fields and `lora.*` into `peft` files. There are two ways to do
that.

**H1: keep today's key paths.** The `peft/*.yaml` files use `# @package _global_` and write into
three places:

```yaml
# conf/peft/qlora_default.yaml
# @package _global_
train: {finetune_mode: qlora}
quant: {bnb_4bit_use_double_quant: true, bnb_4bit_quant_type: nf4}  # where? basemodel is gone
lora: {lora_r: 8, lora_alpha: 32, lora_dropout: 0.1, lora_target: qv}
```
Command line stays `lora.lora_r=16`, so less churn. But the group name no longer matches a key,
which breaks the "group → its own key" rule we went through, and the 4-bit fields still need a
new home.

**H2 (decided): `peft` is both the group and the key.**

```yaml
# conf/peft/qlora_default.yaml          # conf/peft/lora_default.yaml   # conf/peft/full_finetune.yaml
mode: qlora                             mode: lora                      mode: full
lora:                                   lora:
  lora_r: 8                               lora_r: 8
  lora_alpha: 32                          ...
  lora_dropout: 0.1
  lora_target: qv
quant:
  load_in_4bit: true
  bnb_4bit_use_double_quant: true
  bnb_4bit_quant_type: nf4
```
```python
@dataclass
class PeftConfig:
    mode: str = MISSING                  # full | lora | qlora
    lora: Optional[LoraConfig] = None    # required for lora/qlora
    quant: Optional[QuantConfig] = None  # required for qlora
```
- Command line: `peft=lora_default`, `peft.lora.lora_r=16`; sweeps use
  `peft: lora_default,qlora_default`. ⚠ This renames sweep keys (`lora.lora_r` → `peft.lora.lora_r`).
- Checked with OmegaConf 2.3.1: a yaml merged into an `Optional[LoraConfig] = None` node is still
  type- and key-checked (`lora_r: abc` and `lora_rr: 8` both fail). `peft: {mode: full}` leaves
  `lora` as `None`.
- A consistency check (`mode` agrees with which blocks are present) runs in `grid.py` before
  submitting, e.g. `peft=full_finetune peft.lora.lora_r=16` fails with a clear message.
- Optional: rename `lora_r/lora_alpha/...` → `r/alpha/...` under `peft.lora`. Recommend **no**,
  since you asked for descriptive names in hydra2.

**Checkpoint key (decided):** a top-level `checkpoint: Qwen/Qwen2.5-1.5B-Instruct`, set by the
experiment (A: `MISSING`).

### Resulting layout

```
conf/config.yaml                 # B: defaults [base_config, train: default, peft: qlora_default,
                                 #    decode: default, _self_]; slurm:, wandb:, hydra.job.chdir
conf/train/default.yaml          # C (renamed from qlora_default.yaml; finetune_mode removed)
conf/peft/{full_finetune,lora_default,qlora_default}.yaml   # C
conf/decode/default.yaml         # C
deleted: conf/basemodel/, conf/lora/, conf/train/qlora_default.yaml, conf/hydra/ (untracked, Q8)
```
Experiment (`egs/mmtc/fr-en/sft_lora.yaml`) under H2:
```yaml
defaults:
  - config
  - dataset
  - override peft: qlora_default    # optional while it equals the default
  - _self_
run_name: sft_lora
checkpoint: Qwen/Qwen2.5-1.5B-Instruct
instruction: "Translate French to English"
# data_root, src, trg come from dataset.yaml (+ local_setting.sh for data_root)
train: {seed: 37}
wandb: {tags: [sft, fr-en, lora]}
```
`sft_lora.sweep.yaml` changes its axis to `peft.lora.lora_r: 8,16`.

### Data location (decided): per-folder, untracked `local_setting.sh`

Where the data lives is machine-specific; which files make up the dataset is not. So:

```sh
# egs/mmtc/fr-en/local_setting.sh          (gitignored: egs/**/local_setting.sh; one per machine)
export HFMT_DATA_ROOT=/path/to/data/fr-en
```
```yaml
# egs/mmtc/fr-en/dataset.yaml              (tracked, generic)
data_root: ${oc.env:HFMT_DATA_ROOT}
```
- `egs/launch.sh sft` (formerly `egs/sft.sh`) runs `unset HFMT_DATA_ROOT`, then sources
  `<experiment folder>/local_setting.sh` if it exists, before calling `grid.py`. The `unset` keeps a
  value left in your shell from silently pointing a different folder at the wrong data.
- A tracked `local_setting.sh.example` (one line) sits next to each `dataset.yaml`. When the variable
  is missing, `grid.py` turns OmegaConf's error into a hint: create
  `<experiment folder>/local_setting.sh` (see `.example`).
- It is needed **only at launch**. `grid.py` saves resolved absolute paths in `run_config.yaml`, so
  compute nodes, re-runs of a saved config and inference on a run folder don't need it.
- Direct debugging runs (`python -m hfmt.sft_translation --config-dir egs/mmtc/fr-en ...`) don't go
  through `launch.sh`, so the experiment header comment says to `source egs/mmtc/fr-en/local_setting.sh`
  first.
- `tools/run_on_free_gpu.sh` goes through the launcher (now `egs/launch.sh <sft|inf>`). `egs/sft_job.sh` needs none
  either. `test_grid.py` sets a dummy `HFMT_DATA_ROOT` (`grid.py` doesn't check that files exist).
- One variable name in every folder: only one folder's file is sourced per launch.

**Q3, full fine-tune hyperparameters (decided).** `peft/full_finetune.yaml` sets only `peft.*`.
Training settings suited to a full fine-tune (e.g. a lower learning rate) will later become their
own `conf/train/<option>.yaml`, which the developer will add; that is out of scope for 09.

## Inference (request part 3)

### What exists
`inf_translation.py` (argparse: `-e evalset -c checkpoint -p adapter -o outdir -i instruction`)
always loads a 4-bit base under an adapter, uses hard-coded decode settings (128/greedy/batch 16)
and writes `eval.inf.trg`. A saved `<run folder>/run_config.yaml` plus `<run folder>/model/`
already hold everything needed to decode with a trained model.

### Proposal
- **Shared model code.** Move `format_input_prompt`, `inference_on_eval_data` and a new
  `load_model(cfg)` (the qlora/lora/full branches now in `sft_translation.py`) into a small module,
  e.g. `hfmt/mt_model.py`. Training and inference then load the base the same way, which fixes the
  4-bit-on-`lora` bug. `load_model(cfg, adapter_dir=...)` adds `PeftModel.from_pretrained` for
  inference; `full` loads `<run folder>/model` directly.
- **`inf_translation.py` on `@hydra.main`**, same schema, run on a saved config:
  `python -m hfmt.inf_translation --config-dir <run folder> --config-name run_config hydra.run.dir=...`.
  It decodes `data.test` with the `decode.*` settings and writes predictions. If `data.test.trg`
  is given, it also writes BLEU/chrF/TER to a local file (**Q7**). No W&B for inference
  (confidentiality is simpler; aggregate scores could be added later).
- **`egs/launch.sh inf <run folder>/run_config.yaml [overrides]`**: always a single run (one trained
  model per call; revised after phase A, see Q4a). Same `LOCAL=1`/`DRYRUN=1` as training.
  **One launcher (decided in phase C):** `egs/sft.sh` and the planned `egs/inf.sh` are merged into
  `egs/launch.sh <sft|inf>`, so `egs/` has three launch files: `launch.sh`, `sft_job.sh`, `inf_job.sh`.
  - `grid.py --inf` (about 15 lines): the input must be a `run_config.yaml` and comma lists are
    refused, so there is exactly one job. It composes the saved config + your overrides +
    `model_dir=<run folder>/model`, runs the usual checks on the login node (typos fail before
    submitting), checks that `model/` exists, and saves `<run folder>/inf/<inf_name>/inf_config.yaml`.
    It never touches `run_config.yaml`. Same `JOBS`/`SBATCH_ARGS` output as for training.
  - No `local_setting.sh` needed: saved configs hold resolved paths.
  - Examples: `bash egs/launch.sh inf egs/mmtc/fr-en/outputs/sft_lora/run_config.yaml decode.num_beams=5 inf_name=beam5`,
    or another test set via `'data.test.src=[...]' 'data.test.trg=[...]' inf_name=other`.
  - A whole sweep is a shell loop (one `sbatch` per folder):
    `for rc in egs/mmtc/fr-en/outputs/sft_lora_sweep-*/run_config.yaml; do bash egs/launch.sh inf $rc ...; done`.
- **One job script per launcher** (developer's naming): `egs/job.sh` is renamed `egs/sft_job.sh`
  (run by `launch.sh sft`), and a new `egs/inf_job.sh` (run by `launch.sh inf`) runs
  `hfmt.inf_translation` on `inf_config`. Each hard-codes its module and config name, so neither
  needs extra arguments.

### Inference decisions (Q4a–e, all decided as recommended)
- **Q4a (revised):** inference (`launch.sh inf`) takes only a run folder's `run_config.yaml`, one model per call
  (originally (i): experiments and sweeps too; changed to (ii) to keep `grid.py --inf` small; a
  sweep is decoded with a shell loop). A separate inference-experiment kind is not added.
- **Q4b:** output in `<run folder>/inf/<inf_name>/`. `inf_name` is a new top-level key, default
  `test` in B (`conf/config.yaml`), set on the command line or in D. Under the Rule, A has it as
  `MISSING`. Only `grid.py --inf` reads it.
- **Q4c:** zero-shot decoding of an un-fine-tuned checkpoint is out of scope for 09.
- **Q4d:** `inf_translation.py`'s argparse CLI is replaced in place. The 7 un-migrated `inf*.sh`
  recipes become stale, like `zh-en/sft1.sh` after 08.
- **Q4e:** same `slurm:` block; override per call (e.g. `slurm.time=1:00:00`). No separate
  inference default.

## Open questions (summary)

| # | Question | Recommendation |
|---|---|---|
| Q1 | Hierarchy for `peft` (H1 `_global_` vs H2 `peft.*`), and the checkpoint key path | **Decided:** H2; top-level `checkpoint` |
| Q2 | Drop `load_in_4bit` (implied by the `quant` block)? | **Decided:** keep, as `peft.quant.load_in_4bit` |
| Q3 | Should `full_finetune.yaml` also set training hyperparameters? | **Decided:** no; a later `conf/train/` option (developer) |
| Q4a–e | Inference input, output naming, zero-shot, old CLI, resources | **Decided** (see "Inference decisions"); Q4a revised to single-run (`run_config.yaml` only) |
| Q5 | `output_root` → E: how to redirect output (tests, scratch) | **Decided:** (a) keep the key unchanged (A `Optional = None`, no yaml sets it); default `<experiment folder>/outputs` |
| Q6 | `wandb` defaults in A or B | **Decided:** B (`enabled/project/entity/tags: []`); D sets `tags` |
| Q7 | Score test predictions in inference (local file only)? | **Decided:** yes, when `data.test.trg` is given |
| Q8 | Delete untracked `conf/hydra/` (3 files, not in git, so not recoverable)? | **Decided:** delete (phase A) |
| Q10 | Where `src`, `trg`, `instruction` live | **Decided:** `instruction` is a top-level experiment key; `src`/`trg` go in `dataset.yaml` (under `data:`) |
| Q9 | Saved configs from 08 (`egs/mmtc/fr-en/outputs/*/run_config.yaml`) won't validate against the new schema | **Decided:** no migration; the developer deletes the old run folders |

## Risks

- **Old saved configs (Q9).** 08 run folders' `run_config.yaml` won't validate against the new
  schema. Not migrated; the developer deletes them.
- **W&B column names change.** The config is logged under `hfmt.*`, so 08 runs (`hfmt.lora.lora_r`)
  and 09 runs (`hfmt.peft.lora.lora_r`) won't line up in W&B tables. Grouping across the two needs
  care.
- **No defaults in A.** A direct `python -m hfmt.sft_translation` without `- config` in the
  defaults fails with missing values. That's intended, but it is new behavior.
- **Order of a group swap vs an experiment value.** Setting `peft.lora.*` in an experiment while
  choosing `peft=full_finetune` creates a partial lora block. The consistency check (Q1/H2) must
  catch it with a clear message.
- **qlora inference must load the same 4-bit base as training**, or quality silently changes.
  Shared `load_model(cfg)` addresses this. The test run checks dev-set BLEU from training against
  an inference re-decode.
- **Confidentiality.** Inference writes predictions only locally. No W&B call in `inf_translation.py`.
  The scoring file holds only aggregate numbers.

## Phases

One commit per phase (`09_hydra_detailed_reorg-<phase>: ...`), each with its own check.

| Phase | Content | Done when |
|---|---|---|
| A | Schema to types-only (Rule) + H2 hierarchy; `conf/` reorg (`peft/`, `train/default`, slurm/wandb in B, deletions); experiments updated; `local_setting.sh` (+ `.example`, `.gitignore`, sourced by `sft.sh`, hint in `grid.py`); `sft_translation.py` key paths; consistency check; `test_grid.py` updated | `--cfg job` for `sft_lora` equals the 08 output, except the intended renames (diffed); tests pass; each `peft` option composes; bad combinations fail in `grid.py` |
| B | `hfmt/mt_model.py` (shared prompt/decode/`load_model`); `sft_translation.py` uses it; `inf_translation.py` on Hydra with optional scoring | 20-step `LOCAL=1` SFT for each of `full`, `lora`, `qlora` on a GPU node; direct inference on each run folder reproduces the final test predictions |
| C | `grid.py --inf` (single run), `egs/launch.sh <sft|inf>` (replaces `sft.sh`), `egs/inf_job.sh`, `job.sh` → `sft_job.sh`; tests for inference mode | `DRYRUN`/`LOCAL` inference on a run folder lands in `<run folder>/inf/<inf_name>/`; `run_config.yaml` untouched; typo, comma list and non-`run_config` input fail before submitting |
| D | Docs: CLAUDE.md, README, `decisions.md` entry | docs match code |
| E | Slurm validation: short sweep with `peft: lora_default,qlora_default`, then `launch.sh inf` on each run folder (shell loop) | done by the developer, as in 08 |
