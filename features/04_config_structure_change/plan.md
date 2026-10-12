# 04_config_structure_change — Plan

## Goal

Restructure the Hydra config so `conf/` holds only **atomistic, mix-and-match
components**, experiments live under `egs/`, and LoRA can be turned off to fully
fine-tune the base model.

Target component set in `conf/`:

| group        | holds                                                        | change        |
|--------------|-------------------------------------------------------------|---------------|
| `data`       | train/eval data paths, langs, instruction                   | unchanged     |
| `decode`     | generation / eval-decoding knobs                            | unchanged     |
| `basemodel`  | checkpoint, pretrain, quantization (bnb) params             | renamed from `model`, LoRA removed |
| `train`      | general trainer hyperparameters                             | unchanged     |
| `lora`       | `enabled` flag + LoRA adapter params                        | **new**, split out of `model` |

Experiments (`egs/experiment/<name>.yaml`) compose these components and fold in
what used to be separate `wandb` and `sweep` configs (both inline).

## Decisions locked with the developer (2026-10-12)

1. **Full-finetune path: complete.** Not just a toggle — handle dtype, guard all
   peft-only calls, and make inference work on a full (non-adapter) checkpoint.
2. **Experiment location: flat `egs/experiment/<name>.yaml`, discovered via
   `hydra.searchpath`** set in `conf/config.yaml` (works for both `run.sh` and the
   direct `python -m hfmt.sft_translation` path).
3. **wandb: inline into the experiment, drop the group.** Delete `conf/wandb/`;
   experiments set `wandb.*` inline over the `WandbConfig` dataclass defaults.
4. **Sweep: inline into the experiment under a typed `sweep: {enabled, params}`
   block, toggled by `sweep.enabled`.** Delete `conf/sweep/`. Because the submitit
   launcher only runs under `--multirun` and Hydra's basic sweeper has no native
   on/off switch, `run.sh` runs a tiny compose-only probe (`hfmt/sweep_probe.py`,
   no `transformers` import) that reads `sweep` and, only when enabled, emits
   `hydra.sweeper.params.*` overrides into the always-`--multirun` launch. Disabled
   → no grid → one Slurm job. Ad-hoc CLI sweeps still work.
5. **Sweeps range over whole component files, and run labels resolve to the selected
   YAML file names, not the leaf values.** `params` axes are group choices like
   `basemodel: qwen2.5-7b,qwen2.5-1.5b` (any atomistic component), so each job's
   label is `basemodel-qwen2.5-7b__lora-aggressive` — the file names. Leaf-value
   axes (`train.learning_rate: 2e-5,2e-4`) remain allowed.

## Can Hydra find configs in `egs/`? (the developer's question)

**Yes, and not only via `run.sh`.** Add to `conf/config.yaml`:

```yaml
hydra:
  searchpath:
    - file://${oc.env:HFMT_ROOT}/egs
```

Hydra then searches `egs/` for config groups, so `+experiment=mmtc_fr-en_sft1`
resolves to `egs/experiment/mmtc_fr-en_sft1.yaml` under any launch path. The
component groups (`basemodel`, `data`, `train`, `lora`, `decode`) stay in `conf/`;
an experiment references them with `override /basemodel: qwen2.5-7b` etc. across the
combined search path. **Fallback** if env-var interpolation in `hydra.searchpath`
misbehaves on the pinned Hydra version: have `run.sh` pass
`--config-dir "${HFMT_ROOT}/egs"` (verified in group A before committing to the
searchpath approach).

## How the sweep toggle works (the new requirement)

An experiment carries its sweep grid inline in a typed block:

```yaml
# egs/experiment/mmtc_fr-en_sft1.yaml  (@package _global_)
sweep:
  enabled: true
  params:
    basemodel: qwen2.5-7b,qwen2.5-1.5b   # sweep over whole component files
    lora: default,aggressive             # any atomistic component is fair game
    train.learning_rate: 2e-5,2e-4       # leaf-value axes still allowed
```

`run.sh` always launches `--multirun` (submitit needs it) and bridges the flag:

```sh
EXTRA=$(python "${HFMT_ROOT}/hfmt/sweep_probe.py" "$experiment")  # '' when disabled
python -m hfmt.sft_translation --multirun +experiment="$experiment" $EXTRA "$@"
```

`sweep_probe.py` is compose-only (imports `hydra`/`omegaconf` + the light
`hydra_config` schema, **never `transformers`**). It composes
`+experiment=<name>`, reads `cfg.sweep`, and when `enabled` prints one **bare task
override** `<axis>=<csv>` per axis (e.g. `basemodel=qwen2.5-7b,qwen2.5-1.5b
lora.lora_r=8,16`), else nothing. The basic sweeper expands those comma-lists into
the job grid; disabled ⇒ no axes ⇒ a single Slurm job. The CLI can still force a
sweep ad hoc (`bash egs/run.sh <exp> lora.lora_r=8,16`), independent of the flag.

**Spike-verified mechanics (Hydra 1.3.7, group A):**
- `hydra.sweeper.params.<axis>=<csv>` passed on the **CLI is rejected** ("Sweeping
  over Hydra's configuration is not supported"), so the probe must emit **bare**
  `<axis>=<csv>` overrides, not `hydra.sweeper.params.*`.
- Bare overrides sweep both **group choices** (`basemodel=a,b` → one job per
  component file) and **leaf values** (`marker=x,y`), and cross-product correctly.

**Why not purely inline `hydra.sweeper.params` + a boolean?** Hydra reads
`hydra.sweeper.params` directly and ignores any sibling boolean, so a YAML
`enabled: false` could not stop the grid without launch-time logic. Keeping the
grid under our typed `sweep.params` and promoting it in `run.sh` only when enabled
makes the boolean authoritative and keeps the sweep config self-contained and
W&B-loggable (axis names/values are hyperparameters, never data content).

## Work breakdown

### Group A — Schema split (`hfmt/hydra_config.py`)
- Rename `ModelConfig` → `BaseModelConfig`; keep `checkpoint`, `pretrain`,
  `load_in_4bit`, `bnb_4bit_*`. Add `dtype: str = "bfloat16"` for the non-quantized
  (full-FT) load path.
- New `LoraConfig` dataclass: `enabled: bool = True`, `lora_r`, `lora_alpha`,
  `lora_dropout`, `lora_target`. (Schema-side name only; no clash with peft's
  `LoraConfig`, which is imported inside `sft_translation.py`.)
- New `SweepConfig` dataclass: `enabled: bool = False`, `params: Dict[str, str] =
  {}` (axis → comma-separated choices/values). Lives in our schema so it is typed
  and W&B-loggable; the actual grid expansion happens via `run.sh`/`sweep_probe.py`.
- `Config`: replace `model` with `basemodel` + `lora`; add `sweep`; keep `data`,
  `train`, `decode`, `wandb` (wandb stays a schema default so experiments need only
  set `group`/`tags`).
- Verify the pinned-Hydra searchpath behavior here (spike) before group D.

### Group B — `conf/` component groups
- `git mv conf/model/ conf/basemodel/`; strip LoRA keys from
  `basemodel/qwen2.5-1.5b.yaml` and `qwen2.5-7b.yaml`.
- Add `conf/lora/default.yaml` (`enabled: true`, qv / r8 / alpha32 / dropout0.1).
- Edit `conf/config.yaml` defaults list → `basemodel, data, train, lora, decode`
  (drop `wandb`); add the `hydra.searchpath` entry.
- Delete `conf/wandb/` (defaults now come from the `WandbConfig` dataclass).

### Group C — Code wiring + full-FT path (`hfmt/sft_translation.py`)
- Repoint every `cfg.model.*`: base-model reads → `cfg.basemodel.*`; LoRA reads →
  `cfg.lora.*` (checkpoint/pretrain/bnb at ~372–383; lora_target/r/alpha/dropout at
  ~389–406).
- Make **quantization conditional**: only build `BitsAndBytesConfig` and pass
  `quantization_config` when `cfg.basemodel.load_in_4bit`; otherwise load in
  `cfg.basemodel.dtype`.
- Make **LoRA conditional** on `cfg.lora.enabled`:
  - enabled → current path (`prepare_model_for_kbit_training` + `get_peft_model` +
    `print_trainable_parameters`).
  - disabled → skip those; keep `gradient_checkpointing_enable()` and
    `model.config.use_cache = False`; guard the peft-only
    `print_trainable_parameters()` call.
- **Guardrail:** if `lora.enabled=false` and `basemodel.load_in_4bit=true`, fail
  fast with a clear message (QLoRA can't full-FT a frozen quantized base).
- `model.save_pretrained(outdir/model)` already works for both (adapter vs full
  weights) — no change, just confirm.

### Group D — Move experiments to `egs/`
- Create `egs/experiment/`; migrate `mmtc_fr-en_sft1.yaml` into it, rewriting
  `override /model:` → `override /basemodel:` and adding `override /lora: default`.
- Delete `conf/experiment/`.
- Confirm `+experiment=mmtc_fr-en_sft1` resolves via searchpath (both `run.sh` and
  direct `python -m ... --cfg job`).

### Group E — Inline wandb + the sweep toggle
- In `egs/experiment/mmtc_fr-en_sft1.yaml`: keep inline `wandb.group`/`wandb.tags`
  (already there); it uses dataclass wandb defaults for the rest.
- Add the inline `sweep: {enabled, params}` block; port the old coarse grid into it,
  rewriting `model.lora_r`/`model.lora_target` → `lora.lora_r`/`lora.lora_target`
  and (optionally) expressing the base-model axis as a group choice
  `basemodel: qwen2.5-7b,qwen2.5-1.5b`. Default `enabled: false` so a plain
  `run.sh <exp>` is a single run; flip to `true` (or override on CLI) to sweep.
- Write `hfmt/sweep_probe.py` (compose-only, no `transformers`): registers the
  schema + a no-op `hfmt_runlabel` resolver, composes `+experiment=<name>`, and
  prints **bare** `<axis>=<csv>` task overrides when `cfg.sweep.enabled`, else
  nothing (spike: `hydra.sweeper.params.*` CLI overrides are rejected). Must resolve
  via the same `hydra.searchpath`.
- Update `run.sh` to call the probe and inject `$EXTRA` into the `--multirun` launch.
- Delete `conf/sweep/`. Confirm: enabled→grid fans out; disabled→1 job; ad-hoc CLI
  sweeps (`bash egs/run.sh <exp> lora.lora_r=8,16`) still fan out.

### Group F — Run labels resolve to component file names
- Harden `build_run_label` / `override_dirname` so **group-choice** overrides for
  every atomistic component (`basemodel`, `lora`, `train`, `decode`) yield
  `<group>-<option>` labels (file names), e.g. `basemodel-qwen2.5-7b`. Leaf-value
  axes keep `<leaf>-<value>`.
- **Confidentiality nuance (CLAUDE.md):** allow the bare `data=<option>` *config
  file name* in labels (a name/path, not content) but keep dropping dotted
  `data.<leaf>` overrides (e.g. `data.train_yaml=`, which carries a path). Document
  the reasoning inline.
- Confirm the run-dir basename still equals the W&B run name (feature 03 invariant):
  the probe-injected bare axes (`basemodel=a,b`) are expanded by the sweeper into one
  concrete override per job (`basemodel=a`), so each job's `override_dirname` stays a
  clean group-choice/leaf token set.

### Group G — Full sweep of stale references + docs
- Grep the repo for `model\.`, `conf/model`, `conf/wandb`, `conf/sweep`,
  `conf/experiment`, `+sweep=`, `hydra.sweeper.params` and fix: README
  (`### Config layout`, `### Running`, `### Sweeping parameters` — document the new
  `sweep.enabled` toggle and component-file sweeps), `decisions.md`, the untracked
  `egs/mmtc/fr-en/`.
- `inf_translation.py`: full-FT inference already works via the non-PEFT branch
  (load `<run_dir>/model` as `--checkpoint`, no `--peft`). Optionally honor a dtype
  there; document the two load modes. (No adapter-loading change needed.)
- Add a `decisions.md` entry for the restructure; update this feature's
  `todo.md`/validation notes.

## Risks / open questions

1. **Full-FT vs quantization coupling** (handled by the group-C guardrail): "turn
   off LoRA" only makes sense with `load_in_4bit=false`. Full-FT of the 7B on one
   GPU is likely infeasible; realistic target is a small base model. Code will
   *support* it; feasibility is the user's call.
2. **Hard rename, no back-compat alias.** Structured configs reject unknown keys, so
   all `model.*` usage breaks at once (CLI habits, README, sweep presets). Group F's
   grep sweep is the mitigation; acceptable in a small single-team repo.
3. **`hydra.searchpath` env interpolation** on the pinned Hydra — spiked in group A;
   `--config-dir` fallback ready.
4. **Sweep probe is new machinery.** `run.sh` now shells out to
   `sweep_probe.py` before launching. Risks: it must compose with the same
   searchpath/schema the real run uses (else the flag/axes diverge), stay fast
   (compose-only, no `transformers`), and emit CLI-safe tokens (quote comma-lists).
   Mitigation: a focused compose test; the probe shares `hydra_config.register_configs`
   so the schema can't drift.
5. **Group-choice sweeps vs the full-FT guardrail.** A grid that ranges over
   base-model files which differ in `load_in_4bit`, combined with `lora.enabled=false`,
   could trip the group-C guardrail mid-sweep. Acceptable (fail-fast per job), but
   documented so it isn't surprising.
6. **Confidentiality of `data` in labels.** Including a `data` *config file name* in
   run labels/W&B is allowed (name/path, not content); dotted `data.*` path leaves
   stay excluded. Called out so the W&B boundary stays intact.
7. **No automated tests** in this workflow; validation is `--cfg job` compose checks
   for each component combination (default, full-FT, sweep on/off, component-file
   sweep) plus a dry inspection of resolved output and labels. Captured in
   `todo.md`/validation.

## Validation strategy (pre-todo sketch)
- `python -m hfmt.sft_translation +experiment=mmtc_fr-en_sft1 --cfg job` resolves
  with `basemodel`/`lora`/`sweep` present and no `model` key.
- `... lora.enabled=false basemodel.load_in_4bit=false --cfg job` composes; with
  `load_in_4bit=true` it hits the guardrail.
- `sweep_probe.py mmtc_fr-en_sft1` prints the expected `hydra.sweeper.params.*`
  when `sweep.enabled=true` and nothing when false.
- `bash egs/run.sh mmtc_fr-en_sft1` → one job; with `sweep.enabled=true` (or CLI
  axes) → the expected grid, with labels like `basemodel-qwen2.5-7b__lora-...`.
- Grep shows zero remaining `cfg.model.`/`conf/model`/`conf/wandb`/`conf/sweep`
  references.
