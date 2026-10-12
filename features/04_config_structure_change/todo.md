# 04_config_structure_change — TODO

Derived from `plan.md`. Groups are ordered by dependency; check items off as
completed. Commit per group using `04_config_structure_change-<phase>: <summary>`.

## A — Schema split + searchpath spike (`hfmt/hydra_config.py`)

- [x] Spike: `hydra.searchpath: [file://${oc.env:HFMT_ROOT}/egs]` IS honored by
      Hydra 1.3.7 — groups on it resolve in the base defaults list and via
      `+experiment=` with cross-path `override /basemodel:`. (env=`hfmt-hydra`)
- [x] Spike: basic sweeper fans out **bare task overrides** `basemodel=a,b` (one job
      per component file) and cross-products with leaf axes. NOTE: passing
      `hydra.sweeper.params.*` on the CLI is **rejected** — the probe must emit bare
      `<axis>=<csv>` overrides instead.
- [x] Rename `ModelConfig` → `BaseModelConfig`; keep `checkpoint`, `pretrain`,
      `load_in_4bit`, `bnb_4bit_*`; **remove** the LoRA fields.
- [x] Add `dtype: str = "bfloat16"` to `BaseModelConfig` for the non-quantized
      (full-FT) load path.
- [x] Add `LoraConfig` dataclass: `enabled: bool = True`, `lora_r`, `lora_alpha`,
      `lora_dropout`, `lora_target` (keep "lora" in the names; no abbreviation).
- [x] Add `SweepConfig` dataclass: `enabled: bool = False`,
      `params: Dict[str, str] = field(default_factory=dict)`.
- [x] Update `Config`: drop `model`; add `basemodel`, `lora`, `sweep`; keep `data`,
      `train`, `decode`, `wandb`.
- [x] Update the module docstring / confidentiality note to mention the new groups.
- [x] Sanity: `register_configs()` imports cleanly, registers all groups, and does
      NOT pull in `transformers` (verified in `hfmt-hydra` env).

## B — `conf/` component groups

- [x] `git mv conf/model/ conf/basemodel/`.
- [x] Strip LoRA keys from `conf/basemodel/qwen2.5-1.5b.yaml` and `qwen2.5-7b.yaml`;
      add `dtype: bfloat16` (comment: used only when `load_in_4bit=false`).
- [x] Create `conf/lora/` group; add `conf/lora/default.yaml`
      (`enabled: true`, `lora_r: 8`, `lora_alpha: 32`, `lora_dropout: 0.1`,
      `lora_target: qv`).
- [x] Edit `conf/config.yaml` defaults list → `base_config, basemodel, data, train,
      lora, decode, _self_` (drop `model`/`wandb`).
- [x] Add `hydra.searchpath: [file://${oc.env:HFMT_ROOT}/egs]` to `conf/config.yaml`
      (searchpath verified in A, no fallback needed).
- [x] Delete `conf/wandb/` (defaults now come from the `WandbConfig` dataclass).
- [x] Compose smoke (compose API): default resolves with `basemodel`/`lora`/`sweep`
      present, `wandb` from the dataclass, and no `model` key.

## C — Code wiring + full-FT path (`hfmt/sft_translation.py`)

- [ ] Repoint base-model reads to `cfg.basemodel.*` (checkpoint, pretrain, bnb at
      ~372–383).
- [ ] Repoint LoRA reads to `cfg.lora.*` (lora_target at ~389–399; r/alpha/dropout
      at ~401–406).
- [ ] Make quantization conditional on `cfg.basemodel.load_in_4bit`: build + pass
      `BitsAndBytesConfig` only then; otherwise load the model in
      `getattr(torch, cfg.basemodel.dtype)`.
- [ ] Gate the LoRA block on `cfg.lora.enabled`:
      - enabled → `prepare_model_for_kbit_training` + `get_peft_model` +
        `print_trainable_parameters` (current path).
      - disabled → skip those; still call `gradient_checkpointing_enable()` (when
        configured) and set `model.config.use_cache = False`.
- [ ] Guard the peft-only `print_trainable_parameters()` so it only runs when LoRA
      is enabled.
- [ ] Add the fail-fast guardrail: `lora.enabled=false` **and**
      `basemodel.load_in_4bit=true` → clear error (QLoRA can't full-FT a frozen
      quantized base).
- [ ] Confirm `model.save_pretrained(outdir/model)` works for both adapter and full
      checkpoints (no code change expected).

## D — Move experiments to `egs/`

- [ ] Create `egs/experiment/`.
- [ ] Migrate `conf/experiment/mmtc_fr-en_sft1.yaml` → `egs/experiment/`; rewrite
      `override /model:` → `override /basemodel:` and add `override /lora: default`.
- [ ] Delete `conf/experiment/`.
- [ ] Verify `+experiment=mmtc_fr-en_sft1 --cfg job` resolves via searchpath both
      through `run.sh` and direct `python -m hfmt.sft_translation`.

## E — Inline wandb + the sweep toggle

- [ ] In `egs/experiment/mmtc_fr-en_sft1.yaml`: keep inline `wandb.group`/`wandb.tags`.
- [ ] Add the inline `sweep: {enabled: false, params: {...}}` block; port the old
      coarse grid, rewriting `model.lora_r`/`model.lora_target` →
      `lora.lora_r`/`lora.lora_target`; optionally add a `basemodel: ...,...`
      group-choice axis as the example.
- [ ] Write `hfmt/sweep_probe.py`: compose-only (import `hydra`, `omegaconf`,
      `hfmt.hydra_config`; **no** `transformers`); register schema + a no-op
      `hfmt_runlabel` resolver; compose `+experiment=<argv[1]>`; if
      `cfg.sweep.enabled`, print one **bare** `<axis>=<csv>` task override per axis
      (NOT `hydra.sweeper.params.*` — CLI-rejected per spike), else print nothing.
      Resolve via the same `hydra.searchpath`.
- [ ] Update `egs/run.sh` to call the probe and inject `$EXTRA` into the
      always-`--multirun` launch.
- [ ] Delete `conf/sweep/`.
- [ ] Verify: `sweep.enabled=false` → 1 job; `true` → expected grid; ad-hoc CLI
      (`lora.lora_r=8,16`) still fans out; probe prints correct tokens for both
      states.

## F — Run labels resolve to component file names

- [ ] Harden `build_run_label`/`override_dirname` so group-choice overrides for
      `basemodel`, `lora`, `train`, `decode` produce `<group>-<option>` labels;
      leaf-value axes keep `<leaf>-<value>`.
- [ ] Confidentiality: allow bare `data=<option>` in labels, keep dropping dotted
      `data.<leaf>` overrides; add an inline comment citing CLAUDE.md.
- [ ] Confirm run-dir basename still equals the W&B run name (feature 03 invariant).
- [ ] Confirm probe-injected `hydra.sweeper.params.*` do not leak into per-job
      `override_dirname` (labels stay clean group-choice tokens).

## G — Stale references + docs

- [ ] Grep + fix `model\.`, `conf/model`, `conf/wandb`, `conf/sweep`,
      `conf/experiment`, `+sweep=`, `hydra.sweeper.params` across the repo.
- [ ] README: update `### Config layout`, `### Running`, `### Sweeping parameters`
      (document `sweep.enabled`, component-file sweeps, and the probe-based toggle).
- [ ] `inf_translation.py`: document the two load modes (adapter via `--peft`;
      full-FT via `--checkpoint <run_dir>/model`); optionally honor a dtype.
- [ ] Add a `decisions.md` entry for the restructure.
- [ ] Update this feature's validation notes; check off completed items.

## Validation (run before marking the feature done)

- [ ] `+experiment=mmtc_fr-en_sft1 --cfg job`: `basemodel`/`lora`/`sweep` present, no
      `model`.
- [ ] `lora.enabled=false basemodel.load_in_4bit=false --cfg job` composes;
      `load_in_4bit=true` hits the guardrail.
- [ ] `sweep_probe.py mmtc_fr-en_sft1` prints the grid tokens when enabled, nothing
      when disabled.
- [ ] `bash egs/run.sh mmtc_fr-en_sft1` → 1 job; enabled/CLI axes → expected grid
      with file-name labels.
- [ ] Grep shows zero remaining `cfg.model.`/`conf/model`/`conf/wandb`/`conf/sweep`.
