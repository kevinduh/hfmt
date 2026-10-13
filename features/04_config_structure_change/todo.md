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

- [x] Repoint base-model reads to `cfg.basemodel.*` (tokenizer at 326; checkpoint,
      pretrain, bnb in the Model Configuration block).
- [x] Repoint LoRA reads to `cfg.lora.*` (lora_target branch; r/alpha/dropout).
- [x] Make quantization conditional on `cfg.basemodel.load_in_4bit`: build + pass
      `BitsAndBytesConfig` only then; otherwise load the model with
      `torch_dtype=getattr(torch, cfg.basemodel.dtype)`.
- [x] Gate the LoRA block on `cfg.lora.enabled`:
      - enabled → `get_peft_model` (+ `prepare_model_for_kbit_training` only on the
        quantized path) + `print_trainable_parameters`.
      - disabled → skip those; `gradient_checkpointing_enable()` (when configured)
        now runs before the branch for both paths; set `model.config.use_cache=False`.
- [x] Guard the peft-only `print_trainable_parameters()` (inside the `lora.enabled`
      branch).
- [x] Add the fail-fast guardrail: `lora.enabled=false` **and**
      `basemodel.load_in_4bit=true` → clear error.
- [x] Confirm `model.save_pretrained(outdir/model)` works for both (PeftModel saves
      an adapter; AutoModelForCausalLM saves full weights) — no code change.
- [x] Verify: no `cfg.model` refs remain; `py_compile` passes; real `--cfg job`
      composes for default and the full-FT override (`lora.enabled=false
      basemodel.load_in_4bit=false`). (Runtime guardrail/full-FT branch not exercised
      — needs GPU + model download.)

## D — Move experiments to `egs/`

- [x] Create `egs/experiment/`.
- [x] Migrate `conf/experiment/mmtc_fr-en_sft1.yaml` → `egs/experiment/` (`git mv`);
      rewrite `override /model:` → `override /basemodel:` and add
      `override /lora: default` + `override /decode: default`.
- [x] Delete `conf/experiment/` (now empty).
- [x] Verify `+experiment=mmtc_fr-en_sft1 --cfg job` resolves via searchpath (7B
      base, seed 37, wandb group/tags applied). NOTE: `--multirun` + `--cfg` can't be
      combined (Hydra CLI rule); `--multirun` doesn't change `+experiment=`
      resolution, so real multirun launch is deferred to integration (needs deps/Slurm).

## E — Inline wandb + the sweep toggle

- [x] In `egs/experiment/mmtc_fr-en_sft1.yaml`: keep inline `wandb.group`/`wandb.tags`.
- [x] Add the inline `sweep: {enabled: false, params: {...}}` block; port the old
      coarse grid, rewriting `model.lora_*` → `lora.lora_*`; documented a
      `basemodel: "a,b"` group-choice axis as the example in the comment.
- [x] Preserve the analysis-derived refine1 sweep (would have been lost with
      `conf/sweep/`) as a first-class experiment
      `egs/experiment/mmtc_fr-en_sft1_refine1.yaml` (fixes all-linear/r8/max_steps=1000,
      sweeps lr×seed, `enabled: true`). [not in original plan; see note below]
- [x] Write `hfmt/sweep_probe.py`: compose-only (import `hydra`, `omegaconf`,
      `hfmt.hydra_config`; **no** `transformers`); register schema + no-op
      `hfmt_runlabel`; compose `+experiment=<name>` (+ forwarded `sweep.enabled`);
      print **bare** `<axis>=<csv>` overrides when `cfg.sweep.enabled`, else nothing.
- [x] Update `egs/run.sh` to call the probe (forwarding only `sweep.enabled=*`) and
      inject `${sweep_axes}` into the always-`--multirun` launch.
- [x] Delete `conf/sweep/`.
- [x] Verify: `sweep.enabled=false` → 1 job; `true` → 24-job grid; refine1 → 8-job
      grid; ad-hoc CLI (`lora.lora_r=8,16`) fans out via passthrough; probe tokens
      correct for all states; concrete grid points compose type-valid.
      Robustness: probe only accepts `sweep.enabled` from CLI (a comma-list
      `sweep.params.*` override crashes single-run compose — use bare overrides).

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
