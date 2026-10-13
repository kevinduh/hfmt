# 04_config_structure_change — Validation

End-to-end validation of the migrated config structure and the example experiments.
Compose-level checks were run off-cluster (see `todo.md`); this file tracks the **on-GPU
training runs**, ordered fastest → slowest so pipeline bugs surface cheaply before the
long sweeps.

Run from a login node with `HFMT_ROOT` exported. Tip: smoke-test any row first with a tiny
budget, e.g. `bash egs/run.sh <exp> train.max_steps=20 train.eval_steps=10`, then drop the
override for the real run.

## Suggested run order (fastest → slowest)

Wall-clock is rough (anchored on ~80 min for 7B QLoRA @ ~800 steps, scaled by model size);
jobs run at most `array_parallelism=4` at once.

| # | Experiment | Model(s) | Jobs | ~Wall-clock | Notes |
|---|------------|----------|------|-------------|-------|
| 1 | `mmtc_fr-en_sft1_fullft` | 0.5B full-FT | 1 | ~20–30 min | Smallest model; also validates the full-FT path |
| 2 | `mmtc_fr-en_sft1` or `_lora` | 1.5B QLoRA | 1 | ~30–40 min | Standard single run; the two are identical — test one |
| 3 | `mmtc_fr-en_sft1_lora_modelsweep` | 0.5B + 1.5B QLoRA | 2 (1 wave) | ~35 min wall | First fan-out; 1.5B job dominates |
| 4 | `mmtc_fr-en_sft1_fullft_modelsweep` | 0.5B + 1.5B full-FT | 2 (1 wave) | ~45 min wall | Now runnable (both fit a single GPU) |
| 5 | `mmtc_fr-en_sft1_lora_model_lr_sweep` | 0.5B/1.5B QLoRA ×3 LR | 6 (2 waves) | ~60–70 min | 1.5B jobs set the pace |
| 6 | `mmtc_fr-en_sft1_refine1` | 7B QLoRA, 1000 steps | 8 (2 waves) | ~200 min | The lone 7B example — by far the slowest |

## Per-experiment checklist

For each run, confirm: it submits to Slurm, trains without error, logs **only aggregate
metrics** to W&B (no dataset text — CLAUDE.md), writes outputs under the expected path, and
the W&B run name/group are as expected.

### 1 — `mmtc_fr-en_sft1_fullft` (0.5B, full fine-tune)
- [ ] `bash egs/run.sh mmtc_fr-en_sft1_fullft` submits and completes
- [ ] loads Qwen2.5-0.5B-Instruct, **no quantization**, no LoRA adapter (full-FT path)
- [ ] W&B run name `mmtc_fr-en_sft1_fullft`, group `mmtc_fr-en_sft1_fullft` (no `-sweep-<ts>`), no `sweep` tag
- [ ] outputs under `outputs/mmtc_fr-en_sft1_fullft/sweep-<ts>/mmtc_fr-en_sft1_fullft/`; full checkpoint saved under `model/`
- [ ] `eval.pred.trg` written; BLEU/CHRF/TER logged to W&B

### 2 — `mmtc_fr-en_sft1` (1.5B, QLoRA single run)
- [ ] `bash egs/run.sh mmtc_fr-en_sft1` submits and completes
- [ ] loads Qwen2.5-1.5B-Instruct, 4-bit QLoRA, adapter enabled
- [ ] W&B run name `mmtc_fr-en_sft1`, group `mmtc_fr-en_sft1` (no date), no `sweep` tag
- [ ] adapter saved under `model/`; predictions + metrics present

### 3 — `mmtc_fr-en_sft1_lora_modelsweep` (0.5B + 1.5B QLoRA)
- [ ] launches **2** Slurm jobs (one per base model)
- [ ] run names `basemodel-qwen2.5-0.5b` / `basemodel-qwen2.5-1.5b`; shared group `mmtc_fr-en_sft1_lora_modelsweep-sweep-<ts>`; `sweep` tag present
- [ ] both jobs complete; distinct output dirs, no clobber

### 4 — `mmtc_fr-en_sft1_fullft_modelsweep` (0.5B + 1.5B full-FT)
- [ ] launches **2** jobs, both full-FT (no quant/adapter), both fit a single GPU (no OOM)
- [ ] run names `basemodel-qwen2.5-0.5b` / `basemodel-qwen2.5-1.5b`; grouped + `sweep`-tagged
- [ ] both complete; full checkpoints saved

### 5 — `mmtc_fr-en_sft1_lora_model_lr_sweep` (2 models × 3 LR)
- [ ] launches **6** jobs (2 waves under the 4-cap)
- [ ] run names combine model + LR, e.g. `basemodel-qwen2.5-1.5b__learning_rate-0.0002`
- [ ] all 6 complete; metrics comparable across the grid in W&B

### 6 — `mmtc_fr-en_sft1_refine1` (7B QLoRA, refinement sweep)
- [ ] launches **8** jobs (lr{4} × seed{2}), 2 waves
- [ ] fixed `lora_target=all-linear`, `lora_r=8`, `max_steps=1000` applied; lr/seed swept
- [ ] all complete; results usable for the refinement analysis

## Cross-cutting checks
- [ ] **Sweep toggle:** `sweep.enabled=false` (or omitted) → 1 job; `sweep.enabled=true` / a
      preset's grid → the expected job count. Ad-hoc `bash egs/run.sh <exp> lora.lora_r=8,16`
      still fans out.
- [ ] **Full-FT guardrail:** `lora.enabled=false` with `basemodel.load_in_4bit=true` fails
      fast with a clear message.
- [ ] **Launcher selection:** full-FT and QLoRA experiments pick up their selected
      `hydra/launcher` profile; CLI `hydra/launcher=local` / `hydra.launcher.timeout_min=…`
      override works.
- [ ] **Confidentiality:** W&B carries only aggregate metrics + data-free config; no source/
      target text, prompts, or hypotheses. Run labels/tags contain no data paths.
- [ ] **Full-FT inference:** load a full checkpoint with `inf_translation.py --checkpoint
      <run_dir>/model` (no `--peft`) and decode.

## Off-cluster (already verified)
- [x] All 7 experiments compose (`--cfg job`); default (no experiment) composes.
- [x] Probes emit the expected sweep axes; concrete grid points compose type-valid.
- [x] `hfmt/test_run_label.py` — 23/23 pass (group-choice labels, data-name confidentiality,
      plain-run → experiment name, fallback cases).
- [x] `${experiment}` interpolates in `hydra.sweep.subdir`; plain run → dir leaf/W&B name =
      experiment, base group, no `sweep` tag (naming fix).
