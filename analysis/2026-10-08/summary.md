# Analysis — `mmtc_fr-en_sft1` QLoRA SFT sweep

**Date:** 2026-10-08
**Source:** `wandb_export.json` (W&B project `hfmt`, entity `jrokisk1-johns-hopkins-university`,
exported 2026-10-08, filter `state=finished`, `limit=100`)

> Aggregate metrics and config only — no dataset content or model hypotheses.

## What the runs were

25 QLoRA SFT runs on **`Qwen/Qwen2.5-7B-Instruct`** for **French→English** translation
(experiment `mmtc_fr-en_sft1`). 4-bit nf4 quantization, `max_steps=800`, eval every 50 steps,
`metric_for_best_model=eval_loss`. Each run records `config`, `steps`, and a `trajectory` of
eval/train metrics over steps.

## Sweep structure

A **2×3×2×2 grid (24 runs) + 1 baseline**:

| Axis | Levels |
|---|---|
| `train.learning_rate` | 2e-5, 2e-4 |
| `model.lora_r` | 8, 16, 32 |
| `model.lora_target` | qv, all-linear |
| `train.seed` | 37, 42 |

The lone **baseline** (group `mmtc_fr-en_sft1`, not a sweep group): lr=2e-4, r=8, qv, seed=37 → BLEU 68.9.

## Results

All runs are strong and tightly clustered: best BLEU **67.6–69.9**, CHRF ~75–77, TER ~28–30.

Top configs by best eval BLEU:

| lr | r | target | seed | best BLEU | @step | best CHRF | best TER | best eval_loss |
|---|---|---|---|---|---|---|---|---|
| 2e-4 | 32 | all-linear | 42 | **69.87** | 160 | 76.80 | 27.89 | 0.4499 |
| 2e-4 | 8  | all-linear | 42 | 69.85 | 160 | 76.84 | 27.85 | 0.4498 |
| 2e-4 | 16 | all-linear | 42 | 69.77 | 160 | 76.69 | 28.03 | 0.4493 |
| 2e-4 | 16 | qv         | 42 | 69.57 | 460 | 76.96 | 28.26 | 0.4588 |
| 2e-4 | 8  | qv         | 42 | 69.57 | 460 | 76.86 | 28.19 | 0.4586 |

Worst cell: `2e-5 + qv` (best BLEU 67.6–68.0 across all its runs).

### Marginal mean best-BLEU by axis (24 sweep runs)

| Axis | Level | Mean BLEU |
|---|---|---|
| lr | 2e-4 | 69.41 |
| lr | 2e-5 | 68.46 |
| target | all-linear | 69.33 |
| target | qv | 68.54 |
| lora_r | 8 | 68.99 |
| lora_r | 16 | 68.94 |
| lora_r | 32 | 68.88 |
| seed | 42 | 69.09 |
| seed | 37 | 68.78 |

### lr × target interaction

| | all-linear | qv |
|---|---|---|
| **lr=2e-4** | 69.50 | 69.32 |
| **lr=2e-5** | 69.16 | **67.77** |

## Key findings

1. **`lora_r` is irrelevant here.** Means 68.88 / 68.94 / 68.99 for r=32/16/8 — within noise, and
   the rank is inverted. Use **r=8** and save the compute.

2. **The only real effects are `learning_rate` and `lora_target`, via their interaction.** `qv`
   only hurts when the LR is *also* low (`2e-5+qv` = 67.77); the other three cells are all ~69.2–69.5.

3. **800 steps is mistuned for both ends.** High-lr `all-linear` peaks at **step ~150** then
   plateaus/overfits (≈650 wasted steps); low-lr `qv` reaches its best BLEU only at **step 800**
   (not yet converged). Use a shorter budget for the fast configs, or drop the 2e-5 arm.

4. **Checkpoint selection leaves BLEU on the table.** `metric_for_best_model=eval_loss`, but the
   min-eval-loss step ≠ the max-BLEU step. For `2e-4 + qv` runs, selecting by eval_loss loses up to
   **~1.15 BLEU** vs the BLEU-optimal checkpoint — eval_loss and BLEU diverge. Consider selecting on
   BLEU (or saving a separate best-BLEU checkpoint).

5. **Seed effect (~0.3 BLEU)** is smaller than the lr/target effect (~1 BLEU), but the whole sweep
   spans only ~2.3 BLEU — treat sub-0.5-BLEU gaps between good configs as noise.

## Recommendation

Use **lr=2e-4, lora_r=8, lora_target=all-linear**, trained ~150–200 steps. The next sweep should
drop `lora_r` entirely, drop the 2e-5 arm (or pair it with a longer budget), and reduce `max_steps`.
