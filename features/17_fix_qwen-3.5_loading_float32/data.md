# 17_fix_qwen-3.5_loading_float32 — data

Aggregate numbers only (dtype counts, scores, memory, times). Tests are defined in `plan.md`.

## Verification 1 — load check, before the code change (2026-11-19, CPU job 1253827)

`AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float32)`, pinned Transformers, with and
without `config=AutoConfig.from_pretrained(checkpoint)`. Parameter counts by dtype:

| Checkpoint | Load | Parameters |
|---|---|---|
| Qwen/Qwen3.5-0.8B | plain | 320 × bfloat16 |
| Qwen/Qwen3.5-0.8B | with config | 320 × float32 |
| CohereLabs/tiny-aya-global (control) | plain | 290 × float32 |
| CohereLabs/tiny-aya-global (control) | with config | 290 × float32 |

Confirmed: passing the config makes Qwen3.5 load in the requested float32, and the control is unchanged
in dtype. This is what the suspected mechanism predicts (the auto class loses `dtype` when it swaps
Qwen3.5's composite config for its text sub-config).
Not shown: that a float32 Qwen3.5 trains and saves float32 (Verification 2), Gemma 3 or the bf16
(`lora`) load with an explicit config, and that the control's weight values are identical both ways.

## Verification 2 — 50-step throwaway full fine-tune after the code change (2026-11-19)

`dtypetest_fp32_qwen0.8b`: `egs/mmtc/fr-en/sft_qwen-3.5-0.8b.yaml` with `train.max_steps=50
train.eval_steps=25 wandb.enabled=false`, run with the `load_base_model` change and the dtype guard.

| Check | Result |
|---|---|
| Job got past model loading (guard requires all floating-point parameters float32) | yes: the run finished |
| `model/config.json` `dtype` | `float32` |
| `model/training_info.yaml` `model_dev_check` | ok |
| Dtypes printed in the training log | 321 × float32, 1 × bfloat16 (320 parameters + 1 log line float32; the bfloat16 is the `compute_dtype` log line) |
| "Accelerate precision" | bf16, scaler None (mixed precision unchanged) |
| `model/` size | 2.9G, against 1.5G for the bf16-weight `sft_full_qwen-3.5-0.8b_short/model` (1.9×) |
| eval loss, step 25 → 50 | 0.6015 → 0.5735 |
| train loss (50 steps) | 0.6119 |
| Peak allocated memory | 14.47 GiB |
| Train runtime | 483.5 s |

Shown: with `peft.mode: full`, Qwen3.5-0.8B now trains with float32 weights (all 320 parameters float32
in the log, none bfloat16; the guard passed) under the same bf16 mixed precision as before, saves
float32 weights (config dtype and a `model/` twice the size of the bf16 run's), and trains sensibly
(loss falls, no NaN). The saved model's fresh dev decode matches the dev score of the step it holds, so
the best-checkpoint reload works in float32. This is the feature's definition of "works".
Not shown: anything about quality at full length (Verification 3), memory for the 2B and 4B, resume from
a float32 checkpoint, and the `lora`/`qlora` loads with the explicit config.

Cluster unit tests (2026-11-19): `python hfmt/test_sft_translation.py` 6 passed.

## Verification 2b — `lora` and `qlora` loads with the explicit config (2026-11-19)

25-step throwaway runs of `egs/mmtc/fr-en/sft_lora.yaml` (Qwen3.5-2B), W&B off.

| Run | Dtypes printed in the log | Finished | `model_dev_check` |
|---|---|---|---|
| `dtypetest_lora_qwen2b` | 321 × bfloat16 (320 base parameters + the `compute_dtype` line), 24 × float32 (adapters) | yes, guard passed | ok |
| `dtypetest_qlora_qwen2b` (`peft=qlora_default`) | 186 × uint8 (4-bit weights), 158 × float32, 1 × bfloat16 (log line) | yes | ok |

Shown: the `lora` load still gives a bf16 base (the guard checks it) and the `qlora` load still trains,
decodes and passes its dev check. Not compared with a pre-change run of the same 25 steps.

## Verification 3 — full-length comparison, Qwen3.5-2B fr-en (2026-11-19)

`sft_full_qwen-3.5-2b_short_fp32` (after the change) against the existing `sft_full_qwen-3.5-2b_short`
(2026-11-05, bf16 weights). Same experiment yaml, seed, schedule (150 steps, cosine), learning rate 1e-5
and batch 32 × 2. Metric names differ only by the 2026-11-18 rename (`bleu` → `sp_bleu`,
`bleu_13a_lowercase` → `bleu_13a`; same definitions).

Dtype checks on the new run: log 321 × float32 and 1 × bfloat16 (the `compute_dtype` line), "Accelerate
precision: bf16", `model/config.json` `float32`, `model_dev_check: ok`, no errors.

| | bf16 weights (old) | fp32 weights (new) | Difference |
|---|---|---|---|
| Test sp_bleu | 70.97 | 69.96 | −1.01 |
| Test bleu_13a | 65.63 | 64.21 | −1.42 |
| Test chrF | 77.30 | 76.46 | −0.84 |
| Test TER | 26.61 | 27.78 | +1.17 |
| Dev bleu_13a of `model/` | 62.57 | 61.54 | −1.03 |
| Best step (by dev bleu_13a) | 75 (62.62) | 125 (61.68) | |
| Steps run | 125 (early stop) | 150 | |
| Peak allocated memory | 17.38 GiB | 34.65 GiB | 2.0× |
| Peak reserved memory | 19 GB | 40 GB | |
| Card | A100 40GB | A100 80GB | |
| `model/` size | 3.6G | 7.1G | 2.0× |
| Train runtime | 1287.5 s (125 steps) | 1840.6 s (150 steps) | 10.3 → 12.3 s/step, evaluations included, different card models |

Dev eval loss by step:

| Step | 25 | 50 | 75 | 100 | 125 | 150 |
|---|---|---|---|---|---|---|
| bf16 weights | 0.5086 | 0.5001 | 0.4996 | 0.5268 | 0.5281 | |
| fp32 weights | 0.5376 | 0.6007 | 0.5750 | 0.7092 | 0.7210 | 0.7868 |

fp32 dev bleu_13a by step: 60.16, 61.54, 61.54, 60.79, 61.68, 61.44 (the old run's per-step dev BLEU is
stored under the old key names and was not read here; its best was 62.62 at step 75).

Shown:
- The fix holds at full length: float32 weights trained and saved, with the same mixed precision.
- Memory and disk double, as estimated. **The fp32 2B run does not fit the 40 GB A100 the old run used**
  (34.65 GiB allocated, 40 GB reserved): the 2B full yamls now set `slurm.constraint: 80gb`.
- At the same learning rate the fp32 run scores about 1 to 1.4 BLEU lower on test (above the 0.72
  seed-to-seed spread measured on fr-en) and its dev loss is higher at the first evaluation and climbs
  to 0.79, where the bf16 run stayed near 0.50 to 0.53.

Not shown: why. Suspected, untested: lr 1e-5 was chosen on bf16-weight runs, where updates much smaller
than a weight's bf16 spacing round away, so the same nominal rate moves fp32 weights further and the run
overfits within the first 25 steps. A lower learning rate (and an evaluation before step 25) would test
it. One seed, one model, one pair: this does not say bf16 weights are better in general.
