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
