# 17_fix_qwen-3.5_loading_float32 — plan (v3, simplified)

Status: 2026-11-19. Nothing run yet; the cause is from reading the pinned Transformers (v5.8.0).

**Goal:** with `peft.mode: full`, Qwen3.5 trains with float32 weights and saves float32 weights. Then
one full-length fine-tune to compare with the existing bf16-weight run. Existing results and old runs
are out of scope.

## Cause (suspected; step 1 confirms it)

`AutoModelForCausalLM.from_pretrained(checkpoint, dtype=float32)` first builds the config and moves
`dtype` onto it. Qwen3.5's config is composite, and the auto class then swaps it for its text
sub-config, which stores `"dtype": "bfloat16"` (see `request.md`). The model class never sees the
request and loads bf16. Tiny Aya (plain config) and Gemma 3 (keeps the composite config) are not
affected. Code: `auto_factory.py:340` and `:395-399`, `configuration_utils.py:839-854`.

## Change (`hfmt/mt_model.py` only)

1. `load_base_model`: pass `config=AutoConfig.from_pretrained(cfg.checkpoint)` to `from_pretrained`.
   The auto class then leaves `dtype` in the arguments and the model class uses it. For the other
   families this is the same config the auto class builds itself.
2. `check_parameter_dtype(model, expected, checkpoint)`: raise if any floating-point parameter has
   another dtype. Called after a `full` or `lora` load in `load_base_model`, and after the `full` load
   in `load_trained_model`. A wrong precision then stops the job at load time.
3. Unit test for the guard in `hfmt/test_mt_model.py`.

If step 1 shows the explicit config does not help: load as today, then `model.to(model_dtype)`
(bf16 → fp32 is exact), same guard.

## Verification (cluster; W&B off except the comparison run)

1. **Load check (CPU, one command, before the code change).** Count parameter dtypes for a float32
   load with and without the explicit config:

       source install/path.sh && python - Qwen/Qwen3.5-0.8B CohereLabs/tiny-aya-global <<'EOF'
       import collections, sys, torch
       from transformers import AutoConfig, AutoModelForCausalLM
       for ckpt in sys.argv[1:]:
           for label, extra in (("plain", {}), ("with config", {"config": AutoConfig.from_pretrained(ckpt)})):
               model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.float32, **extra)
               print(ckpt, label, dict(collections.Counter(str(p.dtype) for p in model.parameters())))
       EOF

   Expect: Qwen3.5 plain = bfloat16 (the bug), with config = float32; Tiny Aya float32 both ways.

2. **50-step throwaway run (after the code change):**

       bash egs/launch.sh sft egs/mmtc/fr-en/sft_qwen-3.5-0.8b.yaml \
         run_name=dtypetest_fp32_qwen0.8b train.max_steps=50 train.eval_steps=25 wandb.enabled=false

   - Trains in fp32: the log's parameter dump is float32
     (`grep -o "torch\.[a-z]*[0-9]*" <run>/slurm-*.out | sort | uniq -c`; one bfloat16 hit, the
     `compute_dtype` log line, is expected).
   - Saves fp32: `"dtype": "float32"` in `<run>/model/config.json`, and `model/` is about twice the
     size of the existing `sft_full_qwen-3.5-0.8b_short/model`.
   - Sane: eval loss falls, no NaN, `model_dev_check: ok` in `model/training_info.yaml` (this also
     exercises the best-checkpoint reload).

   Delete the run folder afterwards.

3. **Comparison run**, same settings as the existing `sft_full_qwen-3.5-2b_short`:

       bash egs/launch.sh sft egs/mmtc/fr-en/sft_qwen-3.5-2b.yaml \
         run_name=sft_full_qwen-3.5-2b_short_fp32 slurm.constraint=80gb

   Compare dev BLEU-lc per evaluation, best step, test scores, peak memory and runtime with the old
   run, in `data.md`. One run shows fp32 behaves sensibly at full length, not that it is better.

## Risks

- **Memory and disk about double** for Qwen3.5 full runs (16 bytes per parameter for weights,
  gradients and AdamW state instead of 8; fp32 checkpoints). Hence `slurm.constraint=80gb` above; if the
  2B does not fit at batch 32, use `train.train_batch_size=16 train.gradient_accumulation_steps=4`.
  The 4B full run (about 65 GiB estimated) is not measured here.
- **The learning rate (1e-5) was chosen on bf16-weight runs**; fp32 weights take every update, so the
  comparison run may behave differently, e.g. overfit sooner. That is a finding, not a failed fix.
- Full CPT of a Qwen3.5 checkpoint shares `load_base_model` and also becomes fp32.
