# 17_fix_qwen-3.5_loading_float32 — todo

From `plan.md` (v3, 2026-11-19). In order: A, B, C, D. Cluster items are run by the developer; numbers
go in `data.md` (aggregate only).

## A. Confirm the cause

- [x] A1. (cluster, CPU) Run the load check from `plan.md` (Verification 1). Expect Qwen3.5 plain =
      bfloat16, with config = float32; Tiny Aya float32 both ways. If "with config" is still bfloat16,
      B2 uses the fallback (`model.to(model_dtype)`).

## B. Code change (`hfmt/mt_model.py`)

- [x] B1. Add `check_parameter_dtype(model, expected, checkpoint)`: raises `RuntimeError` naming the
      checkpoint, the requested dtype and the dtypes found, if a floating-point parameter differs.
- [x] B2. `load_base_model`: pass `config=AutoConfig.from_pretrained(cfg.checkpoint)` to both
      `from_pretrained` calls; call the guard after the `full`/`lora` load; one docstring sentence on why.
- [x] B3. `load_trained_model`: call the guard after the `full` load.
- [x] B4. `hfmt/test_mt_model.py`: the guard passes on matching dtypes, raises on a mismatch, ignores
      non-floating parameters.
- [ ] B5. Locally: `python hfmt/test_mt_model.py` (5 passed; it needs no torch) and
      `HFMT_ROOT=$(pwd) python hfmt/test_grid.py` (37 passed), 2026-11-19. Cluster: `python hfmt/test_sft_translation.py` 6 passed. Still to do:
      `black` (not installed in the sandbox) and, on the cluster, `python hfmt/test_mt_model.py`.
      (Not blocking: C and D ran the changed code end to end.)

## C. Verify (cluster)

- [x] C1. Launch the 50-step throwaway run (`plan.md`, Verification 2).
- [x] C2. Log's parameter dump is float32; the job got past loading (guard passed).
- [x] C3. `<run>/model/config.json` has `"dtype": "float32"`; `model/` is about twice the size of the
      existing `sft_full_qwen-3.5-0.8b_short/model`.
- [x] C4. Eval loss falls, no NaN, `model_dev_check: ok`.
- [ ] C5. Record in `data.md` (done 2026-11-19); delete the run folder (developer).

## D. Comparison run and close-out

- [x] D1. (cluster) Launch `sft_full_qwen-3.5-2b_short_fp32` (`plan.md`, Verification 3); repeat C2–C3
      on it.
- [x] D2. Compare with `sft_full_qwen-3.5-2b_short` in `data.md`: dev BLEU-lc per evaluation, best
      step, test scores, peak memory, runtime.
- [x] D3. `decisions.md` entry (cause, fix, guard, memory cost); README "Training": one sentence that
      `full` is fp32 for every family; mark E4 done in `features/15_running_on_v100/todo.md`.
- [x] D4. (added) `lora` and `qlora` loads with the explicit config: 25-step throwaway runs
      `dtypetest_lora_qwen2b`, `dtypetest_qlora_qwen2b` finish with `model_dev_check: ok` (`data.md`, 2b).
- [x] D5. (added) The fp32 2B full run needs more than a 40 GB card: `slurm.constraint: 80gb` in
      `egs/mmtc/{fr-en,ur-en,zh-en}/sft_qwen-3.5-2b.yaml`.
- [ ] D6. (developer) Delete the `dtypetest_*` run folders.

## Left open (outside this feature)

- The fp32 comparison run scored about 1 BLEU below the bf16 run at the same learning rate, with a
  rising dev loss (`data.md`, Verification 3). Suspected: 1e-5 is too high for fp32 weights. Needs a
  learning-rate check before Qwen3.5 full fine-tunes are rerun.
- Not measured: fp32 memory of the 4B full run (about 65 GiB estimated), resume from a float32 checkpoint.
