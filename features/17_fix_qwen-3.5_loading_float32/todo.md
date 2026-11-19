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
      `HFMT_ROOT=$(pwd) python hfmt/test_grid.py` (37 passed), 2026-11-19. Still to do: `black` (not
      installed in the sandbox) and, on the cluster, `python hfmt/test_mt_model.py` and
      `python hfmt/test_sft_translation.py`.

## C. Verify (cluster)

- [ ] C1. Launch the 50-step throwaway run (`plan.md`, Verification 2).
- [ ] C2. Log's parameter dump is float32; the job got past loading (guard passed).
- [ ] C3. `<run>/model/config.json` has `"dtype": "float32"`; `model/` is about twice the size of the
      existing `sft_full_qwen-3.5-0.8b_short/model`.
- [ ] C4. Eval loss falls, no NaN, `model_dev_check: ok`.
- [ ] C5. Record in `data.md`; delete the run folder.

## D. Comparison run and close-out

- [ ] D1. (cluster) Launch `sft_full_qwen-3.5-2b_short_fp32` (`plan.md`, Verification 3); repeat C2–C3
      on it.
- [ ] D2. Compare with `sft_full_qwen-3.5-2b_short` in `data.md`: dev BLEU-lc per evaluation, best
      step, test scores, peak memory, runtime.
- [ ] D3. `decisions.md` entry (cause, fix, guard, memory cost); README "Training": one sentence that
      `full` is fp32 for every family; mark E4 done in `features/15_running_on_v100/todo.md`.
