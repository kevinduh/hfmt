# 15_running_on_v100 — todo

From `plan.md` (v3, 2026-11-13). Groups are ordered by dependency: A before B, B before C; D can follow
B; E is last. Cluster items are run by the developer (no GPU in the sandbox); results go in `data.md`
as aggregate numbers only, nothing to W&B for smoke tests.

## A. Code change

- [x] A1. `hfmt/mt_model.py` `load_base_model`: `torch.cuda.is_bf16_supported(including_emulation=False)`
      → `including_emulation=True`.
- [x] A2. Update the `load_base_model` docstring ("bf16/fp16 for `lora`" → bf16; say the fp16 branch
      remains only for a card with no bf16 at all, emulated or native).
- [x] A3. Add a one-line comment at the changed line pointing to `decisions.md` (why emulation is
      accepted: same numerics on every card, at a speed cost on V100s).
- [x] A4. Grep for other statements of the old behaviour and fix them: `fp16` / `is_bf16_supported` in
      `hfmt/sft_translation.py` comments, `hfmt/merge_model.py` docstring, `README.md`, `CLAUDE.md`,
      `config_reference.md`, `inference_overview.md`.
- [x] A5. Check `hfmt/test_mt_model.py` and `hfmt/test_sft_translation.py` for anything that mocks or
      asserts the `including_emulation=False` call; update if so.
- [x] A6. Run the torch-free tests locally: `HFMT_ROOT=$(pwd) python hfmt/test_grid.py` and
      `python tools/test_run_status.py`.
- [ ] A7. (cluster) Run `python hfmt/test_mt_model.py` and `python hfmt/test_sft_translation.py`.

## B. Verify the change through the normal launch path (cluster)

- [ ] B1. Confirm the environment fix is live: a job submitted from a fresh shell no longer has the
      cluster cuDNN module on its path (`data.md`, T2a: a later check still failed).
- [ ] B2. `DRYRUN=1` then launch one V100 LoRA smoke run with no forced dtype:
      ```
      bash egs/launch.sh sft egs/mmtc/fr-en/sft_lora_gemma-3-4b-it_all-linear.yaml \
        run_name=smoke_v100_default_gemma4b slurm.gres=gpu:v100:1 slurm.constraint=32gb \
        slurm.time="04:00:00" train.max_steps=50 train.eval_steps=25 train.train_batch_size=4 \
        train.gradient_accumulation_steps=16 train.eval_batch_size=4 wandb.enabled=false
      ```
      Read: log line `use_bf16: True`, "Accelerate precision: bf16", base parameters bf16, no NaN
      `eval_loss`, `model_dev_check: ok`, dev BLEU-lc and TER against `smoke_v100_bf16_gemma4b` (the
      forced-dtype run; expect the same within noise), peak memory equal to the A100 control.
- [ ] B3. A100 regression: `bash egs/launch.sh inf <existing Qwen3.5 run>/run_config.yaml
      inf_name=test_after_v100_change`; expect 0 differing lines against the run's own `test1.pred.trg`
      and identical `scores.json`. (Also closes the pending check that removing the cuDNN module left
      results unchanged.)
- [ ] B4. V100 inference of an existing A100-trained run:
      `bash egs/launch.sh inf <run folder>/run_config.yaml slurm.gres=gpu:v100:1 slurm.constraint=32gb
      inf_name=test_v100_bf16`; count differing lines and compare scores with the run's own.
- [ ] B5. `qlora` on a V100 (untested so far): 50-step smoke run of an existing qlora experiment with the
      B2 overrides; same reads as B2, plus the log's `compute_dtype`.
- [ ] B6. `full` on a V100 (untested so far): 50-step smoke run of `sft_qwen-3.5-0.8b.yaml` with the B2
      overrides; same reads as B2.
- [ ] B7. Merge on a V100: `bash egs/launch.sh merge <B2 run folder>/run_config.yaml
      slurm.gres=gpu:v100:1 slurm.constraint=32gb`; check `model_merged/config.json` says bfloat16.
- [ ] B8. Record B2–B7 in `data.md` (one table, "shown / not shown" lines as in the existing sections).

## C. Full-length replication (cluster; after B passes)

- [ ] C1. Decide the replication set with the developer (open question in `plan.md`): which models and
      language pairs, and the `slurm.time` for each, sized from the smoke runs' step and decode times.
- [ ] C2. Launch each as the unchanged experiment yaml with `run_name=<name>_v100
      slurm.gres=gpu:v100:1 slurm.constraint=32gb slurm.time=<from C1>` (batch × accumulation adjusted
      only where 32 GB requires it, effective batch 64).
- [ ] C3. When finished: `python tools/run_status.py <outputs dir>`; check no failed checks, no
      known-bad warnings, `model_dev_check: ok`.
- [ ] C4. Compare test BLEU-lc with the A100 run of the same experiment: accept within 0.5 and inside
      the spread of the `_seed1` pairs. Note wall-clock against the A100 run.
- [ ] C5. Archive under `analysis/<date>_v100_validation/`: source export/score files and `summary.md`
      (runs, results table, wall-clock ratios, findings). Aggregate metrics and config only.
- [ ] C6. Optional, to size the cost: Tiny Aya 50-step smoke run in bf16 on the same V100 card model as
      its fp16 run (pin the node or check `nvidia-smi` in the log); add to `data.md`.

## D. Documentation

- [ ] D1. README "Running SFT experiments": short "Which GPU" paragraph (select with
      `slurm.gres=gpu:v100:1 slurm.constraint=32gb`; same precision as other cards; several times
      slower, so raise `slurm.time`; keep effective batch 64 via batch × accumulation; 9B/12B and full
      fine-tunes belong on A100s; `_v100` run names for replicas).
- [ ] D2. `decisions.md`: record the decision (bf16 everywhere via emulation; alternatives fp16, fp32,
      per-family rule and why not; the speed cost), and after C4 the replication outcome.
- [ ] D3. `plan.md`: update the status line and "Not tested yet" as B and C complete.

## E. Clean-up and follow-ups

- [ ] E1. Delete all `smoke_*` and `*test_*` run folders from this feature (they are not results), and
      the `inf/test_v100*`, `inf/test_after_v100_change` folders if not wanted.
- [ ] E2. Delete `force_dtype_inference.{py,sh}`, `force_dtype_training.{py,sh}` and
      `report_smoke_runs.sh` from the feature folder (confirm with the developer first).
- [ ] E3. Resolve with the developer: should a run record its card (W&B config,
      `model/training_info.yaml`)? If yes, open it as a small follow-up; not part of this feature.
- [ ] E4. Raise outside this feature: Qwen3.5 `full` fine-tunes train bf16 weights, not fp32 (pinned
      Transformers ignores the requested dtype). Needs its own request/decision; the yamls' memory and
      disk notes for those runs are overstated.
