# 15_running_on_v100 — todo

From `plan.md` (v3, 2026-11-13). Groups are ordered by dependency: A before B, B before C; D can follow
B; E is last. Cluster items are run by the developer (no GPU in the sandbox); results go in `data.md`
as aggregate numbers only, nothing to W&B for smoke tests.
All commands run from the hfmt root on the cluster.

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

- [x] B1. Environment fix is live. The cluster's `install/path.sh` had no cuDNN module before the smoke
      runs, so they ran in a real job's environment. Assumed: the submitting shell loads no cuDNN module
      (no `module unload` in `path.sh`).
- [x] B2. V100 LoRA training in bf16. Covered by the `smoke_v100_bf16_*` runs (`data.md`):
      `force_dtype_training.py bfloat16` sets exactly what the new code sets (bf16 load, Trainer
      `bf16=True`), and `is_bf16_supported(including_emulation=True)` is True on the V100 (T0). All three
      families within noise of the A100, no NaN, `model_dev_check: ok`, same peak memory.
- [ ] B3. Optional. A100 regression: decode an existing Qwen3.5 run again and expect 0 differing lines
      and identical scores. Checks that dropping the cuDNN module changed no results. Low risk: with the
      module loaded any cuDNN call failed (T2a #3), so earlier A100 runs never called cuDNN.
      ```
      RUN=egs/mmtc/ur-en/outputs/sft_lora_qwen-3.5-4b_ur-en_all-linear_sweep-0   # or outputs/complete/... if moved
      bash egs/launch.sh inf $RUN/run_config.yaml inf_name=test_after_v100_change
      # when the job has finished:
      diff $RUN/test1.pred.trg $RUN/inf/test_after_v100_change/test1.pred.trg | grep -c '^<'   # expect 0
      diff $RUN/scores.json $RUN/inf/test_after_v100_change/scores.json && echo "scores identical"
      ```
- [x] B4. V100 bf16 inference of an A100-trained run. Covered by T1b (Gemma 3, forced bf16) and T1
      (Qwen3.5, whose weights load as bf16 whatever is requested). Differing-line counts not recorded.
- [ ] B5. `qlora` on a V100 (decided: qlora should run on V100s). Untested: the force scripts skip qlora,
      and bitsandbytes 4-bit with bf16 compute on Volta is the question. 50-step smoke test of Qwen3.5-2B
      qlora (sweep axes pinned to one job) on a V100, with an A100 control at the same settings:
      ```
      for card in v100 a100; do
        constraint=""; [ $card = v100 ] && constraint="slurm.constraint=32gb"
        DRYRUN=1 bash egs/launch.sh sft egs/mmtc/fr-en/sft_qlora_qwen-3.5-2b.sweep.yaml \
          run_name=smoke_${card}_qlora_qwen2b slurm.gres=gpu:${card}:1 $constraint \
          train.learning_rate=1e-4 peft.lora.r=8 \
          train.max_steps=50 train.eval_steps=25 train.train_batch_size=4 \
          train.gradient_accumulation_steps=16 train.eval_batch_size=4 wandb.enabled=false
      done
      # check both compose as one job each, then rerun the loop without DRYRUN=1. When both have finished:
      bash features/15_running_on_v100/report_smoke_runs.sh egs/mmtc/fr-en/outputs 'smoke_*_qlora_qwen2b*'
      grep -h "compute_dtype" egs/mmtc/fr-en/outputs/smoke_v100_qlora_qwen2b*/*.log   # expect torch.bfloat16
      ```
      Pass: the V100 run starts (no bitsandbytes or CUDA kernel error), no NaN `eval_loss`,
      `model_dev_check: ok`, "Accelerate precision: bf16", dev BLEU-lc within 0.5 of the A100 control at
      steps 25 and 50. If it fails, document qlora as A100/L40S only (D1).
- [x] B6. Dropped (decided): `full` fine-tunes, Qwen3.5-0.8B included, stay on A100s; D1 says so.
- [ ] B8. Record B3 (if run) and B5 in `data.md`.

## C. Full-length replication (cluster; after B passes)

- [ ] C1. Decide the replication set with the developer (open question in `plan.md`; suggested: one
      model on one pair, e.g. Tiny Aya, the slowest in bf16, which also sizes `slurm.time` for the worst
      case), and the `slurm.time` for each, sized from the smoke runs' step and decode times.
      **Waiting on the developer.**
- [ ] C2. Launch each as the unchanged experiment yaml with `run_name=<name>_v100
      slurm.gres=gpu:v100:1 slurm.constraint=32gb slurm.time=<from C1>` (batch × accumulation adjusted
      only where 32 GB requires it, effective batch 64). For the suggested set, Tiny Aya fr-en at lr 1e-4
      (A100 counterpart: `sft_lora_tiny-aya-global_all-linear_sweep-0`):
      ```
      bash egs/launch.sh sft egs/mmtc/fr-en/sft_lora_tiny-aya-global_all-linear.sweep.yaml \
        run_name=sft_lora_tiny-aya-global_all-linear_sweep_v100 train.learning_rate=1e-4 \
        slurm.gres=gpu:v100:1 slurm.constraint=32gb slurm.time="24:00:00"
      ```
      `slurm.time` estimate from the bf16 smoke run: 400 steps at ~105 s plus 8 dev decodes at ~25 min
      ≈ 16 h. Check the partition allows 24 h. A time-out is not requeued: resume it with
      `bash egs/launch.sh sft <run folder>/run_config.yaml slurm.gres=gpu:v100:1 slurm.constraint=32gb`.
- [ ] C3. When finished: `python tools/run_status.py <outputs dir>`; check no failed checks, no
      known-bad warnings, `model_dev_check: ok`.
- [ ] C4. Compare test BLEU-lc with the A100 run of the same experiment: accept within 0.5 and inside
      the spread of the `_seed1` pairs. Note wall-clock against the A100 run. For the Tiny Aya pair (if a
      folder was moved, point at `outputs/complete/...`):
      ```
      OUT=egs/mmtc/fr-en/outputs
      A100=$OUT/sft_lora_tiny-aya-global_all-linear_sweep-0; V100=$OUT/sft_lora_tiny-aya-global_all-linear_sweep_v100
      python - $A100 $V100 <<'EOF'
      import json, sys
      a, v = (json.load(open(f"{run}/scores.json")) for run in sys.argv[1:])
      for test in a:
          for metric in ("bleu", "bleu_13a_lowercase", "chrf", "ter"):
              print(test, metric, a[test][metric], v[test][metric], round(v[test][metric] - a[test][metric], 2))
      for run in sys.argv[1:]:
          state = json.load(open(f"{run}/trainer_state.json"))
          print(run, "train_runtime", [e["train_runtime"] for e in state["log_history"] if "train_runtime" in e])
      EOF
      diff $A100/test1.pred.trg $V100/test1.pred.trg | grep -c '^<'   # differing lines, for data.md
      ```
- [ ] C5. Archive under `analysis/<date>_v100_validation/`: source export/score files and `summary.md`
      (runs, results table, wall-clock ratios, findings). Aggregate metrics and config only.

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
      the `inf/test_v100*`, `inf/v100test_*`, `inf/test_after_v100_change` folders if not wanted.
- [ ] E2. Delete `force_dtype_inference.{py,sh}`, `force_dtype_training.{py,sh}` and
      `report_smoke_runs.sh` from the feature folder (confirm with the developer first).
- [ ] E3. Resolve with the developer: should a run record its card (W&B config,
      `model/training_info.yaml`)? If yes, open it as a small follow-up; not part of this feature.
- [ ] E4. Raise outside this feature: Qwen3.5 `full` fine-tunes train bf16 weights, not fp32 (pinned
      Transformers ignores the requested dtype). Needs its own request/decision; the yamls' memory and
      disk notes for those runs are overstated.
