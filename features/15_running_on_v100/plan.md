# 15_running_on_v100 — plan (v3)

Status: 2026-11-13. Decided: V100 jobs run in bf16, like every other card. Measurements behind every
statement here are in `data.md`.

## Decision

In `mt_model.load_base_model`, change

```python
use_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
```

to `including_emulation=True`. A V100 has no native bf16 but PyTorch emulates it, so the check becomes
true there and a V100 job takes the same code path as an A100 or L40S job: bf16 weights for `lora`, bf16
compute for `qlora`, bf16 mixed precision in the Trainer. Nothing changes on A100s and L40Ss.

Selecting a V100 needs no new code: `slurm.gres=gpu:v100:1 slurm.constraint=32gb` on any launch path.

## Why

- **It is the only precision that works for all three model families.** The smoke tests trained and
  decoded Qwen3.5, Gemma 3 and Tiny Aya in bf16 on a V100 with dev scores inside seed noise of the A100
  controls, no NaN, and `model_dev_check: ok`.
- **A V100 run stays comparable with every existing result.** Same number format, same mixed-precision
  setup, same peak memory as on an A100. Only the wall-clock differs.
- **One line, no new concepts.** No per-family rule, no config key, no hardware module. And because
  precision no longer depends on the card, the problems the earlier plan needed guards for go away:
  - resuming on a different card does not change precision;
  - inference of an existing run on a V100 uses the precision it trained at;
  - `model_merged/` is bf16 wherever the merge runs, as documented.

## Tradeoff: it is the slowest option

Emulated bf16 was the slowest V100 setup measured for every family. A V100 job takes several times
as long as the same job on an A100, and each family has a faster alternative that we are giving up:

| Alternative | Where it wins | Why not |
|---|---|---|
| fp16 (what the code does today) | Clearly faster for Tiny Aya | Gemma 3 produces empty output. Qwen3.5 can't use it: the pinned Transformers loads its weights as bf16 whatever is requested. Different numerics from all existing results. |
| fp32 | Faster for Gemma 3 | Doubles weight memory, so the larger models don't fit in 32 GB. Much slower than fp16 for Tiny Aya. Qwen3.5 ignores it too. |
| Per-family rule (fastest working dtype for each) | Best speed everywhere | Needs a validated-families list and a smoke test per new family; V100 runs would differ numerically from A100 runs, which then needs recording, resume and merge guards, and reporting. |

We accept the slowdown: V100s are extra capacity for when A100/L40S queues are the bottleneck, and a
slow run that is directly comparable is worth more than a faster one that needs an asterisk.

Caveat on the speed comparison: `gpu:v100` covers two card models, and the bf16 runs all landed on one
while the faster alternatives landed on the other. Some of the gap may be the card, not the precision.
This doesn't change the decision (bf16 is chosen for correctness and simplicity), only how large the
cost really is.

## Practical consequences

- **Time.** Raise `slurm.time` for V100 runs; the 4 h default will not cover a `shortrun` of a 4B-class
  model. Requeue/resume covers a time-out. Evaluation and dev-decode schedules stay the same as on
  other cards so checkpoint selection is comparable.
- **Memory.** Peak memory matches the A100, so the only question is the 32 GB limit. Keep the effective
  batch at 64 (`train.train_batch_size=b train.gradient_accumulation_steps=64/b`); LoRA settings and
  `decode.batch_size` don't change. The largest models (9B, 12B) are tight; prefer an A100 for them.
- **Full fine-tunes** stay on A100s except for the smallest model (fp32 weights plus optimizer state).
- **Run names.** A V100 replica of an existing experiment needs its own `run_name` (e.g. `<name>_v100`),
  or it overwrites the A100 run folder.

## Steps

1. Make the one-line change; update the `load_base_model` docstring (no fp16 branch is reachable on our
   cards any more).
2. Done: the bf16 smoke runs set exactly what this change sets, in a real job's environment. Optional:
   one A100 inference of an existing run (expect 0 differing lines) to confirm dropping the cuDNN module
   changed no results.
3. Full-length replication on a V100 of a few existing experiments; accept within 0.5 BLEU-lc of the
   A100 run. Archive under `analysis/<date>_v100_validation/`, record the decision in `decisions.md`.
4. README "Running SFT experiments": a short "Which GPU" paragraph (how to select a V100, expect it to
   be several times slower, raise `slurm.time`, batch × accumulation rule).
5. Delete the `smoke_*` run folders and the throwaway `force_dtype_*` scripts.

## Not tested yet

- `qlora` on a V100 (decided: it should run there; a smoke test against an A100 control is in `todo.md`).
  `full` fine-tunes stay on A100s, Qwen3.5-0.8B included (decided), so they are not tested on V100s.
- Anything beyond 50 steps, one seed, one language pair (step 3).

## Open questions

- Should a run record which card it used (W&B config, `training_info.yaml`)? No longer needed for
  correctness, since precision is the same everywhere; useful only to explain run times.
- Which runs make up the replication set in step 3.
- Found along the way, outside this feature: Qwen3.5 `full` fine-tunes have been training bf16 weights,
  not fp32 (the Transformers dtype issue above). Needs its own decision.
