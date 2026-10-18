# 10_inference_more_options — plan (v3, final)

Status: **v3, final.** Q1–Q12 are decided (developer answers, 2026-10-18).
`todo.md` is generated from this version.

## Request (summary)

1. **Base-model inference.** `launch.sh inf <experiment>.yaml` decodes the experiment's vanilla
   `checkpoint`, with no adapter and no training. `launch.sh inf <run folder>/run_config.yaml`
   keeps decoding the trained model.
2. **Several test sets.** `data.test` may list several files. Each test src file is decoded
   on its own, and scored against its trg when one is given. train/dev stay concatenated.
3. **Merging.** A script merges a run's `model/` into the base and writes `model_merged/` in the
   same run folder. An experiment can use `model_merged/` as its `checkpoint` for a second round of
   SFT. Inference on round-1 and round-2 run folders works the same way.
4. **Readability.** Comment `grid.py` and `launch.sh`, which are getting hard to follow.
5. **Cheaper dev decoding.** `train.inference_callback_interval` (int, default 1): the MT callback's
   dev decoding + BLEU runs every N evaluations (every N × `train.eval_steps` steps). Early
   stopping (`super().on_evaluate`) still runs at every evaluation.

## Starting point (`hydra3` @ 27b8cc3)

- `grid.py --inf` accepts only `run_config.yaml`. It sets `model_dir=<run>/model` and saves
  `<run>/inf/<inf_name>/inf_config.yaml`. `inf_translation.py` fails if `model_dir` is unset.
- `launch.sh` sources the experiment folder's `local_setting.sh` for `sft` only. Each kind has a
  job script: `egs/sft_job.sh`, `egs/inf_job.sh`.
- `data.test` is an `Optional[Bitext]` (`src`/`trg` lists). Both scripts decode it as **one
  concatenated set**:
  - training writes `eval.pred.trg` and doesn't score it;
  - inference writes `pred.trg` and `scores.json`.
- Training saves only the adapter (or full weights) in `model/`, with no tokenizer.
  `checkpoint` goes straight to `from_pretrained`, so a local folder already works as a checkpoint.
- `load_base_model` picks the precision from `peft.mode`: full → fp32, lora → bf16, qlora → 4-bit.
- The MT callback (`sft_translation.make_mt_callback`) decodes the whole dev set at every evaluation.
  That took about 80% of `aya_tuk_lora`'s wall time.
- **`conf/train/longrun.yaml` uses `metric_for_best_model: eval_bleu`.** The Trainer raises if that
  metric is missing from an evaluation (`trainer.py:3127`), which affects request 5.
- Inference reproduces training's test output exactly (feature 09). That must stay true.

## Proposal

### 1. Base-model inference (request 1) — Q1, Q2 decided

The kind of yaml decides what gets decoded (Q1). There's no new mode or flag.

- **`grid.py --inf`.** Branch on the input:
  - **`run_config.yaml`:** as today; `model_dir=<run>/model`, output in `<run>/inf/<inf_name>/`.
  - **Any other yaml (an experiment):**
    - compose and check it like `sft` does;
    - leave `model_dir` unset;
    - write output to `<output_root>/<run_name>/base_ckpt_inf/<inf_name>/` (Q2), with
      `output_root` defaulting to `<experiment folder>/outputs`, the same as `sft`.

    The separate `base_ckpt_inf/` subfolder never collides with a trained run's `inf/`, so nothing
    is refused for that reason.
  - **Refused:**
    - a sweep experiment (`hydra.sweeper.params` set), since one run means one model;
    - comma lists.
- **`launch.sh`.** Source `local_setting.sh` for `inf` as well as `sft`. An experiment needs
  `HFMT_DATA_ROOT`. A run folder has no `local_setting.sh`, so trained runs are unaffected.
- **`inf_translation.py`.** `model_dir` unset → `load_base_model(cfg)` + `eval()`. Set → `load_trained_model`
  as today.
- **Precision of the base model.** It comes from the experiment's `peft` option, through `load_base_model`:
  - `lora_default` = bf16 (the natural zero-shot setting);
  - `qlora_default` = 4-bit, without `prepare_model_for_kbit_training`, which exists only to match a trained QLoRA run;
  - `full_finetune` = fp32.

  So `bash egs/launch.sh inf egs/mmtc/fr-en/sft_lora.yaml peft=lora_default`.

### 2. Several test sets (request 2) — Q3, Q4, Q5, Q9 decided

- **One shared helper**, `mt_model.decode_test_sets(cfg, tokenizer, model, outdir, device)`.
  - **Callers:** training's final test step and `inf_translation`, so inference still reproduces training.
  - **Per test file *i*:**
    - loads `test.src[i]` alone;
    - writes `<outdir>/test<i>.pred.trg`, numbered from 1 by position in `data.test.src`
      (Q3, Q9: also when there's only one test file);
    - scores it if it has a reference.
- **`scores.json`**, one per output folder:
  ```json
  {"test1": {"src": "<path>", "trg": "<path>", "bleu": ..., "chrf": ..., "ter": ...},
   "test2": {"src": "<path>", "trg": null}}
  ```
  The paths identify each test set. They are local files only (paths, not content).
- **References (Q4: `null` allowed).** In the schema, `data.test` becomes a `TestSet` with
  `src: List[str]` and `trg: Optional[List[Optional[str]]] = None`.
  - `trg` omitted → no test set is scored.
  - Otherwise `trg` must have one entry per src file. A `null` entry means that test set has no reference.
  - `check_config` checks the lengths.
  - The cost is the `Optional` element type plus one `if` in the helper.
- **Training also scores its test sets now.** It writes the same `scores.json` and logs the
  aggregate numbers to the W&B run summary as `test/test<i>/{bleu,chrf,ter}` (Q5). No paths and no
  text go to W&B (CLAUDE.md).
- **File names change (Q9):**
  - training's `eval.pred.trg` → `test<i>.pred.trg` in the run folder;
  - inference's `pred.trg` → `test<i>.pred.trg` in `inf/<inf_name>/` or `base_ckpt_inf/<inf_name>/`.

  Old run folders keep their old files.
- **No change:** train and dev are still concatenated. `'data.test.src=[a,b]' 'data.test.trg=[a_ref,null]'`
  works on the command line. The numbering follows the list given there.

### 3. Merging and a second round of SFT (request 3) — Q6, Q7, Q8 decided

- **Goes through Slurm (Q6), as a third launcher kind:** `bash egs/launch.sh merge <run folder>/run_config.yaml`.
  - **`grid.py --merge`** checks the saved config. It refuses comma lists, a missing `model/` and
    `peft.mode=full` (whose `model/` already holds full weights). Like `--inf`, it prints `JOBS`
    = the run folder and the sbatch arguments from the run's `slurm:` block. It writes no new
    config, because the merge reads `run_config.yaml` directly.
  - **The job** runs `python -m hfmt.merge_model <run folder>` through the shared job script (Q10, below).
  - **Logs:** `<run>/merge-slurm-%A.out`. **Job name:** `<run_name>-merge`.
  - **`LOCAL=1` and `DRYRUN=1`** work as for the other kinds.
  - **An existing `model_merged/`** gets a warning and is overwritten, following the project
    convention that an existing folder only warns.
- **`hfmt/merge_model.py <run folder>`** reads `<run>/run_config.yaml` (`OmegaConf.load`) and calls
  `mt_model.merge_trained_model(cfg, model_dir, out_dir)`.
  - **Writes** `<run>/model_merged/`, holding the full bf16 weights and the tokenizer.
  - **Also writes** `model_merged/merge_info.yaml`, recording the source run folder,
    `checkpoint`, `peft.mode` and the run's `git_rev`.
- **QLoRA merge base (Q7):** load the base in 4-bit, `dequantize()` it to bf16, then merge.
  - **Why:** the adapter was trained against these 4-bit-rounded weights, so the merged model is
    the model that was evaluated (up to bf16 rounding).
  - **A warning is printed** (to the log and to `merge_info.yaml`). It says the merged weights carry
    4-bit rounding of the original base, so they are slightly lower fidelity than the original
    bf16 checkpoint.
  - **LoRA runs** merge into the bf16 base.
- **Training saves the tokenizer into `model/`.** That makes a full fine-tune's `model/` usable as a
  `checkpoint` directly.
- **Second round:** an experiment sets `checkpoint: <abs path>/outputs/<run>/model_merged`.
  - Nothing else changes: `load_tokenizer`/`load_base_model` read the folder like a Hub name.
  - Round-2 inference is the usual `launch.sh inf <round-2 run>/run_config.yaml`: its `model/`
    (round-2 adapter) on its `checkpoint` (the merged model).
  - To decode the merged model alone, use request 1 with an experiment whose `checkpoint` is
    `model_merged`. Output lands in `base_ckpt_inf/`.
- **No checkpoint-path check in `grid.py` (Q8).** A mistyped local path fails when the job starts.
- **Documented risk.** Round 2 depends on round 1's run folder. Re-running round 1's `run_name`
  reuses that folder (it only warns), and its new merge would change round 2's base. The docs say to
  copy `model_merged/` out of `outputs/` for anything long-lived.

### 4. Comments in `grid.py` and `launch.sh` (request 4)

- **`grid.py`:**
  - the module docstring opens with the four entry flows (sft grid, inf on a run, inf on an
    experiment, merge) and what each writes;
  - each `main*` function gets numbered step comments;
  - each helper gets a one-line comment on what it's for and who calls it;
  - a comment on the stdout contract shows an example of the printed `JOBS=(…)`/`SBATCH_ARGS=(…)` bash.
- **One job script (Q10).** `egs/sft_job.sh` and `egs/inf_job.sh` fold into one
  `egs/job.sh <sft|inf|merge> <folders…>`.
  - **What it does:** picks job folder number `$SLURM_ARRAY_TASK_ID` and runs the module for the kind:
    - `sft` runs `hfmt.sft_translation` on `run_config`;
    - `inf` runs `hfmt.inf_translation` on `inf_config`;
    - `merge` runs `hfmt.merge_model <folder>`.
  - **Callers:** `launch.sh` runs `egs/job.sh "$kind"`, both under sbatch and with `LOCAL=1`.
  - **Also updated:** `tools/run_on_free_gpu.sh` and the docs.
- **`launch.sh`:** one comment per block (argument parsing, `local_setting.sh`, `grid.py` eval,
  submit/LOCAL/DRYRUN), and the usage header lists the three kinds.
- Comments are added in the phase that touches the code, and a final pass reads both files top to bottom.

### 5. Dev-decoding interval (request 5)

- **Schema:** `TrainConfig.inference_callback_interval: int = MISSING`.
- **Default:** `1` in `conf/train/default.yaml` and `conf/train/longrun.yaml`. Every train option
  must set it, under the Rule.
- **When dev is decoded (Q11).** With `N = inference_callback_interval`:
  - **N ≥ 1:** at evaluations where `state.global_step % (N * args.eval_steps) == 0`, and also at the
    last step (`global_step == max_steps`), so every run ends with a dev BLEU.
  - **N = 0:** never (loss-only evaluations).
- **Callback order at every evaluation (Q12):**
  1. **On a decode step:** decode dev, score it, add `eval_bleu/chrf/ter` to `metrics` and `wandb.log`
     them, as today. Otherwise log one line, e.g. "dev decoding skipped at step 10 (interval 2)".
  2. **Always:** `super().on_evaluate(...)` (early stopping).
     - **Best metric `eval_loss`:** works normally.
     - **Best metric a decode metric** (`eval_bleu`/`eval_chrf`/`eval_ter`): on a skipped step HF
       finds no value, warns, and leaves the patience counter alone. Patience therefore counts only
       evaluations that have a score.
  3. **Then, on a skipped step with a decode metric as best:** put a never-best placeholder in
     `metrics` (−inf if `greater_is_better`, else +inf).
     - **Why:** the Trainer's best-checkpoint check runs after the callback (`trainer.py:2083-2085`)
       and raises `KeyError` on a missing metric (`trainer.py:3127`).
     - **Effect:** with the placeholder, that step is never chosen as best.
     - **Not logged:** the Trainer logs eval metrics before callbacks run, so it never reaches W&B or the log history.
- **No guard in `check_config`.** `longrun.yaml` (best = `eval_bleu`) works with any N.
  HF's "did not find eval_bleu so early stopping is disabled" warning on skipped steps is accepted;
  our own "skipped" line puts it in context.

## Risks

1. **Reproduction.** Training and inference must call the same `decode_test_sets`. Phase B re-checks
   0 differing lines on a short lora run and a short qlora run.
2. **QLoRA merge fidelity.** `dequantize()` on bnb 4-bit with this Transformers (5.8) is unverified.
   Phase D compares BLEU of the merged model (decoded as a base model) with the run's own `inf/test`
   scores. Expect very close for both modes, not byte-identical.
3. **Merge memory.** The merge job loads the base in bf16 on the job's GPU (about 3 GB for 1.5B,
   about 15 GB for 7B), and saving needs about the same in CPU RAM. `slurm.mem_gb` (25) covers up
   to about 7B.
4. **Best metric vs interval.** The placeholder relies on two orderings inside the Trainer:
   - the eval metrics are logged before callbacks run;
   - the best-metric check runs after them.

   Phase A verifies this on a run with `metric_for_best_model=eval_bleu`, N=2. Expect no KeyError,
   the best checkpoint at a decode step, and no ±inf in W&B or in `trainer_state.json`'s log history.
5. **Run time.** More test sets mean a longer final decode in training. That's expected, since it's what was asked for.
6. **Shifting numbers.** `test<i>` numbers follow list order. Reordering `data.test.src` renumbers
   the outputs, but `scores.json` records each set's paths.
7. **Stale names.** `inf_translation.py`, `sft_translation.py`, `egs/launch.sh`, README and CLAUDE.md
   mention `pred.trg`/`eval.pred.trg`/"run_config only". All are updated in the same phases.
   `hfmt/train_seq2seq.py` (not Hydra) is left as is.

## Phases

- **A — dev-decoding interval (5):** schema + both train options, callback (decode schedule,
  placeholder), tests. GPU checks: 45 steps with `eval_steps=10 inference_callback_interval=2`.
  - **With `eval_loss` as best:** dev `.pred` files and `eval/bleu` at 20, 40 and the last step (45);
    `eval/loss` at every evaluation.
  - **The same with `metric_for_best_model=eval_bleu`:** no KeyError, and the best checkpoint is a decode step.
- **B — several test sets (2):** the `TestSet` schema + `check_config`, `decode_test_sets`, training
  (scores.json + W&B summary) and inference switched over, tests. GPU check: 20-step lora + qlora runs
  with two test sets (test + valid, valid with `trg: null`); inference reproduces both `test<i>.pred.trg` exactly.
- **C — base-model inference (1):** `grid.py --inf` on an experiment, `launch.sh` sourcing, the
  `inf_translation` branch, the sweep refusal, comments, tests. GPU check: zero-shot Qwen fr-en (bf16).
- **D — merging (3):** tokenizer saved in `model/`, `merge_trained_model` + `merge_model.py`,
  `merge_info.yaml`, `grid.py --merge`, the `launch.sh merge` kind, `egs/job.sh` (fold of the job scripts),
  comments, tests. GPU check:
  - merge a lora run and a qlora run through Slurm;
  - decode each merged model through C and compare with the run's `inf/test`;
  - run a 20-step round 2 from one merged model, then `launch.sh inf` on it.
- **E — readability and docs (4):** a final comment pass on `grid.py`/`launch.sh`; CLAUDE.md
  (Inference bullet, structure), README (Inference, new "Merging and a second round" section),
  and a `decisions.md` entry.
- **F — developer validation:** a real multi-test-set run with interval > 1, a zero-shot baseline,
  and one merge plus round 2.

## Decided

- **Q1** Base inference is chosen by the yaml kind (experiment vs `run_config.yaml`).
- **Q2** Base inference output always goes to `<output_root>/<run_name>/base_ckpt_inf/<inf_name>/`,
  with no refusal.
- **Q3** Each test set's output is `test<i>.pred.trg`, numbered from 1 in `data.test.src` order.
- **Q4** `null` entries in `data.test.trg` are allowed (small cost); omitting `trg` means no references.
- **Q5** Training writes `scores.json` and logs aggregate test scores to the W&B summary.
- **Q6** Merging goes through Slurm (`launch.sh merge`).
- **Q7** QLoRA merges into a dequantized 4-bit base, with a printed warning.
- **Q8** No checkpoint-path check in `grid.py`.
- **Q9** The `test<i>.pred.trg` naming applies even with a single test file (`test1.pred.trg`).
- **Q10** The three job scripts fold into one `egs/job.sh <sft|inf|merge>`.
- **Q11** Dev is also decoded at the last step; `inference_callback_interval: 0` means never.
- **Q12** BLEU-type best metrics work with N > 1. On skipped steps, early stopping runs first and
  finds no value; then a never-best placeholder satisfies the Trainer's best-checkpoint check.
