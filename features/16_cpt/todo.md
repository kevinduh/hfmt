# 16_cpt — todo

Generated from `plan.md` (v10, final). Groups follow the plan's sections and are ordered by dependency:
A (refactor) and B (config) come before the CPT script (D); C (launch) needs B; E and F need B–D; G can
run any time. Commit once per phase, subject `16_cpt-<phase>: <summary>`; Claude presents the message and
the developer commits. Each phase ends with its "Done when" checks.

Ground rules for every phase:
- **No dataset content to W&B** (CLAUDE.md). CPT logs loss, perplexity, counts and the data-free config
  only. Quick tests use made-up sentences and `wandb.enabled=false`.
- **Don't touch the untracked leftovers** (`egs/synth_lrl/`, `analysis/2026-10-*`, `tools/bitext_filter*`,
  `conf/peft/lora_all_linear.yaml`, …) **or the developer's runs.** Stage files by name.
- **Tests without torch** (login node / sandbox): `HFMT_ROOT=$(pwd) python hfmt/test_grid.py` and
  `python tools/test_run_status.py`. With torch (cluster env): `python hfmt/test_sft_translation.py`,
  `python hfmt/test_mt_model.py`, `python hfmt/test_cpt_monotext_bitext.py`.
- **Suspected until shown** (CLAUDE.md): R1 (Qwen3.5 packing leak) and R2 (EOS token) are checked by runs,
  not assumed.
- **Throwaway GPU runs** go to `run_name=p16<phase>test_<model>` and are deleted once confirmed.

## A. Refactor: shared training code in `hfmt/train_common.py` (plan §3, R5)

A pure move: no logic changes in this phase, so SFT behaves exactly as before.

### A1. Reference before the move
- [x] On the cluster, before any edit: a short SFT reference run (`egs/mmtc/fr-en/sft_lora.yaml
      run_name=p16atest_ref train.max_steps=40 train.eval_steps=20 wandb.enabled=false`). Keep its log,
      `trainer_state.json` and `scores.json` for A3.

### A2. Move
- [x] New `hfmt/train_common.py` with the module docstring and lazy heavy imports, as in `sft_translation.py`.
- [x] Move the module-level helpers: `config_token_ids`, `wandb_privacy_settings`, `training_info`.
- [x] Extract from `sft_translation.main()` into functions (each a cut-and-paste of the existing block):
  - [x] `init_wandb(cfg, outdir, resume)`;
  - [x] `build_model(cfg, device, tokenizer) -> (model, use_bf16)` (load, PEFT/LoRA setup, the parameter and
        pad/bos/eos logging);
  - [x] `trainer_args(cfg, outdir, use_bf16, **extra) -> SFTConfig` (`extra` overrides/adds SFTConfig
        keyword arguments);
  - [x] `inspect_first_batch(trainer, tokenizer, model)` (also the precision and parameter-count logs around it);
  - [x] `train(trainer, model, cfg, outdir, resume)` (resume + full-FT reload via `load_checkpoint_weights`,
        before and after `trainer.train`), returning `trainer_stats`;
  - [x] `save_model(trainer, model, tokenizer, outdir) -> (model, info)` (GPU memory log,
        unwrap, `model/`, tokenizer, `trainer_state.json`, `training_info`).
- [x] `sft_translation.py` keeps: `DECODE_METRICS`, `DEV_SCORE_TOLERANCE`, `add_dev_scores_to_log_history`,
      `check_model_dev_scores`, `make_mt_callback`, data prep, the dev check, test decoding. It imports
      the moved helpers, so `from hfmt.sft_translation import ... training_info` in
      `test_sft_translation.py` still works.
- [x] `sft_translation.main()` = config/data prep + calls to the shared functions + dev check + test
      decoding, in the same order as before (same log lines).

### Done when
Result (2026-11-18), A100 80GB, Qwen3.5-2B LoRA, 40 steps, same config and seed:
- SFTConfig dumps identical (131 keys) apart from `run_name`/`output_dir`; the log's sequence of records
  is identical apart from Hugging Face hub HTTP lines (the reference downloaded, the new run used the cache).
- p16atest_ref (old code) vs p16atest_new (new code): eval loss 0.5935/0.5635 vs 0.5942/0.5634; dev
  sp_BLEU 64.13/65.38 vs 64.20/65.33; test sp_BLEU 67.04 vs 66.90; both `model_global_step: 40`,
  `model_dev_check: ok`.
- These differences are run-to-run GPU noise, not the refactor: p16atest_resume ran the *same* new code
  and config, and at step 10 (before any resume) its train loss differs from p16atest_new by more
  (0.7395 vs 0.7329) than the reference does (0.7305).
- Resume: cancelled after checkpoint-20; the second attempt logs `Resume from the last checkpoint: True`,
  its progress bar restarts at 21/40, and it ends with `model/`, `model_dev_check: ok` and `scores.json`
  (test sp_BLEU 67.07).
- [ ] `black`/`flake8` clean on the two files (not installed in the hfmt env; not run); `python -c "import hfmt.sft_translation"` works without
      torch (heavy imports stay inside functions).
- [x] Tests pass (no-torch pair; `test_sft_translation.py` on the cluster).
- [x] Cluster: the same SFT run after the move (`run_name=p16atest_new`, same overrides and seed):
  - same SFTConfig dump in the log;
  - training/eval losses match A1 to bf16 noise; same `model_global_step` and `model_dev_check: ok`;
  - test scores match;
  - resume: cancel a copy after its first checkpoint and `launch.sh sft <run>/run_config.yaml` continues.
- [x] Delete the `p16atest_*` folders (deleted 2026-11-18). `todo.md` A ticked; commit message presented.

## B. Config: schema, `conf/cpt/`, checks, forced train keys (plan §1)

### B1. Schema (`hfmt/hydra_config.py`)
- [x] `DataConfig.monotext: Optional[List[str]] = None`.
- [x] `CPTConfig` (`use_monotext`, `use_bitext`, `bitext_format`, `monotext_ratio`, `max_length`,
      `packing_strategy`, all `MISSING`), with a docstring; `Config.cpt: Optional[CPTConfig] = None`.
- [x] Module docstring: `cpt` and `data.monotext` are read by `cpt_monotext_bitext.py` (and `grid.py`);
      SFT ignores `monotext`, CPT ignores `test`.

### B2. `conf/cpt/default.yaml`
- [x] Header (1 line, what it's for). Values: `use_monotext: true`, `use_bitext: true`,
      `bitext_format: separate`, `monotext_ratio: 0.5`, `max_length: 1024`, `packing_strategy: bfd`.
- [x] Not added to `conf/config.yaml`'s defaults (experiments add `- cpt: default`).

### B3. `check_config` additions
- [x] With `cpt` set:
  - `bitext_format ∈ {separate, concat}`, `packing_strategy ∈ {bfd, wrapped}`;
  - at least one of `use_monotext` / `use_bitext`;
  - `use_monotext` needs a non-empty `data.monotext`;
  - `0 < monotext_ratio < 1` when both sources are on.
- [x] Without `cpt`: nothing new (`data.monotext` allowed and ignored).
- [x] Docstring updated.

### B4. Forced train keys
- [x] `CPT_FORCED_TRAIN = {packing: True, completion_only_loss: False, inference_callback_interval: 0,
      metric_for_best_model: "eval_loss", greater_is_better: False}` with a one-line reason each, in
      `hydra_config.py`.
- [x] `apply_cpt_overrides(cfg) -> list[(key, old, new)]`: sets them in place when `cfg.cpt` is set,
      returns what changed (empty on a saved CPT config or an SFT config).

### Done when
- [x] `test_grid.py` additions (with B's code; the launch-path tests come in C):
  - a CPT experiment composes and passes `check_config`;
  - each `check_config` CPT rule fails with its message;
  - `apply_cpt_overrides` on `train=longrun` changes the five keys, a second call changes nothing, and an
    SFT config is untouched;
  - a `dataset.yaml` with `monotext` composes for an SFT experiment;
  - old saved configs (no `cpt`) still compose.
- [x] Tests pass. `todo.md` B ticked; commit message presented.

## C. Launch path: `grid.py`, `launch.sh`, `job.sh` (plan §4)

### C1. `grid.py`
- [x] CLI: `--cpt` (next to `--inf`/`--merge`) → `main_sft(..., kind="cpt")`; plain → `kind="sft"`.
- [x] `main_sft(kind)`:
  - after composing each job: `kind="cpt"` and no `cpt` → exit ("not a CPT experiment; use launch.sh
    sft"); `kind="sft"` and `cpt` → exit ("a CPT experiment; use launch.sh cpt");
  - `kind="cpt"`: refuse command-line overrides of a forced key (`train.packing=...` etc.: "CPT sets
    train.<key>"); then `apply_cpt_overrides` on each job before `_save`, logging each change
    (`grid: cpt: train.<key> <old> -> <new> (<reason>)`);
  - (`launch.sh` sets `KIND=cpt` itself for a new CPT launch; grid prints `KIND` only for a resume.)
- [x] `main_resume`: the launch kind must match the saved config (`cpt` present ⇔ `cpt`), else exit;
      prints `KIND=cpt_resume` or `KIND=sft_resume`.
- [x] `main_inf`/`main_merge`: unchanged (check a CPT `run_config.yaml` goes through both).
- [x] Q14: in `_compose_checked` (all flows), an absolute `checkpoint` path that doesn't exist exits
      ("checkpoint <path> doesn't exist (not merged/finished yet?)"). Hub ids and relative names untouched.
- [x] Module docstring: the `cpt` flow.

### C2. `launch.sh`
- [x] Accept `cpt`: usage line, `case`, `grid_args=(--cpt "$yaml")`; the usage comment gets a cpt block
      (launch, resume) in the style of the sft one.

### C3. `job.sh`
- [x] Kinds `cpt` and `cpt_resume` → `python -m hfmt.cpt_monotext_bitext` (same arguments as `sft`;
      `cpt_resume` sets `HFMT_SFT_RESUME=1`).
- [x] Requeue rule: `sft|cpt` with `SLURM_RESTART_COUNT > 0` and a checkpoint → `<kind>_resume`.
- [x] Usage line and header comment.

### Done when
- [x] `test_grid.py` additions:
  - `launch.sh cpt` (grid `--cpt`) saves `<output_root>/<run_name>/run_config.yaml` with `cpt` and the
    forced train values;
  - `--cpt` on an SFT yaml and plain on a CPT yaml are refused;
  - a forced-key override on a CPT launch is refused; the same override on an SFT launch is fine;
  - a CPT sweep (`cpt.bitext_format=separate,concat`) gives 2 folders and a `.grid.txt`;
  - resume of a CPT run prints `KIND=cpt_resume`; a kind mismatch is refused;
  - `--inf` and `--merge` accept a CPT run folder;
  - Q14: a missing absolute checkpoint is refused; an existing one and a hub id pass.
- [x] `DRYRUN=1 bash egs/launch.sh cpt <cpt yaml>` prints `sbatch ... egs/job.sh cpt <folder>`; the same
      for a resume of a scratchpad copy of a CPT run folder (`job.sh cpt_resume`).
- [x] `bash -n egs/launch.sh egs/job.sh`. Tests pass. `todo.md` C ticked; commit message presented.

## D. CPT script: `hfmt/cpt_monotext_bitext.py` (plan §2, §3)

### D1. Data prep (module-level functions, options through `fn_kwargs`)
- [ ] Bitext examples: `separate` → each src and trg line its own `text`; `concat` → `src + " " + trg`.
- [ ] Monotext examples: each line its own `text`.
- [ ] Empty (whitespace-only) lines dropped in both.
- [ ] Loading: bitext as in SFT (src/trg files joined line by line); monotext with
      `load_dataset("text", data_files=list(cfg.data.monotext))`.
- [ ] Mixing: both sources → `interleave_datasets([mono, bitext], probabilities=[r, 1-r],
      seed=cfg.train.seed, stopping_strategy="all_exhausted")`; one source → that source.
- [ ] Dev: `data.dev` bitext formatted with `bitext_format`, also when `use_bitext` is false.
- [ ] Final datasets hold only the `text` column.
- [ ] Logs: example counts per source before and after mixing, token share per source (from tokenized
      lengths), examples longer than `cpt.max_length` (R4), and the time of each step (R8).

### D2. `main()`
- [ ] `@hydra.main` like `sft_translation.py`; `check_config`; exit if `cfg.cpt` is unset ("use
      sft_translation.py").
- [ ] `apply_cpt_overrides(cfg)` and log any change (none on a saved config).
- [ ] Log once that `data.test`, `instruction` and `decode` are not used by CPT.
- [ ] Shared calls: `init_wandb`, `load_tokenizer`, `build_model`.
- [ ] `trainer_args(cfg, outdir, use_bf16, dataset_text_field="text", max_length=cfg.cpt.max_length,
      packing_strategy=cfg.cpt.packing_strategy, dataset_num_proc=cfg.slurm.cpus)`; tokenizer's own
      `eos_token` (Q7); `processing_class=tokenizer`.
- [ ] Callback: transformers' `EarlyStoppingCallback(train.early_stopping_patience,
      train.early_stopping_threshold)`. No MT callback, no decoding.
- [ ] `inspect_first_batch` (shows the packed row and the EOS/BOS between examples).
- [ ] `train(...)`, `save_model(...)`.
- [ ] model/ check: `trainer.evaluate()` on the saved model; compare `eval_loss` with `log_history`'s
      `eval_loss` at `model_global_step` (relative tolerance constant, initially 1e-3, set from E/G
      runs); write `model_dev_scores` (`{"eval_loss": ...}`) and `model_dev_check` ("ok" / "unchecked: …" /
      "MISMATCH: …") into `training_info.yaml`; log like SFT. A function `check_model_dev_loss(log_history,
      step, eval_loss, tolerance)` that is testable without torch.
- [ ] W&B summary: `dev/model/eval_loss` (numbers only). No test decoding.
- [ ] Module docstring: how it's launched, direct-run example, what it reads/ignores.

### D3. Tests: `hfmt/test_cpt_monotext_bitext.py` (needs `datasets`, not torch)
- [ ] separate / concat (space) / monotext example building on made-up lines; empty lines dropped.
- [ ] Dev formatting follows `bitext_format`, also with `use_bitext: false`.
- [ ] Interleave with r = 0.25 on two small made-up sets: every example of both appears; monotext share of
      the mixed set ≈ 25%; same seed → same order.
- [ ] `check_model_dev_loss`: ok / mismatch / unchecked.

### Done when
- [ ] `black`/`flake8` clean; `python -c "import hfmt.cpt_monotext_bitext"` works without torch.
- [ ] D3 passes in the cluster env; no-torch tests pass.
- [ ] Cluster smoke test (Tiny Aya, made-up text files in the scratchpad as monotext/bitext, via a
      scratch experiment yaml; `train.max_steps=20 train.eval_steps=10 wandb.enabled=false`), `LOCAL=1` or
      sbatch:
  - batch dump: examples packed in one row, each ending with `<|END_OF_TURN_TOKEN|>` (R2 seen), BOS where
    the tokenizer adds it;
  - forced keys logged; no decoding anywhere in the log;
  - `model/` and `training_info.yaml` with `model_dev_check: ok`; no `scores.json`.
- [ ] Delete the smoke-test folder. `todo.md` D ticked; commit message presented.

## E. `tools/run_status.py` (plan §4, §5)

- [ ] `inspect_folder`: `is_cpt` = the saved config has `cpt`; `has_test = False` for a CPT run.
- [ ] `fix_command`: `cpt` instead of `sft` for a CPT run (`launch.sh cpt <rc>`, `sbatch ... egs/job.sh cpt
      <folder>`).
- [ ] `find_experiments` / `read_experiment` (D5):
  - an experiment's defaults start with `config` followed by a plain string whose last path component
    starts with `dataset` (`dataset`, `../dataset`, `dataset_small100.yaml`); it sets `run_name`;
  - an experiment folder is any searched folder holding such a yaml (outputs/, run folders and
    checkpoints still skipped), not only folders with a `dataset.yaml`; the folder holding a root
    `outputs/` counts as before.
- [ ] Module docstring: CPT runs and the experiment rule.

### Done when
- [ ] `tools/test_run_status.py` additions:
  - a CPT run with `model/training_info.yaml` is "finished" and its checkpoints deletable;
  - an interrupted CPT run's fix command is `launch.sh cpt ...`; a failed one's is `... egs/job.sh cpt ...`;
  - a pair folder with `- ../dataset` yamls and no `dataset.yaml` is found, from its parent and from its
    `outputs/`; a yaml whose second default isn't a dataset still isn't an experiment.
- [ ] Tests pass. Run it read-only on `egs/mmtc` and compare with the output before the change (same
      rows). `todo.md` E ticked; commit message presented.

## F. Experiments: `egs/synlrl/` (plan §5)

- [ ] `egs/synlrl/dataset.yaml`: `# @package data`, one-line header, `seed1` train, `monotext:
      [${oc.env:HFMT_DATA_ROOT}/synth1.${data.src}]`, `flores200.smalldev-256` dev, `flores200.dev` test.
- [ ] For each of `tuk-eng`, `shn-eng`, `ewe-eng` (instruction: Turkmen / Shan / Ewe):
  - [ ] `local_setting.sh.example` (`export HFMT_DATA_ROOT=/path/to/...`, one comment line);
  - [ ] `sft_lora.yaml`: one-line header; `config`, `../dataset`, `override peft:
        lora_tiny_aya_all_linear`; `run_name: sft_lora`; Tiny Aya global; `data.src/trg`;
        `decode.batch_size: 64`; tags `[synlrl, <pair>]`;
  - [ ] `cpt_lora.yaml`: as in plan §5 (`- cpt: default`, `run_name: cpt_lora`, small batch +
        gradient accumulation, tags with `cpt`);
  - [ ] `sft_lora_after_cpt.yaml`: `sft_lora.yaml` with `run_name: sft_lora_after_cpt`, `checkpoint:
        ${oc.env:HFMT_ROOT}/egs/synlrl/<pair>/outputs/cpt_lora/model_merged`.
- [ ] `egs/synlrl/README.md` (short, like `egs/mmtc/README.md`): data files `$HFMT_DATA_ROOT` must hold
      (names only), `local_setting.sh`, the four-command workflow (CPT, merge, SFT after CPT, baseline SFT).
- [ ] Nothing under `egs/synth_lrl/` is touched.

### Done when
- [ ] With a scratch `local_setting.sh` (pointing at the real data root, not committed):
  - `DRYRUN=1 bash egs/launch.sh cpt egs/synlrl/tuk-eng/cpt_lora.yaml` composes and prints the sbatch line;
  - the same for `sft_lora.yaml`;
  - `sft_lora_after_cpt.yaml` is refused by Q14 until `model_merged/` exists (expected).
  - Delete the folders the dry runs created.
- [ ] `tools/run_status.py egs/synlrl` lists the yamls as not launched (needs E).
- [ ] `todo.md` F ticked; commit message presented (files staged by name).

## G. Quick test: Qwen3.5 packing leak (plan R1; independent, any time)

- [ ] `features/16_cpt/test_packing_leak.py`:
  - made-up sentences;
  - load with `mt_model.load_base_model` (lora mode = bf16, no adapter needed);
  - per-example loss (1) one sequence at a time and (2) as one packed padding-free batch from TRL's
    `DataCollatorForLanguageModeling(padding_free=True)` (position_ids reset, no attention mask);
  - models: Qwen3.5-0.8B or 2B (hybrid) and Tiny Aya global or Gemma 3 1B (control);
  - prints the per-example loss difference and the max.
- [ ] `features/16_cpt/test_packing_leak.sh`: sbatch wrapper that sources `install/path.sh` and waits for
      its GPU as `egs/job.sh` does.
- [ ] Give the developer the command; read the output before concluding.
- [ ] Record the result in `plan.md` R1. If the leak is confirmed on Qwen3.5 only: note "use
      `cpt.packing_strategy=wrapped` for Qwen3.5" in the README CPT section (or a
      `conf/cpt/` option for it), and decide with the developer.

## H. Docs (plan "Docs")

- [ ] README: "Continued pre-training" section after "Merging and a second round of SFT":
  - `launch.sh cpt`, the `cpt:` options, `data.monotext` in the shared dataset yaml;
  - the forced train keys and why any `conf/train/` schedule works;
  - `monotext_ratio` (mix, not amount; lines, not tokens);
  - CPT → merge → SFT with `egs/synlrl`;
  - packing caveats (R1 result, batch size R3).
- [ ] README option list: the `cpt/` group.
- [ ] CLAUDE.md:
  - structure tree: `cpt_monotext_bitext.py`, `train_common.py`, `conf/cpt/`, `egs/synlrl/`,
    `test_cpt_monotext_bitext.py`;
  - experiment convention: defaults start with `- config` and a dataset yaml (`- dataset`,
    `- ../dataset`, …);
  - a CPT convention bullet; `launch.sh <sft|cpt|inf|merge>`; tests list.
- [ ] `decisions.md` entry (16_cpt):
  - separate CPT and SFT yamls (and why, over a shared yaml);
  - `cpt` as the run-kind marker, with no required key;
  - CPT never decodes; forced train keys, applied in `grid.py` so the saved config says what ran;
  - the packing strategy after R1;
  - the eval-loss model/ check.
- [ ] Tests pass. `todo.md` H ticked; commit message presented.

## I. Developer validation (cluster)

- [ ] `egs/synlrl/<pair>` once `synth1.<src>` exists: CPT (`cpt_lora.yaml`), merge, `sft_lora_after_cpt.yaml`,
      and the `sft_lora.yaml` baseline. Check:
  - the CPT log: forced keys, example counts and token share per source, packing truncation count;
  - memory and batch size (R3); adjust `train_batch_size` / `gradient_accumulation_steps` in `cpt_lora.yaml`;
  - preprocessing time per step on the real data (R8);
  - `model_dev_check: ok`; set the eval-loss tolerance from the measured difference if needed;
  - `tools/run_status.py egs/synlrl/<pair>/outputs` reports the CPT run as finished.
- [ ] Resume a cancelled CPT run (`launch.sh cpt <run>/run_config.yaml`).
- [ ] A CPT launch with `override train: longrun` logs the forced keys and never decodes.
