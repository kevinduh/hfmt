# 12_resume_sft — todo

Generated from `plan.md` (v2, final). Groups follow the plan's sections and are ordered by
dependency. Commit once per phase, subject `12_resume_sft-<phase>: <summary>`; Claude presents the
message and the developer commits. Each phase ends with its "Done when" checks.

Ground rules for every phase:
- **Change as little as possible.** Follow TRL's resume behavior as is.
- **No dataset content to W&B** (CLAUDE.md).
- **Don't touch the untracked leftovers or the developer's runs.** Stage files by name.
- **Tests:** `HFMT_ROOT=$(pwd) python hfmt/test_grid.py`.
- **Try-outs:**
  - `DRYRUN=1` checks run on copies of run folders in the scratchpad, never on real runs.
  - GPU runs go to `egs/mmtc/fr-en/outputs/p12<phase>…` and are deleted once confirmed.

## A. Login node + launch path (plan §1, §2)

### A1. `grid.py`
- [x] `main_sft`: if `exp_name == SAVED`, hand off to `main_resume`. Delete `reuse_dir` and the
      `run_name=` escape; `job_dir` is always `<output_root>/<name>`.
- [x] `main_resume(exp_dir, exp_name, args, parsed)`, following `main_merge`:
  - no comma lists; only `slurm.*` overrides, otherwise exit;
  - exit if `model/` exists (finished: use `launch.sh inf`);
  - exit if there is no `checkpoint-*`;
  - exit if the highest `checkpoint-N/` has no `trainer_state.json` (incomplete: delete it);
  - `_compose_checked` the saved config + overrides; write nothing;
  - warn if the current `git_rev` differs from the saved one;
  - log `resuming <folder> from checkpoint-N (step N of max_steps)`;
  - print `_print_jobs([run folder], ...)` with log `<run folder>/slurm-%A.out`, plus `KIND=sft_resume`.
- [x] Module docstring: the sft flow takes an experiment yaml; a `run_config.yaml` means resume.

### A2. `launch.sh` and `job.sh`
- [x] `launch.sh`: `KIND=$kind` before the `eval`, then `job=(egs/job.sh "$KIND")` (same effect as
      `${KIND:-$kind}`, but a `KIND` left in the user's shell can't leak in).
- [x] `launch.sh`: `LOCAL=1` appends (`>>`) to `local.log` / `merge-local.log`. (Not run here: a
      LOCAL resume trains; exercised in B.)
- [x] `launch.sh` usage comment: the re-run line becomes resume (`slurm.*` overrides only).
- [x] `job.sh`: `sft_resume) HFMT_SFT_RESUME=1 exec python -m hfmt.sft_translation ...` (same
      arguments as `sft`); usage line and comments.

### A3. Tests (`hfmt/test_grid.py`)
- [x] Replace `test_rerun_saved_config_reuses_its_folder_unless_renamed` with a resume test:
  - a fake `checkpoint-N/trainer_state.json`;
  - prints the run folder, `KIND=sft_resume` and a `slurm.gres` override;
  - leaves the folder's files unchanged.
- [x] A refusals test:
  - non-slurm override;
  - `run_name=`;
  - comma list;
  - finished run (`model/`);
  - no checkpoint;
  - incomplete last checkpoint.

### Done when
- [x] Tests pass.
- [x] `DRYRUN=1` on a scratchpad copy of a run folder:
  - prints `job.sh sft_resume <folder>` with the overridden `--gres`;
  - the refusals give their messages;
  - `cmp` shows `run_config.yaml` unchanged.
- [x] `todo.md` A ticked; commit message presented.

## B. Training job: `sft_translation.py` (plan §3)

- [x] `resume = os.environ.get("HFMT_SFT_RESUME") == "1"`; log it.
- [x] W&B on resume:
  - ID from `<run folder>/wandb/latest-run` (`run-…-<id>`);
  - `wandb.init(..., id=<id>, resume="allow")`;
  - no `wandb/` folder → a new run.
- [x] `trainer.train(resume_from_checkpoint=True if resume else None)`.

### Done when
Result (2026-10-19): p12b was cancelled 3 s after checkpoint-40, then resumed with
`slurm.time=1:00:00` (job 1099586). Checks:
- the log says `Resume from the last checkpoint: True`, the progress bar restarts at 41/60, and the
  log is appended;
- learning rate at steps 50/60 = reference (2.29e-05, 1.97e-07);
- losses: 0.5793/0.6262 vs reference 0.5794/0.6241; eval 0.5904 vs 0.5902;
- test BLEU 61.08 vs 61.02;
- W&B: one run ID (0b1scz0f), "run resumed", history continues at `_step` 7;
- TRL's `num_tokens` restarts from 0 on resume (plan risk 8);
- a second resume is refused.

Both test runs are deleted. `LOCAL=1` appending was not run (a local resume trains on the login
node).
- [x] GPU, `p12b` (lora, max_steps 60, save/eval every 20, small dev decode):
  - `scancel` after checkpoint-40;
  - `launch.sh sft <run>/run_config.yaml` resumes. The log says resuming from checkpoint-40, the
    run ends at step 60 with `model/`, `test1.pred.trg` and `scores.json`;
  - W&B: one run ID; history continues past step 40;
  - an uninterrupted `p12b-ref` of the same config: steps line up, losses close.
- [x] A second resume of the finished run is refused (`model/` exists).
- [x] `todo.md` B ticked; commit message presented.

## C. Docs (plan §4)

- [ ] README "Running SFT experiments": the re-run paragraph becomes resume. Changing a run =
      experiment yaml + overrides + a new `run_name`.
- [ ] CLAUDE.md "Launch path" bullet.
- [ ] `decisions.md`, feature 12 entry:
  - resume-only meaning;
  - environment variable, not a config key;
  - W&B continues the run;
  - TRL's default for callback state.
- [ ] Tests pass. `todo.md` C ticked; commit message presented.

## D. Developer validation

- [ ] Resume a real cancelled run (e.g. after a time limit or OOM, with a `slurm.*` override).
