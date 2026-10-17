I want to simplify the current hydra sweep setup, which (1) assumes a single experiment yaml file like egs/mmtc/fr-en/sft1.yaml can be used for both a single run and multiple runs of all sweeper params, and (2) goes through logic like sweep_probe.py and sweep.enable=true. Instead, I just want to have two kinds of experiment yaml files: one kind is only a single run will be used as such, the other kind has a grid of hyperparams to sweep and will always be swept fully when invoked. The main goal is to simplify the code (reduce number of lines and effort needed to understand how a job starts). To illustrate, I also want you to clean up the experiment yaml files in egs/mmtc/fr-en/ so that there are only two (based on sft1_lora)

## Details (from the code walkthrough)

### Why the current setup exists
`egs/run.sh` always launches with `--multirun`, because the submitit Slurm launcher only runs in
multirun mode. Hydra cannot switch off a grid kept in `hydra.sweeper.params` from the CLI
(`~hydra.sweeper.params` errors; `hydra.sweeper.params={}` is accepted but ignored). Feature 04
therefore added a custom `sweep: {enabled, params}` block plus `hfmt/sweep_probe.py`, which
composes the experiment on the login node and, only when `sweep.enabled`, prints the axes as bare
comma-list overrides for `run.sh` to forward. The only thing this buys is the on/off switch.
With separate single-run and sweep files, that switch is no longer needed.

### Behaviour verified on the pinned Hydra (1.3.7)
I tested this with a scratch app laid out like the repo: a nested experiment group, `# @package _global_`,
selected via `+group=option`, always launched with `--multirun`:
- An experiment file with no grid -> exactly 1 job.
- An experiment file with `hydra.sweeper.params` -> the full cross product, including
  config-group axes (e.g. `basemodel: qwen2.5-0.5b,qwen2.5-1.5b`) despite the file's own
  `override /basemodel` default.
- CLI comma-lists still add axes (`lora.lora_r=8,16,32`), and a single-value CLI override of a
  swept key pins that axis (`train.seed=37`).
- The overrides the sweeper generates appear in `hydra.job.override_dirname`, so the existing
  run-label / W&B naming (`_label_pairs`, `build_run_label` in `hfmt/sft_translation.py`) should
  keep working unchanged.
- Correction to `decisions.md`: Hydra rejects only *comma-list* values on `hydra.*` keys
  from the CLI ("Sweeping over Hydra's configuration is not supported"); a single-value
  `hydra.sweeper.params.<k>=<v>` is accepted.

### What I want
- **Single-run experiment**: no grid at all; `bash egs/run.sh <exp>.yaml` launches exactly one job.
- **Sweep experiment**: the grid lives in Hydra's native `hydra.sweeper.params` inside the
  experiment yaml; invoking it always runs the full grid.
- **Remove** `hfmt/sweep_probe.py`, its call and `${sweep_axes}` in `egs/run.sh`, the
  `sweep`/`SweepConfig` block in `hfmt/hydra_config.py`, and any `sweep`/`sweep.*` special-casing
  that becomes dead (`exclude_keys` in `conf/config.yaml`, filters in `_label_pairs`).
- **egs/mmtc/fr-en/**: reduce the experiments to two, both based on `sft1_lora.yaml`: the single run
  (`sft1_lora.yaml`) and one sweep variant of it. Remove the others (`sft1.yaml`,
  `sft1_fullft.yaml`, `sft1_fullft_modelsweep.yaml`, `sft1_lora_modelsweep.yaml`,
  `sft1_lora_model_lr_sweep.yaml`, `sft1_refine1.yaml`, plus the untracked `sft1_lora.yaml.old`).
  `dataset.yaml` and `inf.sh` stay.
- **Update docs and tests** that describe the old mechanism: README "Sweeping parameters",
  CLAUDE.md (parameter-sweep bullet, plus stale `egs/experiment/` references), `decisions.md`
  (record the new decision and the correction above), comments in `conf/config.yaml` /
  `hfmt/hydra_config.py` / launcher yamls, `hfmt/test_run_label.py`, and
  `tools/run_on_free_gpu.sh` if it relies on the sweep block.

### Things to keep
- Ad-hoc CLI sweeps on any experiment (`lora.lora_r=8,16,32`).
- Run-label / output-dir / W&B naming behaviour and the confidentiality rules (only
  hyperparameters / config-file names in labels; nothing data-bearing to W&B).
- The login-node fast-fail for a missing `dataset.yaml` that `sweep_probe.py` provides today
  (otherwise it only surfaces on the compute node after queueing), as a cheap check in `run.sh`.

### Open questions for plan.md
- Name of the sweep file (e.g. `sft1_lora_sweep.yaml`) and which grid it carries (e.g. the former
  coarse 24-job grid lr x lora_target x lora_r x seed, or the smaller basemodel x lr grid).
- Whether to tell the two kinds apart only by convention (file name and a header comment) or to
  have `run.sh` print the job count / grid before submitting.
- Whether full fine-tuning (`sft1_fullft*` + the `slurm_fullft` launcher) needs a surviving example,
  or whether the README is enough documentation once those yamls are deleted.
- What to do with the untracked `egs/experiment/` directory and the `egs/mmtc/zh-en/` shell
  recipe (out of scope unless they reference the sweep block).
