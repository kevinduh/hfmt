# Plan — 05_wandb_run_name_prefix

## Problem

W&B sweep run *names* collide across experiments. `derive_run_identity` names a sweep job
by its bare swept-param label (`run_name = cfg.wandb.name or label`), with no experiment
prefix. Any two experiments that sweep the same axis produce byte-identical names — e.g.
`mmtc_fr-en_sft1_fullft_modelsweep` and `..._lora_modelsweep` both emit
`basemodel-qwen2.5-0.5b` / `basemodel-qwen2.5-1.5b`. In W&B's flat run list they are
indistinguishable, even though their *groups* (`<base-group>-sweep-<ts>`) differ.

Plain (non-sweep) runs are already prefixed (`<experiment>-<label>`, feature 04). The sweep
branch is the lone inconsistency.

## Approach

Prefix the experiment onto the sweep run name, mirroring the plain-run branch:

```python
run_name = cfg.wandb.name or f"{exp}-{label}"   # was: cfg.wandb.name or label
```

- Scope is one branch in `derive_run_identity` (`hfmt/sft_translation.py`).
- The `cfg.wandb.name` manual override still short-circuits.
- The run *directory* leaf stays the bare label (set independently via the `hfmt_runlabel`
  resolver in `conf/config.yaml`), so only the W&B name changes.

## Risks / trade-offs

- **Loosens the feature 03 invariant** "run-dir basename *is* the W&B run name." On disk it
  still is; in W&B the name is now `<experiment>-<basename>`. This is a mental-model/docs
  change, not a correctness one — `run_id` (the `resume` key) already embedded the experiment
  and is untouched, so requeue behavior is unchanged. Mitigation: update the docstrings,
  README, and decisions.md so the invariant is stated correctly.
- **Redundancy inside a single group's view.** When grouped by W&B `group`, every run repeats
  the long prefix; the swept axis is the informative part there. Accepted: the broken view is
  the *flat* cross-experiment list, and that is the one this fixes. Names also grow (~55 chars).
- **Confidentiality:** unchanged. The experiment name is config (not data); `build_run_label`
  remains the sole data-path chokepoint. No data content enters the name.

## Validation

- Update and run `hfmt/test_run_label.py` (pure functions, no GPU): the sweep-name assertion
  flips from `name == basename` to `name == f"{experiment}-{basename}"`; `wandb.name` override,
  `run_id`, grouping, tags, and plain/single-run tests stay green.
