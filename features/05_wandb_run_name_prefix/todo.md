# TODO — 05_wandb_run_name_prefix

## A. Core change

- [x] Prefix the experiment onto the sweep run name in `derive_run_identity`
      (`hfmt/sft_translation.py`): `run_name = cfg.wandb.name or f"{exp}-{label}"`.
- [x] Keep the `cfg.wandb.name` manual override as the short-circuit.
- [x] Confirm the run *directory* leaf is unaffected (still the bare label via `hfmt_runlabel`).

## B. Docs / invariant

- [x] Update `derive_run_identity` docstring: name is `<experiment>-<label>`; dir basename stays
      the bare label; name and basename differ by the prefix.
- [x] Fix README "Outputs mirror W&B" bullet (no longer "dir basename is exactly the W&B run name").
- [x] Add a `decisions.md` entry recording the decision and that it loosens the feature 03 invariant.

## C. Tests

- [x] Rewrite the sweep-name test: `name == f"{experiment}-{basename}"` (was `name == basename`).
- [x] Verify `wandb.name` override, `run_id`, group, tags, plain-run, and single-run tests stay green.
- [x] Run the suite (`python hfmt/test_run_label.py` and `pytest`): 23 passed.

## Notes

- `request.md` is developer-owned and intentionally not created here.
