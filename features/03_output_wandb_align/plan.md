# Plan — 03_output_wandb_align

**Status:** finalized — all decisions locked (D1–D5; Q1–Q4 resolved in chat). See `todo.md`
for the actionable breakdown and `request.md`.

**Core mechanism validated off-cluster (hydra 1.3.7, in-process basic sweeper).** A standalone
probe mirroring §A/§B confirmed: the custom resolver *does* drive `hydra.sweep.subdir` (R1); a
2×2×2 grid produced strict labels
`lora_r-8__lora_target-qv__learning_rate-2e-05__seed-37`, … all under one shared `sweep-<ts>/`
parent (R3); a `data.train_yaml=/secret/…` override was dropped from the label (R4,
confidentiality + no stray `/`); single runs used the timestamp dir; the empty-label case fell
back to `job0`. **Caveat:** OmegaConf parses `2e-4` to a float, so the label stringifies as
`learning_rate-0.0002` (canonical, but sci-notation is lost — see R10). Still to verify
on-cluster: that `sweep.dir`'s `${now}` is resolved *once on the login node* and shared by all
**submitit** jobs (the probe used the in-process sweeper).

## Request (restated)

1. Run-output directories (`outputs/.../eval.pred.trg`, `dev.step_*.pred`, the saved adapter)
   currently **don't line up** with the W&B run that produced them, so going from a W&B run to
   its prediction files (to score with `analysis/run_sacrebleu.py`) is guesswork.
2. **Use the W&B names** for the paths so the two are aligned.
3. A **nested** directory structure is acceptable/expected — confirm it's the right shape.
4. Directory names must be **shell-safe** (no characters that need escaping).

## The misalignment today (read before editing)

Everything is launched through `egs/run.sh`, which always passes `--multirun`, so **every** run
is a Hydra *sweep* (even a single job). Three naming schemes are derived from **different
sources** and never share a key:

| What | Where set | Example |
|---|---|---|
| Output run dir | `conf/config.yaml:29` `hydra.sweep.subdir=${hydra.job.num}_${now:…}` | `outputs/mmtc_fr-en_sft1/0_2026-10-07_14-01-07/` |
| Adapter dir | `sft_translation.py:488` `model.save_pretrained(outdir + "_b")` | `…/0_2026-10-07_14-01-07_b/` |
| W&B group | `derive_run_identity()` | `mmtc_fr-en_sft1-sweep-2026-10-07_14-01-07` |
| W&B run **name** (shown in UI) | `derive_run_identity()` → swept param tokens | `train.learning_rate=2e-4,model.lora_r=8,…` |

The W&B *id* does encode the dir (`mmtc_fr-en_sft1_0_2026-10-07_14-01-07`), but the UI surfaces
the *name* (param tokens), so there is **no visible path** from a W&B run back to its output dir.
Separately, `${now:…}` lives in `sweep.subdir`, so it is re-resolved per job (visible in the
listing: jobs of one launch carry *different* timestamps) — the dir timestamp is not even a
reliable per-sweep key.

## Decisions (locked with developer)

- **D1 — Nested layout mirroring W&B's group→run hierarchy.**
  ```
  outputs/<experiment>/sweep-<sweep_ts>/<run_label>/        # multirun (the run.sh path)
  outputs/<experiment>/<run_ts>/                            # plain single run (local, no -m)
  ```
  `<experiment>/` groups all its sweeps on disk; `sweep-<sweep_ts>/` clusters one sweep's runs
  (corresponds 1:1 to the W&B group `<experiment>-sweep-<sweep_ts>`); `<run_label>/` is the
  per-run name shown in W&B.
- **D2 — The adapter folds into the run dir as `model/`** (replacing the `outdir + "_b"`
  sibling). One run = one self-contained directory.
- **D3 — Strict, shell-safe, *descriptive* run labels** restricted to `[A-Za-z0-9._-]` (no `=`,
  no `,`, no spaces). **No abbreviation** (Q4): only the config-group prefix is stripped
  (`model.`/`train.`/`decode.`/`data.`), so leaf keys survive **verbatim** —
  `lora_r`, `lora_target`, `lora_alpha`, `lora_dropout`, `learning_rate`, `seed`. Each pair is
  `<leaf>-<value>`; **pairs are joined by `__`** (double underscore) so boundaries stay legible
  even though leaf names themselves contain `_`. Example:
  `lora_r-8__lora_target-all-linear__learning_rate-0.0002__seed-37`.
- **D4 — Single source of truth: the run label is computed once and used for both the directory
  name and the W&B run name**, so they cannot drift.
- **D5 (Q1–Q3) — Custom resolver** (not the Hydra-native-separators fallback); **forward-only**,
  no migration of existing `outputs/` dirs; the `run_sacrebleu.py` convenience is **out of
  scope**.

> Note on exact 1:1 with the W&B group string: the on-disk sweep dir is `sweep-<ts>` (its parent
> dir already carries the experiment), while the W&B group **string** stays
> `<experiment>-sweep-<ts>` (W&B is a flat per-project namespace, so the experiment must remain
> in the group name to stay findable). The mapping is exact and documented, not character-equal.

## Proposed approach

### A. Directory layout (`conf/config.yaml`)

Move the shared timestamp **up** to the sweep root (resolved once on the login node at submit
time → identical across all submitit jobs of a launch, and stable across a Slurm requeue), and
make the per-job subdir the computed run label:

```yaml
hydra:
  run:
    dir: ${output_root}/${experiment}/${now:%Y-%m-%d_%H-%M-%S}
  sweep:
    dir: ${output_root}/${experiment}/sweep-${now:%Y-%m-%d_%H-%M-%S}
    subdir: ${hfmt_runlabel:${hydra.job.override_dirname},${hydra.job.num}}
  job:
    config:
      override_dirname:
        exclude_keys: [experiment, sweep, hydra/launcher]   # structural selectors, not params
```

`${hfmt_runlabel:…}` is a **custom OmegaConf resolver** (new; see B) that turns Hydra's
`override_dirname` into the strict label. `override_dirname` keeps Hydra's default separators
(`kv_sep='='`, `item_sep=','`) — those two characters never appear inside a key or a numeric
value, so the resolver can split pairs reliably (`,` then first `=`) before reformatting.

### B. Run-label builder + resolver (`hfmt/sft_translation.py`)

A single pure function, used by both the resolver (dir name) and `derive_run_identity()` (W&B
name) — guaranteeing D4. No abbreviation (D3/Q4): the config-group prefix is stripped so leaf
keys survive verbatim, and pairs are joined by `__`:

```python
def _strict(s) -> str:                                # collapse to shell-safe charset (keeps _)
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(s)).strip("-")

def build_run_label(override_dirname: str, job_num="0") -> str:
    """Strict, shell-safe ([A-Za-z0-9._-]), *descriptive* label from Hydra's override_dirname.
    Drops structural + data/path tokens (confidentiality) and strips the config-group prefix
    only (leaf keys kept verbatim, e.g. lora_r/lora_target). Pairs are '<leaf>-<value>' joined
    by '__'. Falls back to ``job<N>`` when nothing informative remains."""
    pairs = []
    for tok in filter(None, override_dirname.split(",")):
        key, _, val = tok.partition("=")
        key = key.lstrip("+~")
        if key in ("experiment", "sweep") or key.startswith("hydra"):
            continue
        if key == "data" or key.startswith("data.") or "/" in val:   # confidentiality + no slashes
            continue
        leaf = key.rsplit(".", 1)[-1]                 # model.lora_r -> lora_r (verbatim)
        pairs.append(f"{_strict(leaf)}-{_strict(val)}")
    return "__".join(pairs) if pairs else f"job{job_num}"
```

Registered next to `register_configs()` so it exists before Hydra composes (run.sh runs
`python -m hfmt.sft_translation`, importing the module first):

```python
OmegaConf.register_new_resolver(
    "hfmt_runlabel", lambda od, n="0": build_run_label(str(od), str(n)), replace=True)
```

`replace=True` avoids the "already registered" error on re-import (submitit unpickle, `--cfg`).

### C. W&B identity from the path (`derive_run_identity()`)

Rewrite to **read the resolved output path** instead of recomputing from overrides, so the W&B
name is literally the directory basename:

- `run_label = basename(outdir)` → **W&B run name == dir name** (D4).
- multirun: `sweep_dir = basename(dirname(outdir))` (e.g. `sweep-2026-…`); W&B
  `group = f"{base_group}-{sweep_dir}"`? — simpler: keep `group = f"{base_group}-sweep-<ts>"`
  by taking the `<ts>` from `sweep_dir`. `run_id = sanitize(f"{experiment}_{sweep_dir}_{run_label}")`
  (unique per job, stable across requeue because the whole path is stable).
- single run: `group = base_group`, `run_name = f"{experiment}-{run_label}"` (run_label == ts).
- `_sweep_param_tokens()` is **subsumed** by `build_run_label()`; drop it (and fold its tag
  derivation into the new path — tags can be split back out of the label, or derived from
  `HydraConfig.overrides.task` as today, kept data-free).

### D. Adapter → `model/` (`sft_translation.py:488`)

`model.save_pretrained(os.path.join(outdir, "model"))`. Downstream: `inf_translation.py` takes
the adapter path via `-p/--peft`, so **no code change there** — callers just point `-p` at
`<run_dir>/model`. The legacy `egs/mmtc/fr-en/inf.sh` hardcodes an unrelated `qwen-qlora.1_b`
path (pre-Hydra recipe) — out of scope; note it in docs.

### E. Docs + validation

- Update the stale comments in `conf/config.yaml`, the `derive_run_identity` docstring, the
  README outputs/sweep section, and append a short `decisions.md` entry (aligned-path scheme).
- **Off-cluster (here):** unit-test `build_run_label`/`derive_run_identity` as pure functions
  (no GPU, no training deps); dry-run `python -m hfmt.sft_translation --multirun +experiment=…
  model.lora_r=8,16 --cfg hydra` to confirm the resolved `hydra.sweep.subdir`/`dir` are the
  strict labels and share one `sweep-<ts>` parent.
- **On-cluster smoke (user-run):** a 2-point sweep (`train.seed=37,42 train.max_steps=20`) →
  two dirs `…/sweep-<ts>/seed-37/` & `…/seed-42/`, each with `model/`, `eval.pred.trg`, and two
  non-clobbering W&B runs named `seed-37`/`seed-42` in group `<exp>-sweep-<ts>`.

## Phased breakdown (→ becomes todo.md after approval)

- **A — Label builder + resolver:** `build_run_label`, `_strict`, alias map, resolver
  registration; unit tests.
- **B — Config layout:** `conf/config.yaml` run/sweep dirs + `override_dirname.exclude_keys`.
- **C — W&B identity:** rewrite `derive_run_identity` to read the path; drop
  `_sweep_param_tokens`; keep confidentiality + requeue-resume; update tests.
- **D — Adapter dir:** `_b` → `model/`.
- **E — Docs/decisions + off-cluster validation; user on-cluster smoke checklist.**

## Risks

- **R1 — Resolver availability at dir-resolution time.** *(Validated off-cluster — see Status.)*
  `hydra.sweep.subdir=${hfmt_runlabel:${hydra.job.override_dirname},${hydra.job.num}}` resolves
  correctly with the in-process basic sweeper on hydra 1.3.7. Residual unknown: the cluster
  conda env may pin a different Hydra version, and the real path uses the **submitit** launcher
  (not probed here). **Fallback if it ever regresses:** Hydra-native `override_dirname` with
  `kv_sep='-'`, `item_sep='_'`, `exclude_keys=[…]` (gives `lora_r-16_lora_target-all-linear_…`
  — strict-safe, no aliasing/prefix-strip), with `derive_run_identity` reading that basename.
- **R2 — `override_dirname` parse ambiguity.** Relies on `=`/`,` never appearing in keys/values.
  True for current knobs (values like `all-linear`, `2e-4`, `0.1`, `adamw_torch_fused`). A value
  with a literal `,`/`=` would mis-split — none today; documented constraint.
- **R3 — Sweep-id sharing.** Per-sweep grouping needs `${now}` in `sweep.dir` to resolve once
  and be shared across jobs (and stable across requeue). Moving it from `subdir`→`dir` is what
  makes this true (and fixes today's per-job drift). *Probe confirmed one shared `sweep-<ts>/`
  across a 4-job grid in-process; still verify under submitit on-cluster.*
- **R4 — Confidentiality (CLAUDE.md).** The run label is now **both** the dir name *and* the
  W&B run name, so label construction is the single confidentiality chokepoint: it drops
  `data`/`data.*` keys and any value containing `/`. Only hyperparameters remain → safe. Must
  re-audit if the alias/exclusion lists change.
- **R5 — Name length / filesystem limit.** Many-axis sweeps can exceed 255 bytes per path
  component. Mitigation: cap length and append a short deterministic hash suffix to preserve
  uniqueness (keep the existing truncation in the sanitizer).
- **R6 — Label collisions.** Two runs in one sweep with identical informative overrides →
  identical label → dir/`run_id` clash. A grid never produces duplicate combos; the empty-label
  case falls back to `job<N>`. Flag only.
- **R7 — Backward compatibility.** Existing `0_2026-*` dirs won't be renamed (forward-only).
  Old W&B runs keep their old names. No migration planned (see open question Q2).
- **R8 — Downstream `_b` consumers.** Anything assuming the `…_b` sibling path for the adapter
  breaks; only `sft_translation.py` writes it and `inf_translation.py` takes it as an argument
  (no hardcode). Grep confirms no other references in-repo except legacy `inf.sh` (unrelated).
- **R9 — Off-cluster validation only.** No local GPU/training deps; only config composition +
  pure-function tests run here. A real sweep is a user step (as in `00_hydra`/`02_hydra_sweep`).
- **R10 — Numeric value normalization.** `override_dirname` stringifies the *parsed* value, so
  `train.learning_rate=2e-4` becomes `lr-0.0002` in the label (and thus the W&B name). Canonical
  and unambiguous, but the typed sci-notation is lost. Acceptable; flag in docs. (Avoid only by
  building the label from the raw override *strings* instead of `override_dirname` — not worth
  the added fragility.)

## Open questions

All resolved in chat (2026-10-08):

- **Q1 — Resolver vs. Hydra-native separators.** → **Custom resolver** (D5). Native-seps kept
  only as the R1 fallback if the cluster's Hydra ever regresses.
- **Q2 — Migrate existing outputs?** → **No.** Forward-only (D5); old `N_<ts>`/`_b` dirs stay.
- **Q3 — Scorer convenience.** → **Out of scope** (D5).
- **Q4 — Naming.** → **Descriptive, no abbreviation; keep `lora` in LoRA keys** (D3): strip only
  the group prefix, leaf keys verbatim (`lora_r`, `lora_target`, …), pairs joined by `__`.
