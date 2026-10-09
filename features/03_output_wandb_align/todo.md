# TODO — 03_output_wandb_align

Actionable breakdown of `plan.md` (decisions D1–D5; Q1–Q4 resolved). Scope: **align run-output
paths with W&B names** for the `sft_translation.py` workflow — a nested
`outputs/<experiment>/sweep-<ts>/<run_label>/` layout whose run-dir basename *is* the W&B run
name, strict/shell-safe descriptive labels from a custom Hydra resolver, and the adapter folded
into the run dir as `model/`. Groups are ordered by dependency; items within a group are roughly
ordered too. Check items off as completed.

*No GPU / training deps in this sandbox: validate config composition + pure functions only; a
real multi-job sweep is an on-cluster, user-run step (group E). The core resolver→subdir
mechanism is already prototyped off-cluster (hydra 1.3.7) — see plan Status.*

---

## A. Run-label builder + custom resolver (D3, D4, D5/Q1, Q4)
*The single source of truth for both the dir name and the W&B run name. Pure, testable; no
Hydra/torch needed to exercise `build_run_label`.*

- [x] Add `_strict(s)` to `hfmt/sft_translation.py`: collapse to the shell-safe charset
      `[A-Za-z0-9._-]` (regex `[^A-Za-z0-9._-]+` → `-`, strip leading/trailing `-`). Keeps `_`
      so config field names survive verbatim.
- [x] Add `build_run_label(override_dirname, job_num="0")`: split Hydra's `override_dirname` on
      `,`, then each token on the first `=`; **strip the config-group prefix only**
      (`key.rsplit(".",1)[-1]`) so leaf keys stay verbatim (`lora_r`, `lora_target`,
      `learning_rate`, `seed`) — **no abbreviation** (Q4); emit `<leaf>-<value>` pairs joined by
      `__`. Fall back to `job<N>` when no informative pairs remain.
- [x] **Confidentiality filter (R4, CLAUDE.md):** inside the loop drop `experiment`/`sweep`
      selectors, any `hydra*` key, and any `data`/`data.*` key or `/`-bearing value, so no data
      path enters the label (→ nor the W&B name). Comment ties it to CLAUDE.md. *(Verified: a
      `data.train_yaml=/proprietary/...` override and a `/`-bearing `model.checkpoint` leave no
      trace in the label.)*
- [x] Register the resolver next to `register_configs()` (module import time, before Hydra
      composes): `OmegaConf.register_new_resolver("hfmt_runlabel", lambda od, n="0":
      build_run_label(str(od), str(n)), replace=True)`. `replace=True` avoids the
      "already registered" error on submitit re-import / `--cfg`. *(Verified registered on
      import; `${hfmt_runlabel:${od},${num}}` → `lora_r-8__seed-37`, empty → `job4`.)*
- [x] Unit tests `hfmt/test_run_label.py` (13 cases, pytest + standalone): grid → descriptive
      label, lora/underscores kept, order preserved, data key + bare `data=` + `/`-value +
      structural selectors dropped, `+`/`~` prefixes, empty→`job<N>`, charset strictly
      `[A-Za-z0-9._-]`. **All 13 pass** (run off-cluster with hydra+pyyaml; no GPU/torch needed).

## B. Nested output layout (D1) — `conf/config.yaml`
*Depends on A (the resolver must exist). Move the shared timestamp up to the sweep root so all
jobs of one launch cluster under one `sweep-<ts>/` (R3); per-job subdir = the resolver label.*

- [x] Set `hydra.sweep.dir: ${output_root}/${experiment}/sweep-${now:%Y-%m-%d_%H-%M-%S}` and
      `hydra.sweep.subdir: ${hfmt_runlabel:${hydra.job.override_dirname},${hydra.job.num}}`.
- [x] Keep `hydra.run.dir: ${output_root}/${experiment}/${now:%Y-%m-%d_%H-%M-%S}` for plain
      (non-`-m`) single runs (no overrides → timestamp dir).
- [x] Add `hydra.job.config.override_dirname.exclude_keys: [experiment, sweep, hydra/launcher]`
      (belt-and-suspenders with the resolver's own filtering). *(Confirmed composed via
      `--cfg hydra`; `experiment` dropped from the raw override_dirname.)*
- [x] Refresh the now-stale comments in `conf/config.yaml` (they described the old
      `${job.num}_${now}` subdir) to document the new layout + the resolver + the shared `<ts>`.
- [x] Off-cluster check: ran the **real** `conf/config.yaml` through an in-process multirun
      (no-op app, no torch) with `--multirun +experiment=mmtc_fr-en_sft1 model.lora_r=8,16
      model.lora_target=qv,all-linear train.seed=37` → 4 dirs
      `outputs/mmtc_fr-en_sft1/sweep-<ts>/lora_r-8__lora_target-qv__seed-37`, … all under one
      shared `sweep-<ts>/`. (`--cfg hydra --resolve` can't be used here: it resolves the hydra
      subtree alone, so sibling `${output_root}`/`${experiment}` are out of scope — unrelated to
      this change. submitit-shared `<ts>` (R1/R3) still to confirm on-cluster — group E.)

## C. W&B identity read from the path (D4) — `derive_run_identity()`
*Depends on A+B. Make the W&B run **name == dir basename** so UI and disk match; keep
confidentiality, requeue-resume, and the single/sweep split.*

- [x] Rewrote `derive_run_identity()` to read the resolved `outdir`: `run_label =
      basename(outdir)`; multirun `sweep_dir = basename(dirname(outdir))` (e.g. `sweep-<ts>`).
      Signature simplified (dropped `overrides_task`); call site in `main()` updated.
- [x] Multirun: `group = f"{base_group}-{sweep_dir}"` → `<experiment>-sweep-<ts>` (experiment
      kept in the W&B group string for a findable flat namespace). `run_name = run_label`.
- [x] Single run: `group = base_group`, `run_name = f"{experiment}-{run_label}"` (label == ts).
- [x] `run_id` = sanitized `{experiment}_{sweep_dir}_{run_label}` (multirun) / `{experiment}_
      {run_label}` (single) — unique per job, stable across a Slurm requeue (whole path stable);
      `resume="allow"` unchanged in `init_wandb`.
- [x] **Removed `_sweep_param_tokens()`** (subsumed by `build_run_label`); sweep **tags** now
      split from the label (`run_label.split("__")`, data-free by construction) + the `sweep`
      tag + existing `cfg.wandb.tags`.
- [x] Updated the `derive_run_identity` docstring + the `main()` comment to the new scheme.
- [x] Unit-tested `derive_run_identity` (7 cases): name==basename, group from parent, tags from
      label, run_id unique+stable, single-run path, `wandb.name` override, no `/`/data anywhere.
      **All 20 tests pass.** End-to-end check with the real config confirmed: sweep dir
      `…/sweep-<ts>/lora_r-8__seed-37` → name `lora_r-8__seed-37`, group
      `mmtc_fr-en_sft1-sweep-<ts>`, both jobs share the group with distinct run_ids.

## D. Adapter → `model/` (D2) — `sft_translation.py`
*Depends on nothing in A–C; small and isolated.*

- [x] Changed `model.save_pretrained(outdir + "_b")` → `model.save_pretrained(os.path.join(
      outdir, "model"))` so the adapter nests inside the run dir; added a comment pointing at
      `inf_translation.py -p <run_dir>/model`.
- [x] Grep confirmed no in-repo consumer hardcodes the old `..._b` Hydra path
      (`inf_translation.py` takes `-p/--peft` as an argument; the only other `_b"` hit is the
      unrelated pre-Hydra `egs/mmtc/fr-en/inf.sh` path). New `<run_dir>/model` path to be
      documented in group E.

## E. Docs, decisions & validation handoff
*Depends on all above.*

- [x] README outputs/sweep section: documented the nested layout, run-dir basename == W&B run
      name, the `<run_dir>/model` adapter path, the `2e-4`→`0.0002` normalization (R10), and a
      `run_sacrebleu.py` invocation against `<run_dir>/eval.pred.trg`.
- [x] Appended a `decisions.md` entry: aligned output↔W&B naming via the `hfmt_runlabel`
      resolver; nested `sweep-<ts>/<run_label>` layout; sweep-id moved to `hydra.sweep.dir`
      (evolving the 02 decision); adapter as `model/`; confidentiality chokepoint; forward-only.
- [x] `CLAUDE.md` — the `outputs/` project-structure line ("Hydra run outputs (gitignored)") is
      still accurate; no change needed.
- [x] Wrote `features/03_output_wandb_align/validation.md`: off-cluster results (unit tests,
      real-config in-process multirun, identity derivation) + the **on-cluster smoke checklist**
      (`train.seed=37,42 train.max_steps=20` → `…/sweep-<ts>/seed-37|seed-42/` with `model/` +
      `eval.pred.trg`, two W&B runs in one group, and the submitit-shared `sweep-<ts>` check
      that the in-process probe can't cover — R1/R3).

---

### Out of scope (this feature)
- **Migrating existing `outputs/`** dirs (`N_<ts>`/`_b`) into the new scheme — forward-only
  (Q2). A rename helper can be added later if wanted.
- **`analysis/run_sacrebleu.py` convenience** (auto-locate `eval.pred.trg` from a run dir) — Q3.
- **Key abbreviation / alias map** — explicitly rejected (Q4: descriptive, keep `lora`).
- Layout changes for the other entry points (`inf_translation`, `train_seq2seq`,
  `decode_summarization`).
- Any real GPU training/validation in this sandbox (done on-cluster by the user).
