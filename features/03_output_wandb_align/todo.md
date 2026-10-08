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

- [ ] Set `hydra.sweep.dir: ${output_root}/${experiment}/sweep-${now:%Y-%m-%d_%H-%M-%S}` and
      `hydra.sweep.subdir: ${hfmt_runlabel:${hydra.job.override_dirname},${hydra.job.num}}`.
- [ ] Keep `hydra.run.dir: ${output_root}/${experiment}/${now:%Y-%m-%d_%H-%M-%S}` for plain
      (non-`-m`) single runs (no overrides → timestamp dir).
- [ ] Add `hydra.job.config.override_dirname.exclude_keys: [experiment, sweep, hydra/launcher]`
      (belt-and-suspenders with the resolver's own filtering).
- [ ] Refresh the now-stale comments in `conf/config.yaml` (they describe the old
      `${job.num}_${now}` subdir) to document the new layout + the resolver.
- [ ] Off-cluster check: `python -m hfmt.sft_translation --multirun +experiment=mmtc_fr-en_sft1
      model.lora_r=8,16 --cfg hydra` → resolved `sweep.subdir`/`dir` are strict labels sharing
      one `sweep-<ts>/` parent. (`--cfg` + `-m` caveat from 02: use a small compose probe if
      needed.)

## C. W&B identity read from the path (D4) — `derive_run_identity()`
*Depends on A+B. Make the W&B run **name == dir basename** so UI and disk match; keep
confidentiality, requeue-resume, and the single/sweep split.*

- [ ] Rewrite `derive_run_identity()` to read the resolved `outdir`: `run_label =
      basename(outdir)`; multirun `sweep_dir = basename(dirname(outdir))` (e.g. `sweep-<ts>`).
- [ ] Multirun: `group = f"{base_group}-{sweep_dir}"`-style → `<experiment>-sweep-<ts>`
      (keep the experiment in the W&B group string for a findable flat namespace; document the
      on-disk `sweep-<ts>` ↔ W&B-group mapping). `run_name = run_label`.
- [ ] Single run: `group = base_group`, `run_name = f"{experiment}-{run_label}"` (label == ts).
- [ ] `run_id = _sanitize_label(f"{experiment}_{sweep_dir}_{run_label}")` — unique per job,
      stable across a Slurm requeue (whole path is stable); keep `resume="allow"`.
- [ ] **Remove `_sweep_param_tokens()`** (subsumed by `build_run_label`); re-derive sweep
      **tags** either by splitting the label or from `HydraConfig.overrides.task` kept data-free
      (preserve the `sweep` tag + the existing `cfg.wandb.tags`).
- [ ] Update the `derive_run_identity`/`init_wandb` docstrings to the new scheme.
- [ ] Unit-test `derive_run_identity` for a sweep path and a single-run path (pure; feed fake
      `outdir`s): name==basename, group/id as specified, no data content.

## D. Adapter → `model/` (D2) — `sft_translation.py`
*Depends on nothing in A–C; small and isolated.*

- [ ] Change `model.save_pretrained(outdir + "_b")` → `model.save_pretrained(os.path.join(outdir,
      "model"))` so the adapter nests inside the run dir.
- [ ] Grep confirms no in-repo consumer hardcodes the old `..._b` Hydra path
      (`inf_translation.py` takes `-p/--peft` as an argument; legacy `egs/mmtc/fr-en/inf.sh`
      uses an unrelated pre-Hydra path). Note the new `<run_dir>/model` path in docs (group E).

## E. Docs, decisions & validation handoff
*Depends on all above.*

- [ ] README outputs/sweep section: document the new nested layout, that the run-dir basename ==
      the W&B run name, the `<run_dir>/model` adapter path, and the `2e-4`→`0.0002` numeric
      normalization (R10). Show the `run_sacrebleu.py` invocation against `<run_dir>/eval.pred.trg`.
- [ ] Append a `decisions.md` entry: aligned output↔W&B naming via the `hfmt_runlabel` resolver;
      nested `sweep-<ts>/<run_label>` layout; adapter as `model/`; forward-only (no migration).
- [ ] Update `CLAUDE.md` project-structure notes if the `outputs/` description needs it
      (nested run dirs; `model/` adapter).
- [ ] Write `features/03_output_wandb_align/validation.md`: off-cluster results (resolver probe,
      unit tests, `--cfg hydra` layout) + an **on-cluster smoke checklist** — a 2-point sweep
      `train.seed=37,42 train.max_steps=20` → dirs `…/sweep-<ts>/seed-37/` & `…/seed-42/`, each
      with `model/` + `eval.pred.trg`, two non-clobbering W&B runs named `seed-37`/`seed-42` in
      group `<exp>-sweep-<ts>`, and confirmation that **submitit** shares one `sweep-<ts>` (R1/R3).

---

### Out of scope (this feature)
- **Migrating existing `outputs/`** dirs (`N_<ts>`/`_b`) into the new scheme — forward-only
  (Q2). A rename helper can be added later if wanted.
- **`analysis/run_sacrebleu.py` convenience** (auto-locate `eval.pred.trg` from a run dir) — Q3.
- **Key abbreviation / alias map** — explicitly rejected (Q4: descriptive, keep `lora`).
- Layout changes for the other entry points (`inf_translation`, `train_seq2seq`,
  `decode_summarization`).
- Any real GPU training/validation in this sandbox (done on-cluster by the user).
