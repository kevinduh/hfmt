# TODO — 01_wandb (W&B results export)

Actionable breakdown of `plan.md` (decisions D1–D5 locked). Scope: a standalone script that
exports W&B runs to a **single JSON file** under `analysis/` — no diagnosis/suggestions in
code (that happens via chat over the exported data). Groups are dependency-ordered; check
items off as completed.

---

## A. Docs
*The user runs this outside the sandbox in the project env, so key setup must be trivial.
No separate dependency file: `wandb` is already a project-level dependency (`install/` /
`requirements.txt`); the exporter otherwise uses only the stdlib (`json`).*

- [x] README: add an "Exporting W&B results" section — (1) set the key
      (`export WANDB_API_KEY=...` or `wandb login`), (2) run `python analysis/pull_wandb.py`
      in the project env, (3) where the file lands (`analysis/wandb_export.json`).
- [x] State in the README that the key comes from env/netrc and is **never** a CLI arg (no
      shell-history/log leakage). (The matching `--help` text lands with the script in B.)

## B. Fetch layer (`analysis/pull_wandb.py`)
*Depends on A. A plain argparse CLI using `wandb.Api()`.*

- [x] argparse CLI with: `--project` (default `hfmt`), `--entity`, `--group`,
      `--run-id` (repeatable), `--limit` (default 10), `--state` (default `finished`),
      `--out` (default `analysis/wandb_export.json`). No positional/data args; no key arg.
      `--help` also states the key comes from env/netrc.
- [x] Resolve the run set: explicit `--run-id`s if given; else query
      `api.runs(f"{entity}/{project}", filters=...)` ordered by created-at desc, take latest
      `--limit` (optionally filtered by `--group`/`--state`). → `resolve_runs()`, mock-tested.
- [x] For each run, collect: `id`, `name`, `group`, `tags`, `state`, `created_at`, `url`,
      `config`, `summary`. → `collect_run()` / `_summary_dict()`.
- [x] Pull **full** metric history via `run.scan_history()` (not sampled `history()`); keep
      every logged key, one record per step.
- [x] (validation) `py_compile` + `--help` work without `wandb` (lazy import);
      `resolve_runs` selection logic mock-tested (run-id path, filter+limit, `state=any`).

## C. Assemble & write the single JSON
*Depends on B. Pure, testable without network.*

- [x] Build the top-level object: `meta` (exported_at, project, entity, filters, num_runs) +
      `runs[]` (one entry per run with metadata + config + summary + history), per plan §3.
      → `build_export()`.
- [x] JSON-serialize safely: coerce non-serializable values (numpy scalars, timestamps) to
      plain types; write with an indent for readability. → `_json_default()` (tolist/item/
      isoformat/str), tested on numpy-like + datetime; `write_export()` uses `indent=2`.
- [x] Write to `--out`; print a one-line summary (how many runs, path, file size).
      → `write_export()` makes parent dirs; main prints runs + path + KiB.

## D. Robustness & key handling
*Depends on B/C.*

- [x] Clear, actionable errors (not stack traces) for: missing/invalid API key, no network /
      unreachable W&B, empty run set (bad filter), unknown project/entity.
      → `_friendly_wandb_error()` classifier + `_die()`; wandb-missing + empty-set handled.
- [x] Tolerate runs missing some keys (e.g. no `eval/entropy`) or empty history — export what
      exists, don't crash. → `collect_runs()` skips a bad run with a stderr warning; empty
      history → `[]`; sparse per-row dicts union naturally.
- [x] Never print or write the API key anywhere (incl. error messages / `meta`).
      → key never read/passed/printed; error text uses the exception only; `meta` has no key.

## E. Repo hygiene
- [x] Add `analysis/wandb_export.json` to `.gitignore` (generated artifact; also keeps data
      paths out of version control). (verified via `git check-ignore`)

## G. Reduction (in-script, config-cadence) — D6
*Folded into `pull_wandb.py` (not a separate summarizer) so the export is small enough to
compare many runs without overflowing context.*

- [x] Reduced-by-default export: `config` → `config.hfmt`; metrics → per-metric
      `[step, value]` series (`trajectory.eval` / `trajectory.train`); floats rounded to ~6 sig
      figs. → `reduce_run()` / `_metric_series()` / `_sigfig()`.
- [x] Config-driven cadence: point counts come from each run's `eval_steps`/`logging_steps`
      (echoed in a `steps` block); no interpolation, no fixed cap. → `_cfg_get()`.
- [x] Per-metric series (not per-step rows) so offset grids (eval/loss vs eval/bleu) don't
      fragment into half-empty rows.
- [x] `--raw` escape hatch exports full history + full config for drill-down; `meta.reduced`
      records which form the file is.
- [x] Compact serialization (`dumps_export`): indented structure but each numeric
      `[step, value]` series on one line, so `indent=2` doesn't explode the trajectories into
      thousands of whitespace lines. ~3.7x vs plain `indent=2` (one run: 38 KiB -> 10 KiB).
- [x] Tests: `_sigfig`, `_metric_series` (offset + omit-missing), `reduce_run`
      (shape + config fallback), `build_export` reduced flag, `dumps_export` (series inline +
      round-trip). Measured ~6.7x smaller (vs raw) + ~3.7x (vs indent=2) on the real export.

## F. Validation
*What's possible in-sandbox vs. what the user runs outside.*

- [x] In-sandbox (no network): `python analysis/pull_wandb.py --help` works; `py_compile`
      passes; `analysis/test_pull_wandb.py` (14 tests, pytest-compatible + standalone) covers
      fetch selection, robustness, assemble/serialize, and reduction — all pass.
- [ ] Outside sandbox (user): set key, run `python analysis/pull_wandb.py` against real `hfmt`
      runs; confirm `analysis/wandb_export.json` holds the reduced trajectories (or full with
      `--raw`).

---

### Out of scope (this feature)
- Diagnosis, scoring, or codebase/config suggestions in code — done via chat over the export.
- Rendered reports / dashboards; per-run file layout (chose single file, D5).
- Hydra/W&B sweep work (separate future request).
