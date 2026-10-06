# Plan — 01_wandb (W&B results export)

**Status:** finalized — see `todo.md`. See `request.md`.

## Goal

An easy-to-run script under `analysis/` that **pulls run data from Weights & Biases into the
`analysis/` directory** — nothing more. It exports each run's metrics history + config so the
data can be reviewed later. The actual review and "what to change" suggestions happen
**conversationally with Claude Code using the exported data**, not inside this code.

## Decisions (locked from narrowing)
- **D1 — Export only.** This feature just fetches and writes W&B data to disk. No diagnosis,
  suggestion, or report logic in the code. Analysis/next-steps are done by chatting with
  Claude Code over the exported files.
- **D2 — Runs outside the sandbox.** The user runs it on a machine with outbound W&B access
  (laptop/login node), in the project environment (which already includes `wandb`). It must be
  easy to run, with an easy, clearly documented way to set the W&B API key.
- **D3 — Flexible filters, default latest-N.** Support `--run-id`, `--group`,
  `--project`/`--entity`; default to the most recent N finished runs in the `hfmt` project.
- **D4 — Multiple runs (cross-run ready).** Pull a set of runs into one file so cross-run
  comparison is possible in the later conversation.
- **D5 — Single combined export file.** Write all runs to one JSON file, easy to hand to a
  Claude Code chat. Defaults: `--limit 10`, output `analysis/wandb_export.json` (gitignored).
- **D6 — Reduced by default, in the pull script (no separate summarizer).** To keep the file
  small enough to compare 10-20 runs without overflowing context, the export is reduced inside
  `pull_wandb.py`: keep `config.hfmt` (drop the ~170-key HF dump) and emit metrics as
  per-metric `[step, value]` series (`eval` + `train`). The number of points comes from each
  run's **own config cadence** (`eval_steps` / `logging_steps`) -- no interpolation, no fixed
  cap. Per-metric series (not per-step rows) so metrics logged on offset steps (eval/loss from
  the Trainer vs eval/bleu from the callback) don't fragment. Floats rounded to ~6 sig figs.
  `--raw` exports everything (full history + full config) for drill-down. (~6-7x smaller:
  ~10 KiB/run, so 20 runs ~50k tokens.)
  Serialization: indented structure but each numeric `[step, value]` series printed on one
  line (`dumps_export`), so pretty-printing doesn't explode the dense trajectories into
  thousands of whitespace lines (that alone is ~3.7x vs plain `indent=2`).

## Guiding constraints
- **Confidentiality (CLAUDE.md):** our runs contain only metrics + data-free config (paths,
  hyperparameters), so exporting them is safe. The script must never print/commit the API
  key, and must not fetch or write dataset content.
- **Standalone:** matches `analysis/run_sacrebleu.py` style — a plain argparse script, not a
  Hydra app. Runs in the project environment (`wandb` is already a project dependency).

## Proposed approach

### 1. Where it lives
`analysis/pull_wandb.py` — a standalone argparse CLI that fetches runs and writes them to a
single JSON file (default `analysis/wandb_export.json`, overridable with `--out`).

### 2. What it pulls (per run)
Via `wandb.Api()`:
- `config` — hyperparameters + `run_dir` (the `hfmt` block we log).
- `summary` — final/best scalar values.
- metadata — run id, name, group, tags, state, created-at, url.
- **full metric history** via `run.scan_history()` (not sampled `history()`), covering every
  logged key (`eval/bleu`, `eval/chrf`, `eval/ter`, `eval/loss`, `train/loss`,
  `eval/entropy`, `system/gpu.*`, `train/global_step`, ...).

### 3. Export layout — single JSON file (reduced by default, D6)
One file (`analysis/wandb_export.json`); `runs` is a list so cross-run comparison is trivial.
```json
{
  "meta": { "exported_at": "...", "project": "hfmt", "entity": "...",
            "filters": {...}, "reduced": true, "num_runs": N },
  "runs": [
    {
      "id": "...", "name": "...", "group": "...", "tags": [...],
      "state": "finished", "created_at": "...", "url": "...",
      "config": { ...config.hfmt knobs... },
      "steps":  { "max_step": 800, "eval_steps": 50, "logging_steps": 10,
                  "num_eval_steps": ..., "num_train_steps": ... },
      "trajectory": {
        "eval":  { "loss": [[50,0.49],[100,0.47],...], "bleu": [[60,67.5],...], ... },
        "train": { "loss": [[10,1.21],...], "grad_norm": [...], ... }
      }
    }
  ]
}
```
With `--raw`, each run instead carries full `config` + `summary` + `history` (every logged
step, all keys). This is a generated artifact → `analysis/wandb_export.json` is gitignored.

### 4. Filters / CLI
`--project` (default `hfmt`), `--entity`, `--group`, `--run-id` (repeatable), `--limit N`
(default latest-N, e.g. 10), `--state finished` (default), `--out <dir>`. No positional data
args. The key is **never** a CLI arg.

### 5. API key + instructions
- Auth via `WANDB_API_KEY` env var or `wandb login` (`~/.netrc`); `wandb.Api()` picks it up.
  Document both; `--help` states it.
- No extra dependency file: `wandb` is already part of the project environment
  (`install/` / `requirements.txt`). The script just runs in that env.
- README section "Exporting W&B results" with: set key -> run -> where data lands.

## Phased breakdown (-> todo.md once confirmed)
1. README "Exporting W&B results" section (set key + run + where data lands).
2. `analysis/pull_wandb.py`: argparse CLI, `wandb.Api()` fetch with filters/latest-N.
3. Writer: assemble all runs into the single JSON file (`meta` + `runs[]`).
4. Robustness: missing keys, empty history, no-network/auth errors with clear messages.
5. `.gitignore`: add `analysis/wandb_export.json`.
6. Validation (outside sandbox, by user): pull a few real `hfmt` runs; confirm the file.

## Explicitly out of scope (this feature)
- Any diagnosis, scoring, or codebase/config suggestions in code — done via chat with Claude
  over the exported data.
- Rendered analysis reports / dashboards.
- The Hydra/W&B **sweep** work (separate future request).

## Risks
- **R1 — Network / auth.** `wandb.Api()` needs outbound access + a valid key; that's why it
  runs outside the sandbox. Fail with a clear, actionable message (how to set the key / check
  access) rather than a stack trace.
- **R2 — History volume / sampling.** `scan_history` is full but slower and paginates; large
  or many runs mean more API calls. Acceptable for a handful of runs; note it for big pulls.
- **R3 — Metric-key variability.** Older/other runs may lack some keys (e.g. `eval/entropy`);
  write whatever exists, union columns in `history.csv`, don't crash.
- **R4 — Key handling.** Never accept the key as an arg or echo it; env/netrc only.
- **R5 — Export hygiene.** Exported data is a generated artifact; gitignore it. (Safe content
  per CLAUDE.md, but absolute data *paths* in config will be visible in the export — expected.)
- **R6 — Single-file size.** Full history for many runs in one JSON can get large; fine for
  the default handful, but `--limit` keeps it bounded.

## Open questions

All resolved — folded into Decisions (D5: single JSON file, `--limit 10` default, output
`analysis/wandb_export.json` gitignored).
