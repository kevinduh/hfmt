#!/usr/bin/env python
"""Export Weights & Biases runs to a single JSON file for later review.

Pulls finished runs (metrics history + config) from W&B into one JSON file so the
results can be analyzed afterwards (e.g. by handing the file to Claude Code). This
script only reads aggregate metrics + config -- never dataset content.

By default the export is *reduced* to keep it small enough to compare many runs at
once: each run keeps its `config.hfmt` knobs (not the full HF config dump) and its
metrics as per-metric [step, value] series at the run's own logging cadence
(eval_steps / logging_steps) -- no interpolation, no fixed point cap. Pass --raw to
export everything (full history + full config) for drill-down.

The W&B API key is read from the environment (WANDB_API_KEY) or ~/.netrc (via
`wandb login`). It is NEVER accepted as a command-line argument, so it can't leak
into shell history or logs.

Requires `wandb`, which is part of the project environment (see install/). Run it
OUTSIDE the sandbox, where W&B is reachable:
    export WANDB_API_KEY=...        # or: wandb login
    python analysis/pull_wandb.py                      # latest 10 finished 'hfmt' runs
    python analysis/pull_wandb.py --group mmtc_fr-en_sft1 --limit 20
    python analysis/pull_wandb.py --run-id abc123 --run-id def456
"""

import argparse
import datetime
import json
import math
import os
import sys

# Curated metric columns kept in the reduced trajectory (prefix stripped in output).
EVAL_METRICS = ["eval/loss", "eval/bleu", "eval/chrf", "eval/ter", "eval/entropy",
                "eval/mean_token_accuracy"]
TRAIN_METRICS = ["train/loss", "train/entropy", "train/grad_norm", "train/learning_rate",
                 "train/mean_token_accuracy"]
STEP_KEY = "train/global_step"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Export W&B runs (metrics + config) to a single JSON file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="The W&B API key is read from WANDB_API_KEY or ~/.netrc "
               "(run `wandb login`); it is never passed as an argument.",
    )
    p.add_argument("--project", default="hfmt", help="W&B project name")
    p.add_argument("--entity", default=None,
                   help="W&B entity (team/user); defaults to your W&B default entity")
    p.add_argument("--group", default=None, help="only runs in this W&B group")
    p.add_argument("--run-id", dest="run_ids", action="append", default=[], metavar="ID",
                   help="specific run id (repeatable); overrides --group/--limit/--state")
    p.add_argument("--limit", type=int, default=10,
                   help="max number of most-recent runs to export")
    p.add_argument("--state", default="finished",
                   help="only runs in this state (e.g. finished); use 'any' for no filter")
    p.add_argument("--out", default="analysis/wandb_export.json", help="output JSON file")
    p.add_argument("--raw", action="store_true",
                   help="export full unreduced runs (all metrics, every logged step, full "
                        "config) instead of the reduced, config-cadence trajectory")
    return p.parse_args(argv)


def resolve_runs(api, entity, project, group=None, run_ids=None, state="finished", limit=10):
    """Return a list of wandb Run objects for the requested selection.

    Explicit --run-id selections win and ignore the group/state/limit filters.
    Otherwise query the project's runs (newest first) and take the latest `limit`.
    """
    path = f"{entity}/{project}" if entity else project
    if run_ids:
        return [api.run(f"{path}/{rid}") for rid in run_ids]

    filters = {}
    if group:
        filters["group"] = group
    if state and state.lower() != "any":
        filters["state"] = state

    runs = api.runs(path, filters=filters or None, order="-created_at")
    selected = []
    for i, run in enumerate(runs):  # api.runs is lazy/paginated
        if i >= limit:
            break
        selected.append(run)
    return selected


def collect_runs(runs):
    """Collect each run, skipping (with a warning) any that fail so one bad run
    doesn't abort the whole export."""
    collected = []
    for run in runs:
        try:
            collected.append(collect_run(run))
        except Exception as e:  # noqa: BLE001 - keep going on a single bad run
            print(f"warning: skipping run {getattr(run, 'id', '?')}: {e}", file=sys.stderr)
    return collected


def collect_run(run):
    """Collect one run's metadata, config, summary, and full metric history.

    Uses scan_history() for the *full* (unsampled) per-step history. Missing keys /
    empty history are fine -- each history row is its own dict, so columns union
    naturally across runs and absent metrics are simply absent.
    """
    return {
        "id": run.id,
        "name": run.name,
        "group": run.group,
        "tags": list(run.tags or []),
        "state": run.state,
        "created_at": str(run.created_at),
        "url": run.url,
        "config": dict(run.config),
        "summary": _summary_dict(run.summary),
        "history": list(run.scan_history()),
    }


def _summary_dict(summary):
    """Best-effort plain-dict view of a run's summary object."""
    data = getattr(summary, "_json_dict", None)
    if data is not None:
        return dict(data)
    try:
        return dict(summary)
    except Exception:
        return {}


def _sigfig(x, n=6):
    """Round a float to n significant figures (keeps files small, lossless for analysis)."""
    if isinstance(x, bool) or not isinstance(x, float):
        return x
    if x == 0 or not math.isfinite(x):
        return x
    return round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))


def _metric_series(history, metric_keys):
    """Per-metric [step, value] series (prefix stripped, floats rounded).

    Each metric is its own curve at the steps it was actually logged, so metrics
    logged on offset steps (e.g. eval/loss from the Trainer vs eval/bleu from the
    callback) don't fragment into half-empty rows. Keeps every logged point at the
    run's native (config-driven) cadence -- no interpolation, no fixed cap. Metrics
    that were never logged are omitted.
    """
    out = {}
    for key in metric_keys:
        pts = [[rec.get(STEP_KEY), _sigfig(rec[key])]
               for rec in history if rec.get(key) is not None]
        if pts:
            pts.sort(key=lambda p: (p[0] is None, p[0]))
            out[key.split("/", 1)[1]] = pts
    return out


def _distinct_steps(series):
    return len({p[0] for pts in series.values() for p in pts})


def _cfg_get(config, key):
    """Read a training knob from the HF config or, failing that, our hfmt block."""
    if key in config:
        return config[key]
    return (config.get("hfmt", {}).get("train", {}) or {}).get(key)


def reduce_run(run):
    """Reduce one collected run to config.hfmt + two config-cadence trajectories.

    The number of eval/train points comes from each run's own logging cadence
    (eval_steps / logging_steps), not a one-size-fits-all grid.
    """
    config = run.get("config", {}) or {}
    hfmt = config.get("hfmt", config)  # prefer our knobs; fall back to full config
    history = run.get("history", []) or []
    eval_series = _metric_series(history, EVAL_METRICS)
    train_series = _metric_series(history, TRAIN_METRICS)
    return {
        "id": run.get("id"),
        "name": run.get("name"),
        "group": run.get("group"),
        "tags": run.get("tags", []),
        "state": run.get("state"),
        "created_at": run.get("created_at"),
        "url": run.get("url"),
        "config": hfmt,
        "steps": {
            "max_step": _cfg_get(config, "max_steps"),
            "eval_steps": _cfg_get(config, "eval_steps"),
            "logging_steps": _cfg_get(config, "logging_steps"),
            "num_eval_steps": _distinct_steps(eval_series),
            "num_train_steps": _distinct_steps(train_series),
        },
        "trajectory": {"eval": eval_series, "train": train_series},
    }


def build_export(collected, *, project, entity, filters, reduced=True):
    """Assemble the single export object: meta + the list of collected runs."""
    return {
        "meta": {
            "exported_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "project": project,
            "entity": entity,
            "filters": filters,
            "reduced": reduced,
            "num_runs": len(collected),
        },
        "runs": collected,
    }


def _json_default(o):
    """Coerce values json can't handle (numpy scalars/arrays, datetimes) to plain types."""
    if isinstance(o, (datetime.date, datetime.datetime)):
        return o.isoformat()
    for attr in ("tolist", "item"):  # numpy ndarray -> list, numpy scalar -> python scalar
        m = getattr(o, attr, None)
        if callable(m):
            try:
                return m()
            except Exception:
                pass
    return str(o)


def _isnum(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def dumps_export(export, indent=2):
    """JSON with an indented structure but each numeric [step, value] series printed on a
    single line, so the dense trajectories don't explode into thousands of whitespace lines
    (keeps the file small for handing to a chat while staying scannable)."""
    marks = {}

    def walk(o):
        if isinstance(o, list):
            if o and all(isinstance(e, (list, tuple)) and len(e) == 2
                         and _isnum(e[0]) and _isnum(e[1]) for e in o):
                token = f"\u0000S{len(marks)}\u0000"  # null-char sentinel: never in real data
                marks[token] = json.dumps(o, separators=(",", ":"), default=_json_default)
                return token
            return [walk(e) for e in o]
        if isinstance(o, dict):
            return {k: walk(v) for k, v in o.items()}
        return o

    text = json.dumps(walk(export), indent=indent, default=_json_default)
    for token, compact in marks.items():
        text = text.replace(json.dumps(token), compact)
    return text


def write_export(export, out_path):
    """Write the export object to out_path; return the file size."""
    parent = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(parent, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(dumps_export(export) + "\n")
    return os.path.getsize(out_path)


def _die(msg, code=1):
    """Print a clean error to stderr and exit (no traceback)."""
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(code)


def _friendly_wandb_error(e):
    """Map a W&B/requests exception to an actionable, key-safe message."""
    name = type(e).__name__.lower()
    msg = str(e).lower()
    if "auth" in name or "permission" in name or "api key" in msg or "401" in msg or "403" in msg:
        return ("W&B authentication/permission failed. Set your key with "
                "`export WANDB_API_KEY=...` or run `wandb login`, and check you have "
                "access to this entity/project.")
    if ("conn" in name or "network" in name or "timeout" in name
            or "timed out" in msg or "resolve" in msg or "unreachable" in msg):
        return ("Could not reach W&B (network error). Run this where api.wandb.ai is "
                "reachable (i.e. outside the sandbox).")
    # Fallback: include the exception's own text, which does not contain the key.
    return f"W&B error: {e}"


def main(argv=None):
    args = parse_args(argv)
    try:
        import wandb  # lazy import: --help and py_compile work without wandb installed
    except ImportError:
        _die("wandb is not installed. Activate the project environment (see install/).")

    # The API key is read by wandb from WANDB_API_KEY / ~/.netrc. We never read,
    # pass, or print it here.
    try:
        api = wandb.Api()
        entity = args.entity or api.default_entity
    except Exception as e:  # noqa: BLE001
        _die(_friendly_wandb_error(e))

    if not entity:
        _die("Could not determine a W&B entity. Pass --entity, or run `wandb login`.")

    try:
        runs = resolve_runs(api, entity, args.project, group=args.group,
                            run_ids=args.run_ids, state=args.state, limit=args.limit)
        collected = collect_runs(runs)
    except Exception as e:  # noqa: BLE001
        _die(_friendly_wandb_error(e) + f"\n(querying '{entity}/{args.project}')")

    if not collected:
        _die(f"No runs matched (project={args.project!r}, entity={entity!r}, "
             f"group={args.group!r}, state={args.state!r}). Nothing exported.")

    if not args.raw:
        collected = [reduce_run(r) for r in collected]

    filters = {
        "group": args.group,
        "state": args.state,
        "run_ids": args.run_ids or None,
        "limit": args.limit,
    }
    export = build_export(collected, project=args.project, entity=entity, filters=filters,
                          reduced=not args.raw)
    size = write_export(export, args.out)
    print(f"Wrote {len(collected)} run(s) to {args.out} ({size / 1024:.1f} KiB)")
    return export


if __name__ == "__main__":
    main()
