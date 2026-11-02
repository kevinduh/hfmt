#!/usr/bin/env python
"""Status of SFT run folders: which finished, which are queued, which need a rerun, and the command for each.

    python tools/run_status.py egs/mmtc/fr-en/outputs                  # summary + every unfinished run
    python tools/run_status.py egs/mmtc/fr-en/outputs --all            # also list the finished runs
    python tools/run_status.py egs/mmtc/fr-en/outputs --slurm exclude=node07 --slurm gres=gpu:a100:1

Read-only: it prints the commands, you review and run them (from the repo root, on a login node). A run
folder is any folder with a run_config.yaml (written by hfmt/grid.py). Its state, from its files and Slurm:
  running/pending  in squeue
  finished         model/, and scores.json if the run has test sets
  no_test_scores   model/ but no scores.json: training finished, test decoding didn't -> launch.sh inf
  interrupted      a complete checkpoint-N (trainer_state.json), no model/ -> launch.sh sft <run>/run_config.yaml
                   (resume; also after a crash midway: check REASON first, e.g. OOM -> --slurm gres=...)
  bad_checkpoint   the last checkpoint-N was cut off while saving -> delete it, then run this tool again
  failed           a log but no checkpoint: died before its first save -> sbatch egs/job.sh sft <run>
  not_started      no log, no checkpoint, not queued (e.g. cancelled while pending) -> same as failed
--slurm KEY=VALUE replaces that slurm setting of the saved config in the commands (partition, gres, time,
mem_gb, cpus, constraint, exclude). Without squeue/sacct (or with --no-slurm), queued jobs aren't detected.

Logs hold example text (CLAUDE.md): only the end of each log is read, and only matched error lines are
printed (truncated). A queued resume of a grid folder (job name = run_name) shows as interrupted until it
starts.
"""

import argparse
import getpass
import json
import os
import re
import shlex
import subprocess
import sys
from collections import Counter, defaultdict

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {"model", "model_merged", "wandb", "inf", "base_ckpt_inf"}   # inside run folders; not searched
LOG_RE = re.compile(r"^slurm-(\d+)(?:_(\d+))?\.out$")
CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
# A Python exception's last line (e.g. "torch.AcceleratorError: CUDA error: ...") or slurmstepd's
# (time limit, cancellation, oom-kill). Hydra's "Error executing job with overrides: []" doesn't match.
ERROR_RE = re.compile(r"^([\w.]+(Error|Exception)(:.*)?|slurmstepd: error:.*)$")
PROGRESS_RE = re.compile(r"(\d+)/(\d+) \[")                     # tqdm: " 45%|####  | 180/400 [..."
TAIL_BYTES = 256 * 1024
ERROR_CHARS = 160
SBATCH_FLAGS = {"partition": "--partition={}", "gres": "--gres={}", "time": "--time={}",
                "mem_gb": "--mem={}G", "cpus": "--cpus-per-task={}", "constraint": "--constraint={}",
                "exclude": "--exclude={}"}
QUEUE_STATES = {"RUNNING": "running", "COMPLETING": "running", "CONFIGURING": "running",
                "PENDING": "pending", "REQUEUED": "pending"}
STATE_ORDER = ["running", "pending", "finished", "no_test_scores", "interrupted", "bad_checkpoint",
               "failed", "not_started"]
FIX_TITLES = {
    "interrupted": "resume from the last complete checkpoint (launch.sh sft <run>/run_config.yaml)",
    "no_test_scores": "decode the test sets of the trained model (written to <run>/inf/<inf_name>/)",
    "failed": "rerun from the saved run_config.yaml, in the same folder",
    "not_started": "run from the saved run_config.yaml, in the same folder",
    "bad_checkpoint": "a checkpoint was cut off while saving (resume refuses it)",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Status of SFT run folders, and the command to fix each "
                                            "unfinished one (prints only).")
    p.add_argument("dirs", nargs="+", help="folders to search for run folders, e.g. egs/mmtc/fr-en/outputs")
    p.add_argument("--all", action="store_true", help="also list the finished runs")
    p.add_argument("--slurm", action="append", default=[], metavar="KEY=VALUE",
                   help="slurm setting for the printed commands, replacing the saved one, e.g. "
                        "exclude=node07 or gres=gpu:a100:1 (repeatable)")
    p.add_argument("--no-slurm", action="store_true", help="don't call squeue/sacct")
    args = p.parse_args(argv)
    overrides = []
    for item in args.slurm:
        key, sep, value = item.partition("=")
        if not sep or key not in SBATCH_FLAGS:
            p.error(f"--slurm {item!r}: expected KEY=VALUE with KEY one of {', '.join(SBATCH_FLAGS)}")
        overrides.append((key, value))
    args.slurm = overrides
    return args


# --- one run folder --------------------------------------------------------------

def find_run_folders(roots):
    """Absolute paths of every folder below `roots` that holds a run_config.yaml, sorted."""
    found = []
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not CHECKPOINT_RE.match(d))
            if "run_config.yaml" in filenames:
                found.append(os.path.abspath(dirpath))
    return sorted(found)


def inspect_folder(path):
    """What a run folder's files say: config basics, model/, scores, checkpoints, newest log."""
    info = {"path": path, "config_error": None}
    try:
        with open(os.path.join(path, "run_config.yaml")) as f:
            cfg = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as e:
        cfg, info["config_error"] = {}, str(e).splitlines()[0]
    train = cfg.get("train") or {}
    test = (cfg.get("data") or {}).get("test") or {}
    names = os.listdir(path)
    checkpoints = sorted((int(m.group(1)), os.path.isfile(os.path.join(path, n, "trainer_state.json")))
                         for n in names if (m := CHECKPOINT_RE.match(n)) and os.path.isdir(os.path.join(path, n)))
    complete = [step for step, ok in checkpoints if ok]
    logs = sorted((os.path.getmtime(os.path.join(path, n)), n) for n in names
                  if LOG_RE.match(n) or n == "local.log")
    newest = logs[-1][1] if logs else None
    tail = _tail(os.path.join(path, newest)) if newest else ""
    # Written by training's test decoding, or later by `launch.sh inf <run>/run_config.yaml`.
    scores = _scores(os.path.join(path, "scores.json"))
    if scores is None:
        scores = _scores(os.path.join(path, "inf", str(cfg.get("inf_name") or "test"), "scores.json"))
    info.update(
        run_name=cfg.get("run_name") or os.path.basename(path),
        max_steps=train.get("max_steps"),
        metric=train.get("metric_for_best_model"),
        has_test=bool(test.get("src")),
        slurm=cfg.get("slurm") or {},
        model=os.path.isdir(os.path.join(path, "model")),
        scores=scores,
        checkpoints=checkpoints,
        # A finished run's final trainer_state.json, which outlives delete_checkpoints.sh; else the
        # newest complete checkpoint's.
        best=_best(path) or (_best(os.path.join(path, f"checkpoint-{max(complete)}")) if complete else None),
        job_ids=[_job_id(n) for _, n in logs if LOG_RE.match(n)],   # oldest -> newest
        log=newest,
        error=_last_error(tail),
        progress=_progress(tail, train.get("max_steps")),
    )
    return info


def _job_id(log_name):
    """slurm-123_4.out -> "123_4" (array task), slurm-123.out -> "123"."""
    m = LOG_RE.match(log_name)
    return f"{m.group(1)}_{m.group(2)}" if m.group(2) else m.group(1)


def _tail(path):
    try:
        with open(path, "rb") as f:
            f.seek(max(0, os.path.getsize(path) - TAIL_BYTES))
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _last_error(text):
    """The last exception / slurmstepd error line in `text`, truncated; None if there's none."""
    found = None
    for line in re.split(r"[\r\n]+", text):
        if ERROR_RE.match(line.strip()):
            found = line.strip()
    if found and len(found) > ERROR_CHARS:
        found = found[:ERROR_CHARS - 3] + "..."
    return found


def _progress(text, max_steps):
    """The last training step tqdm reported (a "<step>/<max_steps> [" bar), or None."""
    steps = [int(a) for a, b in PROGRESS_RE.findall(text) if max_steps and int(b) == max_steps]
    return steps[-1] if steps else None


def _scores(path):
    """{test set: BLEU} from scores.json ({} for sets without a reference); None if there's no file."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return {name: s["bleu"] for name, s in data.items() if isinstance(s, dict) and "bleu" in s}


def _best(folder):
    """{"value", "step", "global_step"} of the best evaluation so far, from the trainer_state.json in
    a checkpoint (or, after training, the run folder); None if there's none."""
    try:
        with open(os.path.join(folder, "trainer_state.json")) as f:
            state = json.load(f)
    except (OSError, ValueError):
        return None
    best_dir = state.get("best_model_checkpoint") or ""
    m = CHECKPOINT_RE.match(os.path.basename(best_dir.rstrip("/")))
    return {"value": state.get("best_metric"), "step": int(m.group(1)) if m else None,
            "global_step": state.get("global_step")}


# --- Slurm -------------------------------------------------------------------

def _run(argv):
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=120, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def parse_squeue(text):
    """Rows of `squeue -h -r -o %i|%j|%T|%N|%K`, one per job or array task."""
    rows = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 5:
            rows.append(dict(zip(("job", "name", "state", "node", "task"), parts)))
    return rows


def parse_sacct(text):
    """`sacct -n -P -X -o JobID,State,NodeList,Elapsed` -> {job id: {"state", "node", "elapsed"}}."""
    jobs = {}
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) == 4:
            job, state, node, elapsed = parts
            jobs[job] = {"state": state.split()[0] if state else "", "node": node, "elapsed": elapsed}
    return jobs


def queued_jobs():
    """Your queued and running jobs (parse_squeue rows), or None if squeue isn't available."""
    text = _run(["squeue", "-h", "-r", "-u", getpass.getuser(), "-o", "%i|%j|%T|%N|%K"])
    return None if text is None else parse_squeue(text)


def finished_jobs(job_ids):
    """sacct records of these jobs ({} if sacct isn't available)."""
    jobs, ids = {}, sorted(set(job_ids))
    for i in range(0, len(ids), 100):
        text = _run(["sacct", "-n", "-P", "-X", "-S", "2020-01-01", "-o", "JobID,State,NodeList,Elapsed",
                     "-j", ",".join(ids[i:i + 100])])
        jobs.update(parse_sacct(text or ""))
    return jobs


def _folder_index(folder_name, run_name):
    """3 for "<run_name>-3" / "<run_name>-03" (a grid job's folder), else None."""
    suffix = folder_name[len(run_name) + 1:] if folder_name.startswith(run_name + "-") else ""
    return int(suffix) if suffix.isdigit() else None


def queue_row(info, queue):
    """The squeue row of this folder's job, if it's queued or running. By the job id in the folder's log
    names; else, for a job that hasn't started (no log yet), by name: an array task of a grid launch has
    name = run_name and task = folder index; a single job has name = run_name or the folder name (a rerun
    from this tool's sbatch command)."""
    ids = set(info["job_ids"])
    for row in queue:
        if row["job"] in ids:
            return row
    folder = os.path.basename(info["path"])
    index = _folder_index(folder, info["run_name"])
    for row in queue:
        if row["task"].isdigit():
            if row["name"] == info["run_name"] and int(row["task"]) == index:
                return row
        elif row["name"] == folder:
            return row
    return None


# --- state and fix -----------------------------------------------------------

def classify(info, row=None):
    """The folder's state (see the module docstring)."""
    if row:
        return QUEUE_STATES.get(row["state"], row["state"].lower())
    if info["model"]:
        return "finished" if info["scores"] is not None or not info["has_test"] else "no_test_scores"
    if info["checkpoints"]:
        return "interrupted" if info["checkpoints"][-1][1] else "bad_checkpoint"
    return "failed" if info["log"] else "not_started"


def fix_command(state, info, overrides=()):
    """The command for an unfinished folder (None if there's nothing to run)."""
    rc = shlex.quote(os.path.join(info["path"], "run_config.yaml"))
    launch_overrides = "".join(" " + shlex.quote(f"slurm.{k}={v}") for k, v in overrides)
    if state == "interrupted":
        return f"bash egs/launch.sh sft {rc}{launch_overrides}"
    if state == "no_test_scores":
        return f"bash egs/launch.sh inf {rc}{launch_overrides}"
    if state in ("failed", "not_started"):
        slurm = {**info["slurm"], **dict(overrides)}
        flags = [flag.format(slurm[k]) for k, flag in SBATCH_FLAGS.items() if slurm.get(k) not in (None, "")]
        flags += [f"--job-name={os.path.basename(info['path'])}",
                  f"--output={os.path.join(info['path'], 'slurm-%j.out')}"]
        return "sbatch " + " ".join(shlex.quote(f) for f in flags) + f" egs/job.sh sft {shlex.quote(info['path'])}"
    if state == "bad_checkpoint":
        step = info["checkpoints"][-1][0]
        return f"# delete {shlex.quote(os.path.join(info['path'], f'checkpoint-{step}'))}, then run this tool again"
    return None


# --- report ------------------------------------------------------------------

def _table(rows):
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(str(c).ljust(w) for c, w in zip(r, widths)).rstrip() for r in rows)


def _row(info, state, row, job):
    last = max([s for s in (info["progress"], *(step for step, ok in info["checkpoints"] if ok)) if s is not None],
               default=None)
    best = info["best"]
    metric = (info["metric"] or "").removeprefix("eval_")
    best_text = f"{metric} {best['value']:.2f}@{best['step']}" if best and best["value"] is not None else "-"
    test = "/".join(f"{v:.2f}" for v in (info["scores"] or {}).values()) or "-"
    if row:
        slurm = f"{row['job']} {row['node'] or '-'}"
    elif job:
        slurm = f"{info['job_ids'][-1]} {job['node'] or '-'} {job['state']} {job['elapsed']}"
    else:
        slurm = info["job_ids"][-1] if info["job_ids"] else "-"
    reason = info["config_error"] or (info["error"] if state not in ("finished", "running", "pending") else None)
    return [os.path.relpath(info["path"]), state, f"{last if last is not None else '-'}/{info['max_steps'] or '-'}",
            best_text, test, slurm, reason or ""]


def report(infos, states, rows, jobs, overrides=(), show_all=False, slurm_note=""):
    """The printed report: per run_name counts, unfinished (or all) folders, failure nodes, fix commands."""
    lines = [f"{len(infos)} run folder(s){slurm_note}", ""]
    by_name = defaultdict(Counter)
    for info, state in zip(infos, states):
        by_name[info["run_name"]][state] += 1
    width = max(len(n) for n in by_name)
    for name in sorted(by_name):
        counts = by_name[name]
        order = sorted(counts, key=lambda s: STATE_ORDER.index(s) if s in STATE_ORDER else len(STATE_ORDER))
        parts = ", ".join(f"{counts[s]} {s}" for s in order)
        lines.append(f"  {name.ljust(width)}  {sum(counts.values()):>3}: {parts}")

    listed = [(i, s, r, j) for i, s, r, j in zip(infos, states, rows, jobs) if show_all or s != "finished"]
    if listed:
        header = ["FOLDER", "STATE", "STEP", "BEST DEV", "TEST BLEU", "SLURM JOB / NODE / STATE", "REASON"]
        lines += ["", _table([header] + [_row(*x) for x in listed])]

    failed_nodes = Counter(j["node"] for s, j in zip(states, jobs)
                           if j and j["node"] and s != "finished" and j["state"] in ("FAILED", "NODE_FAIL"))
    if failed_nodes:
        lines += ["", "Nodes of failed jobs: " + ", ".join(f"{n} ({c})" for n, c in failed_nodes.most_common())
                  + "  (several on one node: maybe a bad node -> --slurm exclude=<node>)"]

    fixes = defaultdict(list)
    for info, state in zip(infos, states):
        command = fix_command(state, info, overrides)
        if command:
            fixes[state].append(command)
    if fixes:
        lines += ["", "To fix (review first; run on a login node):",
                  f"cd {shlex.quote(ROOT)} && export HFMT_ROOT={shlex.quote(ROOT)}"]
        for state in [s for s in STATE_ORDER if s in fixes]:
            lines += ["", f"# {state} ({len(fixes[state])}): {FIX_TITLES[state]}"] + fixes[state]
    return "\n".join(lines)


def main(argv=None):
    args = parse_args(argv)
    folders = find_run_folders(args.dirs)
    if not folders:
        sys.exit(f"run_status: no run folders (run_config.yaml) under {' '.join(args.dirs)}")
    infos = [inspect_folder(f) for f in folders]

    queue = None if args.no_slurm else queued_jobs()
    rows = [queue_row(info, queue) if queue else None for info in infos]
    states = [classify(info, row) for info, row in zip(infos, rows)]
    wanted = [info["job_ids"][-1] for info, state in zip(infos, states)
              if info["job_ids"] and state not in ("finished", "running", "pending")]
    sacct = finished_jobs(wanted) if queue is not None and wanted else {}
    jobs = [sacct.get(info["job_ids"][-1]) if info["job_ids"] else None for info in infos]
    note = "" if queue is not None else ("  (Slurm not queried)" if args.no_slurm else "  (squeue not available: queued jobs not detected)")
    print(report(infos, states, rows, jobs, args.slurm, args.all, note))


if __name__ == "__main__":
    main()
