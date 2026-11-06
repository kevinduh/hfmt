#!/usr/bin/env python
"""Status of SFT run folders: training and test inference, the scores of both, and the command for each
unfinished one.

    python tools/run_status.py egs/mmtc/fr-en/outputs                  # every run folder + fix commands
    python tools/run_status.py egs/mmtc/fr-en/outputs --slurm exclude=node07 --slurm gres=gpu:a100:1

Read-only: it prints the commands, you review and run them (from the repo root, on a login node). A run
folder is any folder with a run_config.yaml (written by hfmt/grid.py); folders under a complete/ folder
(where finished runs are moved) are skipped, unless complete/ itself is passed. Its state, from its files
and Slurm:
  running/pending      the training job is in squeue
  finished             model/, and inference scores (<run>/inf/<inf_name>/scores.json) if the run has test sets
  inf_running/pending  model/, and the inference job (launch.sh inf) is in squeue
  no_inference         model/ but no inference scores -> launch.sh inf <run>/run_config.yaml
                       (REASON: the inference error, if one ran and failed)
  bad_model            model/ was cut off while saving (no model/training_info.yaml, no scores) -> delete
                       it, then run this tool again (runs from before 2026-11-02 have no training_info.yaml)
  interrupted          a complete checkpoint-N (trainer_state.json), no model/ -> launch.sh sft <run>/run_config.yaml
                       (resume; also after a crash midway: check REASON first, e.g. OOM -> --slurm gres=...)
  bad_checkpoint       the last checkpoint-N was cut off while saving -> delete it, then run this tool again
  failed               a log but no checkpoint: died before its first save -> sbatch egs/job.sh sft <run>
  not_started          no log, no checkpoint, not queued (e.g. cancelled while pending) -> same as failed
Columns: TEST (train) is the test decoding at the end of training (<run>/scores.json), TEST (inf) that of
launch.sh inf (<run>/inf/<inf_name>/scores.json); both in the BEST DEV metric (bleu if that is loss), one
value per test set. MATCH checks that inference reproduces training: "yes" = equal scores and identical
test<i>.pred.trg files; "NO: ..." = what differs (counts of differing lines); "decode differs" = inference
used other decode.* or data.test settings, so the two aren't comparable.
--slurm KEY=VALUE replaces that slurm setting of the saved config in the commands (partition, gres, time,
mem_gb, cpus, constraint, exclude). Without squeue/sacct (or with --no-slurm), queued jobs aren't detected.

Checks failed: runs that finished but not as intended -- a known-bad library warning in a training log
(BAD_WARNINGS: the Trainer's checkpoint reload loaded nothing, so model/ is the last step; TRL's prompt not
a token prefix of prompt + reply) or model/training_info.yaml's model_dev_check a "MISMATCH" (the saved model's
dev BLEU differs from the dev BLEU of the step it holds). The reload warning counts only for runs without a
model_dev_check (before 2026-11-05): later full runs still log it, from the Trainer's own reload, which
load_checkpoint_weights then redoes correctly, and the dev check measures the result. See decisions.md
2026-11-04 and 2026-11-05.

Logs and predictions hold example text (CLAUDE.md): only the end of each log is read for errors, whole
logs are searched for BAD_WARNINGS, and only matched error lines (truncated) and labels are printed;
prediction files are compared, never printed.
"""

import argparse
import getpass
import itertools
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
ARCHIVE_DIRS = {"complete"}         # run folders moved there by hand when done: searched only if passed itself
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
DECODE_METRICS = ("bleu", "bleu_13a_lowercase", "chrf", "ter")   # scores.json keys (hfmt/mt_model.py score)
# Library warnings that mean a finished run trained or saved something other than intended (decisions.md
# 2026-11-04, 2026-11-05): (text in the log, label printed, whether model_dev_check supersedes it). Whole
# training logs are searched. The Trainer's reload still warns in every Qwen3.5 full run after the fix
# (sft_translation.py reloads the checkpoint again through load_checkpoint_weights), so for a run with a
# model_dev_check that measured check is what counts, not the warning.
BAD_WARNINGS = [
    ("There were missing keys in the checkpoint model loaded", "checkpoint reload loaded nothing: model/ is the last step",
     True),
    ("Mismatch between tokenized prompt", "prompt not a token prefix of prompt + reply (chat template)", False),
]
QUEUE_STATES = {"RUNNING": "running", "COMPLETING": "running", "CONFIGURING": "running",
                "PENDING": "pending", "REQUEUED": "pending"}
STATE_ORDER = ["running", "pending", "finished", "inf_running", "inf_pending", "no_inference", "bad_model",
               "interrupted", "bad_checkpoint", "failed", "not_started"]
ACTIVE_OR_DONE = {"running", "pending", "finished", "inf_running", "inf_pending"}
FIX_TITLES = {
    "no_inference": "decode the test sets of the trained model (written to <run>/inf/<inf_name>/)",
    "bad_model": "model/ was cut off while saving: training didn't finish",
    "interrupted": "resume from the last complete checkpoint (launch.sh sft <run>/run_config.yaml)",
    "failed": "rerun from the saved run_config.yaml, in the same folder",
    "not_started": "run from the saved run_config.yaml, in the same folder",
    "bad_checkpoint": "a checkpoint was cut off while saving (resume refuses it)",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Status of SFT run folders (training and test inference), and "
                                            "the command to fix each unfinished one (prints only).")
    p.add_argument("dirs", nargs="+", help="folders to search for run folders, e.g. egs/mmtc/fr-en/outputs")
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
    """Absolute paths of every folder below `roots` that holds a run_config.yaml, sorted. A `complete/`
    folder below a root is skipped (a root may itself be one)."""
    found = []
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS | ARCHIVE_DIRS and not CHECKPOINT_RE.match(d))
            if "run_config.yaml" in filenames:
                found.append(os.path.abspath(dirpath))
    return sorted(found)


def inspect_folder(path):
    """What a run folder's files say: config basics, model/, checkpoints, newest log, and the test
    decoding of training (<run>/scores.json) and of launch.sh inf (<run>/inf/<inf_name>/)."""
    info = {"path": path}
    cfg, info["config_error"] = _load_yaml(os.path.join(path, "run_config.yaml"))
    train = cfg.get("train") or {}
    test = (cfg.get("data") or {}).get("test") or {}
    names = os.listdir(path)
    checkpoints = sorted((int(m.group(1)), os.path.isfile(os.path.join(path, n, "trainer_state.json")))
                         for n in names if (m := CHECKPOINT_RE.match(n)) and os.path.isdir(os.path.join(path, n)))
    complete = [step for step, ok in checkpoints if ok]
    logs = _logs(path)
    tail = _tail(os.path.join(path, logs[-1])) if logs else ""
    dev_check = _model_dev_check(path)
    scores = _scores(os.path.join(path, "scores.json"))           # training's test decoding
    # launch.sh inf <run>/run_config.yaml: inf_config.yaml when launched, scores.json when done.
    inf_dir = os.path.join(path, "inf", str(cfg.get("inf_name") or "test"))
    inf_logs = _logs(inf_dir)
    inf_tail = _tail(os.path.join(inf_dir, inf_logs[-1])) if inf_logs else ""
    inf_config = os.path.join(inf_dir, "inf_config.yaml")
    inf_cfg = _load_yaml(inf_config)[0] if os.path.isfile(inf_config) else None
    inf_scores = _scores(os.path.join(inf_dir, "scores.json"))
    info.update(
        run_name=cfg.get("run_name") or os.path.basename(path),
        max_steps=train.get("max_steps"),
        metric=train.get("metric_for_best_model"),
        has_test=bool(test.get("src")),
        slurm=cfg.get("slurm") or {},
        model=os.path.isdir(os.path.join(path, "model")),
        # Training writes model/training_info.yaml after the weights, the tokenizer and trainer_state.json,
        # then scores.json; scores from inference also mean model/ loaded.
        model_complete=os.path.isfile(os.path.join(path, "model", "training_info.yaml"))
        or scores is not None or inf_scores is not None,
        scores=scores,
        checkpoints=checkpoints,
        # A finished run's final trainer_state.json, which outlives delete_checkpoints.sh; else the
        # newest complete checkpoint's.
        best=_best(path) or (_best(os.path.join(path, f"checkpoint-{max(complete)}")) if complete else None),
        job_ids=[_job_id(n) for n in logs if LOG_RE.match(n)],   # oldest -> newest
        log=logs[-1] if logs else None,
        error=_last_error(tail),
        progress=_progress(tail, train.get("max_steps")),
        inf_launched=inf_cfg is not None,
        inf_scores=inf_scores,
        inf_job_ids=[_job_id(n) for n in inf_logs if LOG_RE.match(n)],
        inf_log=inf_logs[-1] if inf_logs else None,
        inf_error=_last_error(inf_tail),
        match=compare_test_outputs(path, inf_dir, cfg, inf_cfg, scores, inf_scores),
        warnings=_bad_warnings(path, logs, dev_checked=dev_check is not None),
        model_check=dev_check if dev_check and dev_check.startswith("MISMATCH") else None,
    )
    return info


def _bad_warnings(folder, logs, dev_checked=False):
    """Labels of the BAD_WARNINGS found in the folder's training logs, in BAD_WARNINGS order, leaving out
    those the model/ dev check supersedes when the run has one (``dev_checked``). Whole files: the warnings
    come mid-run, long before the tail."""
    found = []
    for name in logs:
        try:
            with open(os.path.join(folder, name), "rb") as f:
                text = f.read().decode("utf-8", errors="replace")
        except OSError:
            continue
        found += [label for pattern, label, _ in BAD_WARNINGS if pattern in text and label not in found]
    return [label for _, label, superseded in BAD_WARNINGS if label in found and not (superseded and dev_checked)]


def _model_dev_check(path):
    """model/training_info.yaml's model_dev_check (hfmt/sft_translation.py check_model_dev_scores): "ok",
    "unchecked: ..." (no dev decode at that step) or "MISMATCH: ..."; None when absent (runs before
    2026-11-05, or not finished)."""
    check = _load_yaml(os.path.join(path, "model", "training_info.yaml"))[0].get("model_dev_check")
    return check if isinstance(check, str) else None


def _load_yaml(path):
    """(content, None), or ({}, the error's first line) if the file can't be read."""
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}, None
    except (OSError, yaml.YAMLError) as e:
        return {}, str(e).splitlines()[0]


def _logs(folder):
    """Names of the Slurm logs (and LOCAL=1's local.log) in `folder`, oldest first; [] if it doesn't exist."""
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    return [n for _, n in sorted((os.path.getmtime(os.path.join(folder, n)), n) for n in names
                                 if LOG_RE.match(n) or n == "local.log")]


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
    """{test set: {metric: value}} from a scores.json (no metrics for a set without a reference); None if
    there's no readable file."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return {name: {k: s[k] for k in DECODE_METRICS if k in s} for name, s in data.items() if isinstance(s, dict)}


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


def compare_test_outputs(run_dir, inf_dir, cfg, inf_cfg, scores, inf_scores):
    """MATCH: whether launch.sh inf reproduced training's test decoding (CLAUDE.md: it should, exactly).
    "yes", "decode differs", or "NO: <what differs>"; None if one of the two hasn't scored yet."""
    if scores is None or inf_scores is None:
        return None
    if inf_cfg is not None and (inf_cfg.get("decode") != cfg.get("decode")
                                or (inf_cfg.get("data") or {}).get("test") != (cfg.get("data") or {}).get("test")):
        return "decode differs"
    problems = []
    for name in list(scores) + [n for n in inf_scores if n not in scores]:
        if name not in scores or name not in inf_scores:
            problems.append(f"{name} only in {'training' if name in scores else 'inference'}")
            continue
        if any(scores[name][k] != inf_scores[name][k] for k in scores[name].keys() & inf_scores[name].keys()):
            problems.append(f"{name} scores")
        lines = _differing_lines(os.path.join(run_dir, f"{name}.pred.trg"), os.path.join(inf_dir, f"{name}.pred.trg"))
        if lines is None:
            problems.append(f"{name} no pred file")
        elif lines:
            problems.append(f"{name} {lines} lines")
    return "NO: " + ", ".join(problems) if problems else "yes"


def _differing_lines(a, b):
    """How many lines differ between prediction files `a` and `b` (a line only one has counts); None if
    either is missing. Compared as bytes and never printed: predictions are data (CLAUDE.md)."""
    try:
        with open(a, "rb") as fa, open(b, "rb") as fb:
            return sum(x != y for x, y in itertools.zip_longest(fa, fb))
    except OSError:
        return None


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


def _same_job(slurm_id, log_ids):
    """Whether a squeue/sacct job id is one of a folder's log ids. A grid task's log (slurm-101_3.out) names
    the task; slurm-101.out is a plain job (101) or a one-task array (grid.py's single run, resume and
    inference: Slurm calls it 101_0)."""
    return slurm_id in log_ids or slurm_id.split("_")[0] in log_ids


def _sacct_record(sacct, job_id):
    """sacct's record for a log's job id (a one-task array's slurm-<id>.out is task <id>_0 there)."""
    return sacct.get(job_id) or sacct.get(f"{job_id}_0")


def _folder_index(folder_name, run_name):
    """3 for "<run_name>-3" / "<run_name>-03" (a grid job's folder), else None."""
    suffix = folder_name[len(run_name) + 1:] if folder_name.startswith(run_name + "-") else ""
    return int(suffix) if suffix.isdigit() else None


def queue_row(info, queue):
    """The squeue row of this folder's training job, if it's queued or running. By the job id in the
    folder's log names; else, for a job that hasn't started (no log yet), by name: an array task of a grid
    launch has name = run_name and task = folder index; a single run, a resume (grid.py) and this tool's
    sbatch rerun are named after the folder."""
    ids = set(info["job_ids"])
    for row in queue:
        if _same_job(row["job"], ids):
            return row
    folder = os.path.basename(info["path"])
    index = _folder_index(folder, info["run_name"])
    for row in queue:
        if row["name"] == folder or (row["name"] == info["run_name"] and row["task"].isdigit()
                                     and int(row["task"]) == index):
            return row
    return None


def inf_queue_row(info, queue):
    """The squeue row of this folder's inference job (launch.sh inf: named <folder>-inf, logs into
    <run>/inf/<inf_name>/), if it's queued or running."""
    ids, name = set(info["inf_job_ids"]), os.path.basename(info["path"]) + "-inf"
    return next((row for row in queue if _same_job(row["job"], ids) or row["name"] == name), None)


# --- state and fix -----------------------------------------------------------

def classify(info, row=None, inf_row=None):
    """The folder's state (see the module docstring). `row`/`inf_row`: its training/inference squeue row."""
    if row:
        return QUEUE_STATES.get(row["state"], row["state"].lower())
    if info["model"]:
        if not info["model_complete"]:
            return "bad_model"
        if not info["has_test"]:
            return "finished"
        if inf_row:
            return "inf_" + QUEUE_STATES.get(inf_row["state"], inf_row["state"].lower())
        return "finished" if info["inf_scores"] is not None else "no_inference"
    if info["checkpoints"]:
        return "interrupted" if info["checkpoints"][-1][1] else "bad_checkpoint"
    return "failed" if info["log"] else "not_started"


def inference_stage(info, state):
    """Whether the folder's row reports its inference job (training is done) rather than its training job."""
    return state.startswith("inf_") or state == "no_inference" or (state == "finished" and bool(info["inf_job_ids"]))


def fix_command(state, info, overrides=()):
    """The command for an unfinished folder (None if there's nothing to run)."""
    rc = shlex.quote(os.path.join(info["path"], "run_config.yaml"))
    launch_overrides = "".join(" " + shlex.quote(f"slurm.{k}={v}") for k, v in overrides)
    if state == "interrupted":
        return f"bash egs/launch.sh sft {rc}{launch_overrides}"
    if state == "no_inference":
        return f"bash egs/launch.sh inf {rc}{launch_overrides}"
    if state in ("failed", "not_started"):
        slurm = {**info["slurm"], **dict(overrides)}
        flags = [flag.format(slurm[k]) for k, flag in SBATCH_FLAGS.items() if slurm.get(k) not in (None, "")]
        flags += ["--requeue", "--open-mode=append",                       # as hfmt/grid.py's sbatch
                  f"--job-name={os.path.basename(info['path'])}",
                  f"--output={os.path.join(info['path'], 'slurm-%j.out')}"]
        return "sbatch " + " ".join(shlex.quote(f) for f in flags) + f" egs/job.sh sft {shlex.quote(info['path'])}"
    if state == "bad_checkpoint":
        step = info["checkpoints"][-1][0]
        return f"# delete {shlex.quote(os.path.join(info['path'], f'checkpoint-{step}'))}, then run this tool again"
    if state == "bad_model":
        return f"# delete {shlex.quote(os.path.join(info['path'], 'model'))}, then run this tool again"
    return None


# --- report ------------------------------------------------------------------

def _table(rows):
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(str(c).ljust(w) for c, w in zip(r, widths)).rstrip() for r in rows)


def _test_text(scores, metric):
    """`metric` per test set with a reference, e.g. "63.60" or "63.60/41.20"; "-" if there's none."""
    values = [s.get(metric) for s in (scores or {}).values() if s]
    return "/".join("-" if v is None else f"{v:.2f}" for v in values) or "-"


def _row(info, state, row, job, start=os.curdir):
    last = max([s for s in (info["progress"], *(step for step, ok in info["checkpoints"] if ok)) if s is not None],
               default=None)
    best = info["best"]
    metric = (info["metric"] or "").removeprefix("eval_")
    best_text = f"{metric} {best['value']:.2f}@{best['step']}" if best and best["value"] is not None else "-"
    test_metric = metric if metric in DECODE_METRICS else "bleu"
    inference = inference_stage(info, state)
    ids = info["inf_job_ids"] if inference else info["job_ids"]
    if row:
        slurm = f"{row['job']} {row['node'] or '-'}"
    elif job:
        slurm = f"{ids[-1]} {job['node'] or '-'} {job['state']} {job['elapsed']}"
    else:
        slurm = ids[-1] if ids else "-"
    if info["config_error"] or state in ACTIVE_OR_DONE:
        reason = info["config_error"]
    elif inference:
        reason = info["inf_error"] or ("launched, no log: cancelled while pending?"
                                       if info["inf_launched"] and not info["inf_log"] else None)
    else:
        reason = info["error"]
    return [os.path.relpath(info["path"], start), state, f"{last if last is not None else '-'}/{info['max_steps'] or '-'}",
            best_text, _test_text(info["scores"], test_metric), _test_text(info["inf_scores"], test_metric),
            info["match"] or "-", slurm, reason or ""]


def report(infos, states, rows, jobs, overrides=(), base=None, slurm_note=""):
    """The printed report: per run_name counts, every folder (training and inference), failure nodes,
    inference that doesn't reproduce training, fix commands. `rows`/`jobs`: each folder's squeue row /
    sacct record of the job its row reports (inference_stage). FOLDER is relative to `base`."""
    start = base or os.curdir
    lines = [f"{len(infos)} run folder(s)" + (f" under {os.path.relpath(base)}" if base else "") + slurm_note, ""]
    by_name = defaultdict(Counter)
    for info, state in zip(infos, states):
        by_name[info["run_name"]][state] += 1
    width = max(len(n) for n in by_name)
    for name in sorted(by_name):
        counts = by_name[name]
        order = sorted(counts, key=lambda s: STATE_ORDER.index(s) if s in STATE_ORDER else len(STATE_ORDER))
        parts = ", ".join(f"{counts[s]} {s}" for s in order)
        lines.append(f"  {name.ljust(width)}  {sum(counts.values()):>3}: {parts}")

    header = ["FOLDER", "STATE", "STEP", "BEST DEV", "TEST (train)", "TEST (inf)", "MATCH",
              "SLURM JOB / NODE / STATE", "REASON"]
    lines += ["", _table([header] + [_row(*x, start) for x in zip(infos, states, rows, jobs)])]

    failed_nodes = Counter(j["node"] for s, j in zip(states, jobs)
                           if j and j["node"] and s not in ACTIVE_OR_DONE and j["state"] in ("FAILED", "NODE_FAIL"))
    if failed_nodes:
        lines += ["", "Nodes of failed jobs: " + ", ".join(f"{n} ({c})" for n, c in failed_nodes.most_common())
                  + "  (several on one node: maybe a bad node -> --slurm exclude=<node>)"]

    differ = [os.path.relpath(info["path"], start) for info in infos if (info["match"] or "").startswith("NO")]
    if differ:
        lines += ["", f"Inference doesn't reproduce training's test output ({len(differ)}): " + ", ".join(differ)]

    checks = [(os.path.relpath(info["path"], start),
               info["warnings"] + ([f"model/ dev check {info['model_check']}"] if info["model_check"] else []))
              for info in infos]
    checks = [(folder, items) for folder, items in checks if items]
    if checks:
        lines += ["", f"Checks failed ({len(checks)}): the run finished, but not as intended (see decisions.md)"]
        lines += [f"  {folder}: {'; '.join(items)}" for folder, items in checks]

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
    train_rows = [queue_row(info, queue) if queue else None for info in infos]
    inf_rows = [inf_queue_row(info, queue) if queue else None for info in infos]
    states = [classify(info, row, inf_row) for info, row, inf_row in zip(infos, train_rows, inf_rows)]
    # Each folder's row reports one job: its inference job once training is done, else its training job.
    stages = [inference_stage(info, state) for info, state in zip(infos, states)]
    rows = [inf_row if stage else row for stage, row, inf_row in zip(stages, train_rows, inf_rows)]
    last_ids = [(info["inf_job_ids"] if stage else info["job_ids"])[-1:] for info, stage in zip(infos, stages)]
    wanted = [ids[0] for ids, state in zip(last_ids, states) if ids and state not in ACTIVE_OR_DONE]
    sacct = finished_jobs(wanted) if queue is not None and wanted else {}
    jobs = [_sacct_record(sacct, ids[0]) if ids else None for ids in last_ids]
    note = "" if queue is not None else ("  (Slurm not queried)" if args.no_slurm else "  (squeue not available: queued jobs not detected)")
    base = os.path.commonpath([os.path.dirname(f) for f in folders])
    print(report(infos, states, rows, jobs, args.slurm, base, note))


if __name__ == "__main__":
    main()
