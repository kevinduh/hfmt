"""Tests for tools/run_status.py (no Slurm needed: squeue/sacct output is passed in as text).

Run standalone (pytest isn't installed in the hfmt env), or with pytest:
    python tools/test_run_status.py
"""

import contextlib
import io
import json
import os
import sys
import tempfile

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_status as R  # noqa: E402

SLURM = {"partition": "gpu", "gres": "gpu:a100:1", "time": "4:00:00", "mem_gb": 25, "cpus": 1,
         "array_parallelism": 4, "constraint": None, "exclude": None}
TRACEBACK = """Error executing job with overrides: []
Traceback (most recent call last):
  File "/exp/hfmt/hfmt/sft_translation.py", line 180, in main
    raise ValueError("not this one")
torch.AcceleratorError: CUDA error: CUDA-capable device(s) is/are busy or unavailable
Search for `cudaErrorDevicesUnavailable' in https://docs.nvidia.com/cuda/ for more information.
"""


def make_run(root, folder, run_name="sweep", *, model=False, scores=None, inf_scores=None, checkpoints=(),
             logs=(), test=True, max_steps=400):
    """A run folder like grid.py + training leave it. checkpoints: (step, complete[, best_step, best]);
    logs: (file name, text); scores/inf_scores: the scores.json content."""
    path = os.path.join(root, folder)
    os.makedirs(path)
    cfg = {"run_name": run_name, "inf_name": "test", "slurm": SLURM,
           "train": {"max_steps": max_steps, "metric_for_best_model": "eval_bleu"},
           "data": {"test": {"src": ["/data/test.fr"], "trg": ["/data/test.en"]} if test else None}}
    with open(os.path.join(path, "run_config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)
    if model:
        os.makedirs(os.path.join(path, "model"))
    for where, content in ((path, scores), (os.path.join(path, "inf", "test"), inf_scores)):
        if content is not None:
            os.makedirs(where, exist_ok=True)
            with open(os.path.join(where, "scores.json"), "w") as f:
                json.dump(content, f)
    for step, complete, *best in checkpoints:
        ckpt = os.path.join(path, f"checkpoint-{step}")
        os.makedirs(ckpt)
        if complete:
            best_step, best_value = best or (step, 60.0)
            with open(os.path.join(ckpt, "trainer_state.json"), "w") as f:
                json.dump({"global_step": step, "best_metric": best_value,
                           "best_model_checkpoint": os.path.join(path, f"checkpoint-{best_step}")}, f)
    for i, (name, text) in enumerate(logs):
        log = os.path.join(path, name)
        with open(log, "w") as f:
            f.write(text)
        os.utime(log, (1_000_000 + i, 1_000_000 + i))          # later in the list = newer
    return path


def bleu(value):
    return {"test1": {"src": "/data/test.fr", "trg": "/data/test.en", "bleu": value, "chrf": 70.0, "ter": 30.0}}


# --- states ------------------------------------------------------------------

def test_states_from_files():
    root = tempfile.mkdtemp()
    runs = [
        ("finished", make_run(root, "sweep-0", model=True, scores=bleu(66.3))),
        ("finished", make_run(root, "single", "single", model=True, test=False)),           # no test sets
        ("finished", make_run(root, "sweep-1", model=True, scores={"test1": {"src": "/x", "trg": None}})),
        ("finished", make_run(root, "sweep-2", model=True, inf_scores=bleu(65.0))),         # decoded by launch.sh inf
        ("no_test_scores", make_run(root, "sweep-3", model=True)),
        ("interrupted", make_run(root, "sweep-4", checkpoints=[(100, True), (150, True, 100, 63.2)])),
        ("bad_checkpoint", make_run(root, "sweep-5", checkpoints=[(100, True), (150, False)])),
        ("failed", make_run(root, "sweep-6", logs=[("slurm-77_6.out", TRACEBACK)])),
        ("not_started", make_run(root, "sweep-7")),
    ]
    for expected, path in runs:
        assert R.classify(R.inspect_folder(path)) == expected, path


def test_queued_state_wins():
    path = make_run(tempfile.mkdtemp(), "sweep-0", model=True)          # model/ saved, test decoding running
    info = R.inspect_folder(path)
    assert R.classify(info, {"job": "1_0", "name": "sweep", "state": "RUNNING", "node": "c1", "task": "0"}) == "running"
    assert R.classify(info, {"job": "1_0", "name": "sweep", "state": "PENDING", "node": "", "task": "0"}) == "pending"


def test_find_run_folders_skips_run_contents():
    root = tempfile.mkdtemp()
    a = make_run(root, "sweep-0", checkpoints=[(50, True)])
    b = make_run(os.path.join(root, "nested"), "single", "single")
    os.makedirs(os.path.join(a, "inf", "test"))
    open(os.path.join(a, "inf", "test", "run_config.yaml"), "w").close()   # never a run folder
    assert R.find_run_folders([root]) == sorted([a, b])


# --- logs --------------------------------------------------------------------

def test_last_error_line():
    assert R._last_error(TRACEBACK) == ("torch.AcceleratorError: CUDA error: CUDA-capable device(s) "
                                        "is/are busy or unavailable")
    assert R._last_error("loss 0.4\nError executing job with overrides: []\n") is None
    slurm = "slurmstepd: error: *** JOB 12 ON c07 CANCELLED AT 2026-10-30T12:00:00 DUE TO TIME LIMIT ***"
    assert R._last_error("RuntimeError: first\n" + slurm) == slurm
    long = R._last_error("ValueError: " + "x" * 500)
    assert len(long) == R.ERROR_CHARS and long.endswith("...")


def test_progress_from_tqdm():
    text = " 10%|#  | 40/400 [00:10<01:00]\r 11%|#  | 44/400 [00:11<01:00]\r100%|###| 125/125 [dev decoding]"
    assert R._progress(text, 400) == 44                        # bars of other lengths are ignored
    assert R._progress(text, None) is None


def test_inspect_folder_fields():
    path = make_run(tempfile.mkdtemp(), "sweep-4",
                    checkpoints=[(100, True), (150, True, 100, 63.25)],
                    logs=[("slurm-70_4.out", "old"), ("slurm-71.out", " 40%| | 160/400 [")])
    info = R.inspect_folder(path)
    assert info["job_ids"] == ["70_4", "71"] and info["log"] == "slurm-71.out"
    assert info["best"] == {"value": 63.25, "step": 100, "global_step": 150}
    assert info["progress"] == 160 and info["error"] is None
    assert info["run_name"] == "sweep" and info["max_steps"] == 400


# --- Slurm -------------------------------------------------------------------

def test_parse_squeue_and_sacct():
    rows = R.parse_squeue("101_0|sweep|RUNNING|c07|0\n101_3|sweep|PENDING||3\n205|sweep-6|PENDING||N/A\n\nbad")
    assert rows[1] == {"job": "101_3", "name": "sweep", "state": "PENDING", "node": "", "task": "3"}
    assert len(rows) == 3
    jobs = R.parse_sacct("77_6|FAILED|c07|00:00:41\n78|CANCELLED by 1234|c08|00:10:00\n")
    assert jobs["77_6"] == {"state": "FAILED", "node": "c07", "elapsed": "00:00:41"}
    assert jobs["78"]["state"] == "CANCELLED"


def test_queue_row_matching():
    root = tempfile.mkdtemp()
    running = R.inspect_folder(make_run(root, "sweep-03", logs=[("slurm-101_3.out", "")]))
    pending = R.inspect_folder(make_run(root, "sweep-04"))                 # no log yet: match by name + index
    rerun = R.inspect_folder(make_run(root, "sweep-05", logs=[("slurm-77_5.out", TRACEBACK)]))
    other = R.inspect_folder(make_run(root, "other-04", "other"))
    queue = R.parse_squeue("101_3|sweep|RUNNING|c07|3\n101_4|sweep|PENDING||4\n300|sweep-05|PENDING||N/A")
    assert R.queue_row(running, queue)["job"] == "101_3"
    assert R.queue_row(pending, queue)["job"] == "101_4"                   # "sweep-04" is task 4
    assert R.queue_row(rerun, queue)["job"] == "300"                       # this tool's sbatch: name = folder
    assert R.queue_row(other, queue) is None                               # same index, other run_name


# --- fix commands ------------------------------------------------------------

def test_fix_commands():
    root = tempfile.mkdtemp()
    failed = R.inspect_folder(make_run(root, "sweep-6", logs=[("slurm-77_6.out", TRACEBACK)]))
    cmd = R.fix_command("failed", failed, [("exclude", "c[07-08]"), ("gres", "gpu:l40s:1")])
    assert cmd.startswith("sbatch --partition=gpu --gres=gpu:l40s:1 --time=4:00:00 --mem=25G --cpus-per-task=1 ")
    assert "'--exclude=c[07-08]'" in cmd                                   # quoted for the shell
    assert f"--job-name=sweep-6 --output={failed['path']}/slurm-%j.out egs/job.sh sft {failed['path']}" in cmd
    assert "--constraint" not in cmd                                       # unset in the saved config

    interrupted = R.inspect_folder(make_run(root, "sweep-4", checkpoints=[(100, True)]))
    assert R.fix_command("interrupted", interrupted, [("exclude", "c07")]) == \
        f"bash egs/launch.sh sft {interrupted['path']}/run_config.yaml slurm.exclude=c07"
    assert R.fix_command("no_test_scores", interrupted) == f"bash egs/launch.sh inf {interrupted['path']}/run_config.yaml"
    bad = R.inspect_folder(make_run(root, "sweep-5", checkpoints=[(100, True), (150, False)]))
    assert R.fix_command("bad_checkpoint", bad) == f"# delete {bad['path']}/checkpoint-150, then run this tool again"
    assert R.fix_command("finished", failed) is None and R.fix_command("running", failed) is None


def test_parse_args_slurm_overrides():
    assert R.parse_args(["d", "--slurm", "exclude=c07", "--slurm", "mem_gb=48"]).slurm == \
        [("exclude", "c07"), ("mem_gb", "48")]
    for bad in ("exclude", "array_parallelism=2"):
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                R.parse_args(["d", "--slurm", bad])
        except SystemExit:
            continue
        raise AssertionError(bad)


# --- report ------------------------------------------------------------------

def test_report_and_main():
    root = tempfile.mkdtemp()
    make_run(root, "sweep-0", model=True, scores=bleu(66.31))
    make_run(root, "sweep-1", logs=[("slurm-77_1.out", TRACEBACK)])
    make_run(root, "sweep-2", logs=[("slurm-77_2.out", TRACEBACK)])
    make_run(root, "full", "full", checkpoints=[(40, True, 20, 67.5)])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        R.main([root, "--no-slurm", "--slurm", "exclude=c07"])
    text = out.getvalue()
    assert text.startswith("4 run folder(s)  (Slurm not queried)")
    assert "sweep    3: 1 finished, 2 failed" in text and "full     1: 1 interrupted" in text
    assert "sweep-0" not in text                                           # finished: not listed
    assert "torch.AcceleratorError: CUDA error" in text and "bleu 67.50@20" in text
    assert "# failed (2): rerun from the saved run_config.yaml" in text
    assert text.count("egs/job.sh sft") == 2 and "slurm.exclude=c07" in text
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        R.main([root, "--no-slurm", "--all"])
    assert "66.31" in out.getvalue()                                       # --all lists finished runs


def test_report_failed_nodes():
    root = tempfile.mkdtemp()
    infos = [R.inspect_folder(make_run(root, f"sweep-{i}", logs=[(f"slurm-9_{i}.out", TRACEBACK)])) for i in range(3)]
    jobs = [{"state": "FAILED", "node": "c07", "elapsed": "00:00:41"},
            {"state": "FAILED", "node": "c07", "elapsed": "00:00:39"},
            {"state": "CANCELLED", "node": "c08", "elapsed": "00:01:00"}]   # cancelled: not evidence of a bad node
    text = R.report(infos, ["failed"] * 3, [None] * 3, jobs)
    assert "Nodes of failed jobs: c07 (2)" in text and "c08 (" not in text
    assert "9_0 c07 FAILED 00:00:41" in text


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
