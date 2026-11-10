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
DECODE = {"max_length": 1024, "max_new_tokens": 128, "num_beams": 1, "do_sample": False, "batch_size": 16}
TEST = {"src": ["/data/test.fr"], "trg": ["/data/test.en"]}
PREDS = ["SECRET hypothesis one\n", "SECRET hypothesis two\n"]   # stand-in for data: must never be printed
TRACEBACK = """Error executing job with overrides: []
Traceback (most recent call last):
  File "/exp/hfmt/hfmt/sft_translation.py", line 180, in main
    raise ValueError("not this one")
torch.AcceleratorError: CUDA error: CUDA-capable device(s) is/are busy or unavailable
Search for `cudaErrorDevicesUnavailable' in https://docs.nvidia.com/cuda/ for more information.
"""


def write_logs(folder, logs):
    for i, (name, text) in enumerate(logs):
        log = os.path.join(folder, name)
        with open(log, "w") as f:
            f.write(text)
        os.utime(log, (1_000_000 + i, 1_000_000 + i))          # later in the list = newer


def make_run(root, folder, run_name="sweep", *, model=False, scores=None, inf_scores=None, checkpoints=(),
             logs=(), inf_logs=(), preds=None, inf_preds=None, inf_decode=None, test=True, max_steps=400,
             metric="eval_bleu"):
    """A run folder like grid.py + training (+ launch.sh inf) leave it. model: True (saved, with
    training_info.yaml) or "partial" (cut off while saving); checkpoints: (step, complete[, best_step,
    best]); logs: (file name, text); scores/inf_scores: the scores.json content; preds/inf_preds: the
    lines of test1.pred.trg. Any inf_* argument also writes inf/test/inf_config.yaml (decode: inf_decode)."""
    path = os.path.join(root, folder)
    os.makedirs(path)
    cfg = {"run_name": run_name, "inf_name": "test", "slurm": SLURM, "decode": DECODE,
           "train": {"max_steps": max_steps, "metric_for_best_model": metric},
           "data": {"test": TEST if test else None}}
    with open(os.path.join(path, "run_config.yaml"), "w") as f:
        f.write("defaults:\n  - base_config\n  - _self_\n\n" + yaml.safe_dump(cfg))   # as grid.py saves it
    if model:
        os.makedirs(os.path.join(path, "model"))
        if model is True:
            open(os.path.join(path, "model", "training_info.yaml"), "w").close()
    inf_dir = os.path.join(path, "inf", "test")
    if inf_scores is not None or inf_logs or inf_preds is not None or inf_decode is not None:
        os.makedirs(inf_dir)
        with open(os.path.join(inf_dir, "inf_config.yaml"), "w") as f:
            yaml.safe_dump({**cfg, "decode": inf_decode or DECODE, "model_dir": os.path.join(path, "model")}, f)
    for where, content in ((path, scores), (inf_dir, inf_scores)):
        if content is not None:
            with open(os.path.join(where, "scores.json"), "w") as f:
                json.dump(content, f)
    for where, lines in ((path, preds), (inf_dir, inf_preds)):
        if lines is not None:
            with open(os.path.join(where, "test1.pred.trg"), "w") as f:
                f.writelines(lines)
    for step, complete, *best in checkpoints:
        ckpt = os.path.join(path, f"checkpoint-{step}")
        os.makedirs(ckpt)
        if complete:
            best_step, best_value = best or (step, 60.0)
            with open(os.path.join(ckpt, "trainer_state.json"), "w") as f:
                json.dump({"global_step": step, "best_metric": best_value,
                           "best_model_checkpoint": os.path.join(path, f"checkpoint-{best_step}")}, f)
    write_logs(path, logs)
    if inf_logs:
        write_logs(inf_dir, inf_logs)
    return path


def bleu(value, bleu_13a=None):
    s = {"src": "/data/test.fr", "trg": "/data/test.en", "bleu": value, "chrf": 70.0, "ter": 30.0}
    if bleu_13a is not None:
        s["bleu_13a_lowercase"] = bleu_13a
    return {"test1": s}


def decoded(root, folder, run_name="sweep", inf_preds=PREDS, inf_value=66.3, bleu_13a=None, **kw):
    """A trained run whose test sets training and launch.sh inf both decoded (identical unless told)."""
    return make_run(root, folder, run_name, model=True, scores=bleu(66.3, bleu_13a), preds=PREDS,
                    inf_scores=bleu(inf_value, bleu_13a), inf_preds=inf_preds, **kw)


# --- states ------------------------------------------------------------------

def test_states_from_files():
    root = tempfile.mkdtemp()
    runs = [
        ("finished", decoded(root, "sweep-0")),
        ("finished", make_run(root, "single", "single", model=True, test=False)),           # no test sets
        ("finished", make_run(root, "sweep-1", model=True, inf_scores=bleu(65.0))),         # training's decode failed
        ("no_inference", make_run(root, "sweep-2", model=True, scores=bleu(66.3))),         # training decoded only
        ("no_inference", make_run(root, "sweep-3", model=True, scores={"test1": {"src": "/x", "trg": None}})),
        ("no_inference", make_run(root, "sweep-4", model=True, scores=bleu(66.3),
                                  inf_logs=[("slurm-90.out", TRACEBACK)])),                 # inference failed
        ("no_inference", make_run(root, "sweep-5", model=True)),                            # test decoding failed
        ("bad_model", make_run(root, "sweep-6", model="partial", checkpoints=[(100, True)])),
        ("interrupted", make_run(root, "sweep-7", checkpoints=[(100, True), (150, True, 100, 63.2)])),
        ("bad_checkpoint", make_run(root, "sweep-8", checkpoints=[(100, True), (150, False)])),
        ("failed", make_run(root, "sweep-9", logs=[("slurm-77_9.out", TRACEBACK)])),
        ("not_started", make_run(root, "sweep-10")),
    ]
    for expected, path in runs:
        assert R.classify(R.inspect_folder(path)) == expected, path


def test_queued_state_wins():
    path = make_run(tempfile.mkdtemp(), "sweep-0", model=True)          # model/ saved, test decoding running
    info = R.inspect_folder(path)
    assert R.classify(info, {"job": "1_0", "name": "sweep", "state": "RUNNING", "node": "c1", "task": "0"}) == "running"
    assert R.classify(info, {"job": "1_0", "name": "sweep", "state": "PENDING", "node": "", "task": "0"}) == "pending"
    inf_row = {"job": "2_0", "name": "sweep-0-inf", "state": "PENDING", "node": "", "task": "0"}
    assert R.classify(info, None, inf_row) == "inf_pending"
    assert R.classify(info, None, {**inf_row, "state": "RUNNING"}) == "inf_running"


def test_find_run_folders_skips_run_contents():
    root = tempfile.mkdtemp()
    a = make_run(root, "sweep-0", checkpoints=[(50, True)])
    b = make_run(os.path.join(root, "nested"), "single", "single")
    os.makedirs(os.path.join(a, "inf", "test"))
    open(os.path.join(a, "inf", "test", "run_config.yaml"), "w").close()   # never a run folder
    done = make_run(os.path.join(root, "complete"), "sweep-1", model=True)  # moved there when finished
    assert R.find_run_folders([root]) == sorted([a, b])
    assert R.find_run_folders([os.path.join(root, "complete")]) == [done]  # unless passed itself


# --- training vs inference ---------------------------------------------------

def test_match_training_and_inference():
    root = tempfile.mkdtemp()
    match = lambda path: R.inspect_folder(path)["match"]
    assert match(decoded(root, "same")) == "yes"
    assert match(decoded(root, "lines", inf_preds=PREDS[:1] + ["other\n", "extra\n"], inf_value=60.0)) == \
        "NO: test1 scores, test1 2 lines"
    assert match(decoded(root, "nopred", inf_preds=None)) == "NO: test1 no pred file"
    assert match(decoded(root, "beam5", inf_decode={**DECODE, "num_beams": 5})) == "decode differs"
    assert match(make_run(root, "inf-only", model=True, inf_scores=bleu(65.0))) is None
    # Older training scores.json without bleu_13a_lowercase: only the metrics both have are compared.
    assert match(make_run(root, "old", model=True, scores=bleu(66.3), preds=PREDS,
                          inf_scores=bleu(66.3, 64.0), inf_preds=PREDS)) == "yes"


def test_test_scores_in_best_dev_metric():
    scores = {"test1": {"bleu": 66.3, "bleu_13a_lowercase": 63.6}, "test2": {}, "test3": {"bleu": 50.0}}
    assert R._test_text(scores, "bleu_13a_lowercase") == "63.60/-"   # test2 has no reference
    assert R._test_text(scores, "bleu") == "66.30/50.00"
    assert R._test_text(None, "bleu") == "-" and R._test_text({"test1": {}}, "bleu") == "-"


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
    assert info["inf_job_ids"] == [] and not info["inf_launched"] and info["inf_scores"] is None


def test_inspect_folder_inference_fields():
    path = make_run(tempfile.mkdtemp(), "sweep-1", model=True, scores=bleu(66.3),
                    inf_logs=[("slurm-90.out", "old"), ("slurm-91.out", TRACEBACK)])
    info = R.inspect_folder(path)
    assert info["inf_launched"] and info["inf_job_ids"] == ["90", "91"] and info["inf_log"] == "slurm-91.out"
    assert info["inf_error"].startswith("torch.AcceleratorError") and info["error"] is None


def test_best_from_run_folder_after_checkpoints_deleted():
    path = make_run(tempfile.mkdtemp(), "sweep-0", model=True, scores=bleu(66.3))   # no checkpoint-*/ left
    with open(os.path.join(path, "trainer_state.json"), "w") as f:                   # trainer.save_state()
        json.dump({"global_step": 350, "best_metric": 64.1,
                   "best_model_checkpoint": os.path.join(path, "checkpoint-200")}, f)
    assert R.inspect_folder(path)["best"] == {"value": 64.1, "step": 200, "global_step": 350}



RELOAD_WARNING = "WARNING:transformers.trainer:There were missing keys in the checkpoint model loaded: ['lm_head.weight'].\n"
RELOAD_LABEL = "checkpoint reload loaded nothing: model/ is the last step"


def _full_run(dev_check, log):
    """A finished run whose training log is ``log`` and whose training_info.yaml has ``dev_check`` (None:
    a run from before the dev check existed)."""
    path = make_run(tempfile.mkdtemp(), "full", model=True, scores=bleu(66.3), logs=[("slurm-10.out", log)])
    if dev_check is not None:
        with open(os.path.join(path, "model", "training_info.yaml"), "w") as f:
            yaml.safe_dump({"model_global_step": 200, "model_dev_check": dev_check}, f)
    return R.inspect_folder(path)


def test_reload_warning_only_counts_without_a_dev_check():
    # Before the fix: no measured check, so the warning is the only evidence.
    old = _full_run(None, RELOAD_WARNING)
    assert old["warnings"] == [RELOAD_LABEL] and old["model_check"] is None
    # After it the Trainer's own reload still warns, load_checkpoint_weights redoes it, the check measures it.
    for check in ("ok", "unchecked: no dev BLEU at step 200 in log_history"):
        fixed = _full_run(check, RELOAD_WARNING)
        assert fixed["warnings"] == [] and fixed["model_check"] is None
    bad = _full_run("MISMATCH: model/ dev bleu 24.74 vs 34.33 at step 200 (tolerance 1.0)", RELOAD_WARNING)
    assert bad["warnings"] == [] and bad["model_check"].startswith("MISMATCH")


def test_prompt_mismatch_warning_counts_with_a_dev_check():
    info = _full_run("ok", "WARNING: Mismatch between tokenized prompt and the start of tokenized prompt+completion\n")
    assert info["warnings"] == ["prompt not a token prefix of prompt + reply (chat template)"]


# --- Slurm -------------------------------------------------------------------

def test_parse_squeue_and_sacct():
    rows = R.parse_squeue("101_0|sweep|RUNNING|c07|0\n101_3|sweep|PENDING||3\n205|sweep-6|PENDING||N/A\n\nbad")
    assert rows[1] == {"job": "101_3", "name": "sweep", "state": "PENDING", "node": "", "task": "3"}
    assert len(rows) == 3
    jobs = R.parse_sacct("77_6|FAILED|c07|00:00:41\n78|CANCELLED by 1234|c08|00:10:00\n500_0|FAILED|c09|00:01:00\n")
    assert jobs["77_6"] == {"state": "FAILED", "node": "c07", "elapsed": "00:00:41"}
    assert jobs["78"]["state"] == "CANCELLED"
    assert R._sacct_record(jobs, "500")["node"] == "c09"       # slurm-500.out: a one-task array's task 500_0
    assert R._sacct_record(jobs, "78")["state"] == "CANCELLED" and R._sacct_record(jobs, "79") is None


def test_queue_row_matching():
    root = tempfile.mkdtemp()
    running = R.inspect_folder(make_run(root, "sweep-03", logs=[("slurm-101_3.out", "")]))
    pending = R.inspect_folder(make_run(root, "sweep-04"))                 # no log yet: match by name + index
    rerun = R.inspect_folder(make_run(root, "sweep-05", logs=[("slurm-77_5.out", TRACEBACK)]))
    resume = R.inspect_folder(make_run(root, "sweep-06", checkpoints=[(50, True)], logs=[("slurm-77_6.out", "")]))
    other = R.inspect_folder(make_run(root, "other-04", "other"))
    single = R.inspect_folder(make_run(root, "single", "single", logs=[("slurm-500.out", "")]))
    queue = R.parse_squeue("101_3|sweep|RUNNING|c07|3\n101_4|sweep|PENDING||4\n300|sweep-05|PENDING||N/A\n"
                           "400_0|sweep-06|PENDING||0\n500_0|single|RUNNING|c08|0")
    assert R.queue_row(running, queue)["job"] == "101_3"
    assert R.queue_row(pending, queue)["job"] == "101_4"                   # "sweep-04" is task 4
    assert R.queue_row(rerun, queue)["job"] == "300"                       # this tool's sbatch: name = folder
    assert R.queue_row(resume, queue)["job"] == "400_0"                    # grid.py's resume: name = folder
    assert R.queue_row(single, queue)["job"] == "500_0"                    # slurm-500.out is task 500_0
    assert R.queue_row(other, queue) is None                               # same index, other run_name


def test_inf_queue_row_matching():
    root = tempfile.mkdtemp()
    running = R.inspect_folder(make_run(root, "sweep-1", model=True, inf_logs=[("slurm-600.out", "")]))
    pending = R.inspect_folder(make_run(root, "sweep-2", model=True, inf_decode=DECODE))   # launched, no log
    idle = R.inspect_folder(make_run(root, "sweep-3", model=True))
    queue = R.parse_squeue("600_0|sweep-1-inf|RUNNING|c07|0\n601_0|sweep-2-inf|PENDING||0\n"
                           "602_0|sweep-3|PENDING||0")                     # a training job, not inference
    assert R.inf_queue_row(running, queue)["job"] == "600_0"
    assert R.inf_queue_row(pending, queue)["job"] == "601_0"
    assert R.inf_queue_row(idle, queue) is None


# --- fix commands ------------------------------------------------------------

def test_fix_commands():
    root = tempfile.mkdtemp()
    failed = R.inspect_folder(make_run(root, "sweep-6", logs=[("slurm-77_6.out", TRACEBACK)]))
    cmd = R.fix_command("failed", failed, [("exclude", "c[07-08]"), ("gres", "gpu:l40s:1")])
    assert cmd.startswith("sbatch --partition=gpu --gres=gpu:l40s:1 --time=4:00:00 --mem=25G --cpus-per-task=1 ")
    assert "'--exclude=c[07-08]'" in cmd                                   # quoted for the shell
    assert " --requeue --open-mode=append " in cmd                         # as hfmt/grid.py's sbatch
    assert f"--job-name=sweep-6 --output={failed['path']}/slurm-%j.out egs/job.sh sft {failed['path']}" in cmd
    assert "--constraint" not in cmd                                       # unset in the saved config

    interrupted = R.inspect_folder(make_run(root, "sweep-4", checkpoints=[(100, True)]))
    assert R.fix_command("interrupted", interrupted, [("exclude", "c07")]) == \
        f"bash egs/launch.sh sft {interrupted['path']}/run_config.yaml slurm.exclude=c07"
    assert R.fix_command("no_inference", interrupted, [("exclude", "c07")]) == \
        f"bash egs/launch.sh inf {interrupted['path']}/run_config.yaml slurm.exclude=c07"
    bad = R.inspect_folder(make_run(root, "sweep-5", checkpoints=[(100, True), (150, False)]))
    assert R.fix_command("bad_checkpoint", bad) == f"# delete {bad['path']}/checkpoint-150, then run this tool again"
    assert R.fix_command("bad_model", bad) == f"# delete {bad['path']}/model, then run this tool again"
    for state in ("finished", "running", "inf_running", "inf_pending"):
        assert R.fix_command(state, failed) is None


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
    decoded(root, "sweep-0", metric="eval_bleu_13a_lowercase", bleu_13a=63.6)
    make_run(root, "sweep-1", logs=[("slurm-77_1.out", TRACEBACK)])
    make_run(root, "sweep-2", logs=[("slurm-77_2.out", TRACEBACK)])
    make_run(root, "sweep-3", model=True, scores=bleu(66.31), preds=PREDS)                 # no inference yet
    make_run(root, "sweep-4", model=True, scores=bleu(66.3), preds=PREDS,
             inf_logs=[("slurm-90.out", "ValueError: inference broke\n")])               # inference failed
    decoded(root, "sweep-5", inf_preds=PREDS[:1] + ["other\n"], inf_value=60.0)            # doesn't reproduce
    make_run(root, "full", "full", checkpoints=[(40, True, 20, 67.5)])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        R.main([root, "--no-slurm", "--slurm", "exclude=c07"])
    text = out.getvalue()
    assert text.startswith(f"7 run folder(s) under {os.path.relpath(root)}  (Slurm not queried)")
    assert "sweep    6: 2 finished, 2 no_inference, 2 failed" in text and "full     1: 1 interrupted" in text
    rows = {line.split()[0]: line for line in text.splitlines() if line.startswith(("sweep-", "full "))}
    assert rows["sweep-0"].split()[1] == "finished" and " yes " in rows["sweep-0"]
    assert rows["sweep-0"].count("63.60") == 2 and "66.30" not in rows["sweep-0"]   # TEST in the BEST DEV metric
    assert "66.31" in rows["sweep-3"] and "no_inference" in rows["sweep-3"]
    assert "ValueError: inference broke" in rows["sweep-4"] and "90" in rows["sweep-4"]
    assert "NO: test1 scores, test1 1 lines" in rows["sweep-5"] and "60.00" in rows["sweep-5"]
    assert "Inference doesn't reproduce training's test output (1): sweep-5" in text
    assert "torch.AcceleratorError: CUDA error" in text and "bleu 67.50@20" in text
    assert "# no_inference (2): decode the test sets" in text and text.count("egs/launch.sh inf") == 2
    assert "# failed (2): rerun from the saved run_config.yaml" in text
    assert text.count("egs/job.sh sft") == 2 and "slurm.exclude=c07" in text
    assert "SECRET" not in text                                            # predictions are never printed


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
