"""Tests for hfmt/grid.py (feature 08_hydra_redo). No GPU or training deps needed.

Runs grid.main() in-process on the real fr-en experiments, with output_root pointed at a temp
dir, and checks the job folders and saved configs it writes.

Run standalone (pytest isn't installed in the hfmt env):
    HFMT_ROOT=$(pwd) python hfmt/test_grid.py
"""

import contextlib
import io
import os
import sys
import tempfile

from omegaconf import OmegaConf

ROOT = os.environ.setdefault("HFMT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from hfmt import grid  # noqa: E402

SINGLE = os.path.join(ROOT, "egs/mmtc/fr-en/sft_lora.yaml")
SWEEP = os.path.join(ROOT, "egs/mmtc/fr-en/sft_lora.sweep.yaml")


def run(exp, *overrides, out=None):
    """Run grid.main; return (output_root, sorted job-folder names, stdout). SystemExit propagates."""
    out = out or tempfile.mkdtemp()
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
        grid.main([exp, *overrides, f"output_root={out}"])
    dirs = sorted(d for d in os.listdir(out) if os.path.isdir(os.path.join(out, d)))
    return out, dirs, stdout.getvalue()


def saved(out, name):
    return OmegaConf.load(os.path.join(out, name, "run_config.yaml"))


def fails(exp, *overrides):
    out = tempfile.mkdtemp()
    try:
        run(exp, *overrides, out=out)
    except SystemExit:
        return os.listdir(out) == []  # failed before writing anything
    return False


def test_single_is_one_job_without_suffix():
    out, dirs, stdout = run(SINGLE)
    assert dirs == ["sft_lora"]
    assert "--array=0-0%4" in stdout and "--gres=gpu:a100:1" in stdout


def test_sweep_runs_full_grid():
    out, dirs, _ = run(SWEEP)
    assert dirs == [f"sft_lora_sweep-{i}" for i in range(4)]
    cfg = saved(out, "sft_lora_sweep-2")  # grid order: lr outer, lora_r inner
    assert (cfg.train.learning_rate, cfg.lora.lora_r) == (2e-4, 8)
    lines = open(os.path.join(out, "sft_lora_sweep.grid.txt")).read().splitlines()
    assert lines[2] == "2: train.learning_rate=0.0002 lora.lora_r=8"


def test_cli_value_pins_and_comma_list_adds_axis():
    assert len(run(SWEEP, "lora.lora_r=8")[1]) == 2
    assert len(run(SWEEP, "train.seed=1,2")[1]) == 8


def test_comma_list_on_single_file_makes_a_grid():
    assert run(SINGLE, "lora.lora_r=8,16")[1] == ["sft_lora-0", "sft_lora-1"]


def test_zero_padding_above_ten_jobs():
    _, dirs, stdout = run(SINGLE, "train.seed=" + ",".join(map(str, range(11))))
    assert dirs[0] == "sft_lora-00" and dirs[-1] == "sft_lora-10"
    assert "sft_lora-%2a/slurm-%A_%a.out" in stdout


def test_saved_config_is_resolved_and_complete():
    out, _, _ = run(SINGLE)
    text = open(os.path.join(out, "sft_lora", "run_config.yaml")).read()
    assert "${" not in text and text.startswith("defaults:\n  - base_config\n  - _self_\n")
    cfg = saved(out, "sft_lora")
    assert cfg.output_root == out and os.path.isabs(cfg.data.train.src[0])
    assert "hydra" not in cfg
    assert os.path.isfile(os.path.join(out, "sft_lora", "git_rev"))


def test_rerun_saved_config_reuses_its_folder_unless_renamed():
    out, _, _ = run(SWEEP)
    rc = os.path.join(out, "sft_lora_sweep-1", "run_config.yaml")
    run(rc, "train.max_steps=2000", out=out)
    assert saved(out, "sft_lora_sweep-1").train.max_steps == 2000
    _, dirs, _ = run(rc, "run_name=renamed", out=out)
    assert "renamed" in dirs and saved(out, "renamed").lora.lora_r == 16


def test_errors_fail_before_writing():
    assert fails(SINGLE, "train.lerning_rate=1")      # unknown key
    assert fails(SWEEP, "lora.lora_r=abc,8")          # bad value in one grid job
    assert fails(SINGLE, "slurm.gres=a,b")            # slurm can't be an axis
    tmp = tempfile.mkdtemp()
    cfg_named = os.path.join(tmp, "config.yaml")
    open(cfg_named, "w").write(open(SINGLE).read())
    assert fails(cfg_named)                            # clashes with conf/config.yaml


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
