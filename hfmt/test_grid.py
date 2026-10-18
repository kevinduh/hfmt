"""Tests for hfmt/grid.py (features 08_hydra_redo, 09_hydra_detailed_reorg). No GPU or training deps needed.

Runs grid.main() in-process on the real fr-en experiments, with output_root pointed at a temp
dir, and checks the job folders and saved configs it writes. HFMT_DATA_ROOT is a dummy path
(grid.py doesn't open data files), so no local_setting.sh is needed.

Run standalone (pytest isn't installed in the hfmt env):
    HFMT_ROOT=$(pwd) python hfmt/test_grid.py
"""

import contextlib
import io
import os
import shutil
import sys
import tempfile

from omegaconf import OmegaConf

ROOT = os.environ.setdefault("HFMT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["HFMT_DATA_ROOT"] = "/dummy/data/fr-en"
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
    assert (cfg.train.learning_rate, cfg.peft.lora.lora_r) == (1e-5, 8)
    lines = open(os.path.join(out, "sft_lora_sweep.grid.txt")).read().splitlines()
    assert lines[2] == "2: train.learning_rate=1e-05 peft.lora.lora_r=8"


def test_cli_value_pins_and_comma_list_adds_axis():
    assert len(run(SWEEP, "peft.lora.lora_r=8")[1]) == 2
    assert len(run(SWEEP, "train.seed=1,2")[1]) == 8


def test_comma_list_on_single_file_makes_a_grid():
    assert run(SINGLE, "peft.lora.lora_r=8,16")[1] == ["sft_lora-0", "sft_lora-1"]


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
    assert "renamed" in dirs and saved(out, "renamed").peft.lora.lora_r == 16


def test_errors_fail_before_writing():
    assert fails(SINGLE, "train.lerning_rate=1")      # unknown key
    assert fails(SWEEP, "peft.lora.lora_r=abc,8")          # bad value in one grid job
    assert fails(SINGLE, "slurm.gres=a,b")            # slurm can't be an axis
    tmp = tempfile.mkdtemp()
    cfg_named = os.path.join(tmp, "config.yaml")
    open(cfg_named, "w").write(open(SINGLE).read())
    assert fails(cfg_named)                            # clashes with conf/config.yaml


def test_each_peft_option_composes():
    for opt, mode in [("full_finetune", "full"), ("lora_default", "lora"), ("qlora_default", "qlora")]:
        out, _, _ = run(SINGLE, f"peft={opt}")
        peft = saved(out, "sft_lora").peft
        assert peft.mode == mode
        assert (peft.lora is not None, peft.quant is not None) == (mode != "full", mode == "qlora")


def test_peft_mode_must_match_blocks():
    assert fails(SINGLE, "peft=full_finetune", "peft.lora.lora_r=16")  # partial lora block on full
    assert fails(SINGLE, "peft=qlora_default", "peft.mode=lora")       # lora with a quant block
    assert fails(SINGLE, "peft.mode=adapter")                          # unknown mode


def test_inference_callback_interval():
    out, _, _ = run(SINGLE)
    assert saved(out, "sft_lora").train.inference_callback_interval == 1   # conf/train/default.yaml
    out, _, _ = run(SINGLE, "train=longrun", "train.inference_callback_interval=0")
    assert saved(out, "sft_lora").train.inference_callback_interval == 0   # every train option sets it
    assert fails(SINGLE, "train.inference_callback_interval=-1")


def test_missing_value_fails():
    tmp = tempfile.mkdtemp()                           # experiment that sets no checkpoint
    shutil.copy(os.path.join(os.path.dirname(SINGLE), "dataset.yaml"), tmp)
    exp = os.path.join(tmp, "no_ckpt.yaml")
    open(exp, "w").write("".join(l for l in open(SINGLE) if not l.startswith("checkpoint:")))
    assert fails(exp)


def test_missing_data_root_gives_hint():
    os.environ.pop("HFMT_DATA_ROOT")
    try:
        run(SINGLE)
        assert False, "should have failed"
    except SystemExit as e:
        assert "local_setting.sh" in str(e.code)
    finally:
        os.environ["HFMT_DATA_ROOT"] = "/dummy/data/fr-en"


def test_dataset_with_absolute_paths_needs_no_data_root():
    tmp = tempfile.mkdtemp()                           # no data_root/src/trg, no local_setting.sh
    open(os.path.join(tmp, "dataset.yaml"), "w").write(
        "# @package data\n" + "".join(f"{s}: {{src: [/abs/{s}.fr], trg: [/abs/{s}.en]}}\n" for s in ("train", "dev")))
    shutil.copy(SINGLE, tmp)
    saved_root = os.environ.pop("HFMT_DATA_ROOT")
    try:
        out, dirs, _ = run(os.path.join(tmp, "sft_lora.yaml"))
    finally:
        os.environ["HFMT_DATA_ROOT"] = saved_root
    data = saved(out, "sft_lora").data
    assert data.data_root is None and data.test is None and list(data.train.src) == ["/abs/train.fr"]



def inf(rc, *overrides):
    """Run grid.main --inf; return stdout. SystemExit propagates."""
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
        grid.main(["--inf", rc, *overrides])
    return stdout.getvalue()


def trained_run():
    """A saved single run folder with a (fake) trained model/; returns (run folder, run_config path)."""
    out, _, _ = run(SINGLE)
    run_dir = os.path.join(out, "sft_lora")
    os.makedirs(os.path.join(run_dir, "model"))
    return run_dir, os.path.join(run_dir, "run_config.yaml")


def test_inf_saves_its_own_config_and_leaves_run_config():
    run_dir, rc = trained_run()
    before = open(rc).read()
    stdout = inf(rc, "decode.num_beams=5", "inf_name=beam5")
    inf_dir = os.path.join(run_dir, "inf", "beam5")
    cfg = OmegaConf.load(os.path.join(inf_dir, "inf_config.yaml"))
    assert cfg.model_dir == os.path.join(run_dir, "model") and cfg.decode.num_beams == 5
    assert open(rc).read() == before and os.path.isfile(os.path.join(inf_dir, "git_rev"))
    assert "--array=0-0%4" in stdout and "--job-name=sft_lora-inf" in stdout
    assert f"--output={inf_dir}/slurm-%A.out" in stdout and f"JOBS=({inf_dir})" in stdout


def test_inf_default_name_is_test():
    run_dir, rc = trained_run()
    inf(rc)
    assert os.path.isfile(os.path.join(run_dir, "inf", "test", "inf_config.yaml"))


def test_inf_errors_fail_before_writing():
    run_dir, rc = trained_run()
    for bad in [("decode.num_beams=1,5",),                # comma list
                ("decode.num_bems=5",)]:                  # unknown key
        try:
            inf(rc, *bad)
            assert False, bad
        except SystemExit:
            pass
    for exp in [SINGLE, os.path.join(trained_run()[0], "missing.yaml")]:   # not a run_config
        try:
            inf(exp)
            assert False, exp
        except SystemExit:
            pass
    shutil.rmtree(os.path.join(run_dir, "model"))         # no trained model
    try:
        inf(rc)
        assert False
    except SystemExit:
        pass
    assert not os.path.exists(os.path.join(run_dir, "inf"))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
