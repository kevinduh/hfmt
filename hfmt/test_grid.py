"""Tests for hfmt/grid.py (features 08_hydra_redo, 09_hydra_detailed_reorg, 10_inference_more_options, 12_resume_sft). No GPU or training deps needed.

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
from hfmt.hydra_config import CPT_FORCED_TRAIN, apply_cpt_overrides  # noqa: E402

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
    assert "--requeue --open-mode=append" in stdout                    # always added, not slurm: keys


def test_sweep_runs_full_grid():
    out, dirs, _ = run(SWEEP)
    assert dirs == [f"sft_lora_sweep-{i}" for i in range(4)]
    cfg = saved(out, "sft_lora_sweep-2")  # grid order: lr outer, lora.r inner
    assert (cfg.train.learning_rate, cfg.peft.lora.r) == (1e-5, 8)
    lines = open(os.path.join(out, "sft_lora_sweep.grid.txt")).read().splitlines()
    assert lines[2] == "2: train.learning_rate=1e-05 peft.lora.r=8"


def test_cli_value_pins_and_comma_list_adds_axis():
    assert len(run(SWEEP, "peft.lora.r=8")[1]) == 2
    assert len(run(SWEEP, "train.seed=1,2")[1]) == 8


def test_comma_list_on_single_file_makes_a_grid():
    assert run(SINGLE, "peft.lora.r=8,16")[1] == ["sft_lora-0", "sft_lora-1"]


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


def cancelled_run(step=40):
    """A saved single run folder with a (fake) complete checkpoint; returns (run folder, run_config path)."""
    out, _, _ = run(SINGLE)
    run_dir = os.path.join(out, "sft_lora")
    os.makedirs(os.path.join(run_dir, f"checkpoint-{step}"))
    open(os.path.join(run_dir, f"checkpoint-{step}", "trainer_state.json"), "w").close()
    return run_dir, os.path.join(run_dir, "run_config.yaml")


def resume(rc, *overrides):
    """Run grid.main on a saved run_config.yaml (a resume; no output_root override); return stdout."""
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
        grid.main([rc, *overrides])
    return stdout.getvalue()


def test_resume_saved_config_uses_its_folder_and_writes_nothing():
    run_dir, rc = cancelled_run()
    before = {f: open(os.path.join(run_dir, f)).read() for f in ("run_config.yaml", "git_rev")}
    names = sorted(os.listdir(run_dir))
    stdout = resume(rc, "slurm.gres=gpu:l40s:1")
    assert f"JOBS=({run_dir})" in stdout and "KIND=sft_resume" in stdout
    assert "--gres=gpu:l40s:1" in stdout and f"--output={run_dir}/slurm-%A.out" in stdout
    assert sorted(os.listdir(run_dir)) == names
    assert all(open(os.path.join(run_dir, f)).read() == v for f, v in before.items())
    assert "KIND" not in run(SINGLE)[2]                    # only a resume sets it


def test_resume_refusals():
    run_dir, rc = cancelled_run()
    resume(rc)                                             # the good case passes
    out, _, _ = run(SINGLE)
    no_ckpt = os.path.join(out, "sft_lora", "run_config.yaml")
    incomplete, _ = cancelled_run()
    os.makedirs(os.path.join(incomplete, "checkpoint-60"))   # newest checkpoint cut off while saving
    finished, _ = cancelled_run()
    os.makedirs(os.path.join(finished, "model"))
    for bad in [(rc, "train.max_steps=2000"),              # only slurm.* may change
                (rc, "run_name=renamed"),                  # same run, same folder
                (rc, "output_root=/tmp/x"),
                (rc, "slurm.gres=a,b"),                    # comma list
                (no_ckpt,),
                (os.path.join(incomplete, "run_config.yaml"),),
                (os.path.join(finished, "run_config.yaml"),)]:
        try:
            resume(*bad)
            assert False, bad
        except SystemExit:
            pass


def test_errors_fail_before_writing():
    assert fails(SINGLE, "train.lerning_rate=1")      # unknown key
    assert fails(SWEEP, "peft.lora.r=abc,8")          # bad value in one grid job
    assert fails(SINGLE, "slurm.gres=a,b")            # slurm can't be an axis
    tmp = tempfile.mkdtemp()
    cfg_named = os.path.join(tmp, "config.yaml")
    open(cfg_named, "w").write(open(SINGLE).read())
    assert fails(cfg_named)                            # clashes with conf/config.yaml


def test_each_peft_option_composes():
    for opt, mode in [("full_finetune", "full"), ("lora_default", "lora"), ("qlora_default", "qlora"),
                      ("lora_qwen3_5_all_linear", "lora"), ("lora_gemma3_all_linear", "lora"),
                      ("lora_tiny_aya_all_linear", "lora")]:
        out, _, _ = run(SINGLE, f"peft={opt}")
        peft = saved(out, "sft_lora").peft
        assert peft.mode == mode
        assert (peft.lora is not None, peft.quant is not None) == (mode != "full", mode == "qlora")


def test_peft_mode_must_match_blocks():
    assert fails(SINGLE, "peft=full_finetune", "peft.lora.r=16")  # partial lora block on full
    assert fails(SINGLE, "peft=qlora_default", "peft.mode=lora")       # lora with a quant block
    assert fails(SINGLE, "peft.mode=adapter")                          # unknown mode


def test_inference_callback_interval():
    out, _, _ = run(SINGLE)
    assert saved(out, "sft_lora").train.inference_callback_interval == 1   # conf/train/default.yaml (the default)
    out, _, _ = run(SINGLE, "train=longrun", "train.inference_callback_interval=0")
    assert saved(out, "sft_lora").train.inference_callback_interval == 0   # every train option sets it
    assert fails(SINGLE, "train.inference_callback_interval=-1")


def test_several_test_sets():
    out, _, _ = run(SINGLE, "data.test.src=[/a.fr,/b.fr]", "data.test.trg=[/a.en,null]")
    test = saved(out, "sft_lora").data.test
    assert list(test.src) == ["/a.fr", "/b.fr"] and list(test.trg) == ["/a.en", None]
    out, _, _ = run(SINGLE, "data.test.src=[/a.fr,/b.fr]", "data.test.trg=null")    # no references
    assert saved(out, "sft_lora").data.test.trg is None
    assert fails(SINGLE, "data.test.src=[/a.fr,/b.fr]", "data.test.trg=[/a.en]")   # length mismatch


def test_lora_target_modules():
    out, _, _ = run(SINGLE, "peft.lora.target_modules=[q_proj,k_proj,v_proj,o_proj]")
    assert list(saved(out, "sft_lora").peft.lora.target_modules) == ["q_proj", "k_proj", "v_proj", "o_proj"]
    out, dirs, _ = run(SINGLE, "peft.lora.target_modules=[q_proj,v_proj],[q_proj,k_proj,v_proj,o_proj]")
    assert dirs == ["sft_lora-0", "sft_lora-1"]                              # a list-valued grid axis
    assert open(os.path.join(out, "sft_lora.grid.txt")).read().splitlines() == [
        "0: peft.lora.target_modules=[q_proj,v_proj]", "1: peft.lora.target_modules=[q_proj,k_proj,v_proj,o_proj]"]
    assert fails(SINGLE, "peft.lora.target_modules=[]")                     # empty
    assert fails(SINGLE, "peft.lora.lora_r=8")                              # old key name


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


# --- CPT config (feature 16_cpt) ---

def cpt_experiment(monotext="[/abs/mono.fr]"):
    """A temp experiment folder: fr-en's dataset.yaml + sft_lora.yaml with `- cpt: default` and a
    data.monotext list. Returns (folder, experiment name)."""
    tmp = tempfile.mkdtemp()
    shutil.copy(os.path.join(os.path.dirname(SINGLE), "dataset.yaml"), tmp)
    text = open(SINGLE).read().replace("  - override peft:", "  - cpt: default\n  - override peft:", 1)
    open(os.path.join(tmp, "cpt_lora.yaml"), "w").write(text + f"\ndata:\n  monotext: {monotext}\n")
    return tmp, "cpt_lora"


def compose_checked(exp_dir, name, *overrides):
    """grid's compose + resolve + check_config + missing-keys check; SystemExit on any problem."""
    grid.register_configs()                                       # grid.main does this for a launch
    with contextlib.redirect_stderr(io.StringIO()):
        return grid._compose_checked(exp_dir, name, list(overrides))


def compose_fails(exp_dir, name, *overrides):
    try:
        compose_checked(exp_dir, name, *overrides)
    except SystemExit:
        return True
    return False


def test_cpt_experiment_composes():
    cfg = compose_checked(*cpt_experiment())
    assert (cfg.cpt.use_monotext, cfg.cpt.use_bitext, cfg.cpt.bitext_format) == (True, True, "separate")
    assert (cfg.cpt.monotext_ratio, cfg.cpt.max_length, cfg.cpt.packing_strategy) == (0.5, 1024, "bfd")
    assert list(cfg.data.monotext) == ["/abs/mono.fr"]
    assert compose_checked(os.path.dirname(SINGLE), "sft_lora").cpt is None    # SFT: no cpt block


def test_cpt_rules():
    exp = cpt_experiment()
    assert compose_fails(*exp, "cpt.bitext_format=both")
    assert compose_fails(*exp, "cpt.packing_strategy=greedy")
    assert compose_fails(*exp, "cpt.use_monotext=false", "cpt.use_bitext=false")   # no data
    assert compose_fails(*exp, "data.monotext=null")                               # use_monotext without files
    assert compose_fails(*exp, "data.monotext=[]")
    assert compose_fails(*exp, "cpt.monotext_ratio=0")                             # both sources: 0 < r < 1
    assert compose_fails(*exp, "cpt.monotext_ratio=1")
    assert compose_fails(*exp, "cpt.use_bitext=false", "cpt.monotext_ratio=abc")   # wrong type
    compose_checked(*exp, "cpt.use_bitext=false", "cpt.monotext_ratio=1")          # one source: ratio unused
    compose_checked(*exp, "cpt.use_monotext=false", "data.monotext=null")          # bitext only: no files needed
    compose_checked(*exp, "cpt.bitext_format=concat", "cpt.packing_strategy=wrapped")


def test_apply_cpt_overrides():
    cfg = compose_checked(*cpt_experiment(), "train=longrun")     # an SFT schedule: decodes, BLEU-selected
    changed = apply_cpt_overrides(cfg)
    assert {key for key, *_ in changed} == set(CPT_FORCED_TRAIN)
    assert all(cfg.train[key] == value for key, (value, _) in CPT_FORCED_TRAIN.items())
    assert apply_cpt_overrides(cfg) == []                         # a saved CPT config: nothing to change
    sft = compose_checked(os.path.dirname(SINGLE), "sft_lora", "train=longrun")
    before = OmegaConf.to_container(sft.train)
    assert apply_cpt_overrides(sft) == [] and OmegaConf.to_container(sft.train) == before


def test_sft_ignores_monotext_in_shared_dataset():
    tmp = tempfile.mkdtemp()                                      # dataset.yaml lists monotext too
    open(os.path.join(tmp, "dataset.yaml"), "w").write(
        open(os.path.join(os.path.dirname(SINGLE), "dataset.yaml")).read() + "monotext:\n  - /abs/mono.fr\n")
    shutil.copy(SINGLE, tmp)
    out, dirs, _ = run(os.path.join(tmp, "sft_lora.yaml"))
    cfg = saved(out, "sft_lora")
    assert list(cfg.data.monotext) == ["/abs/mono.fr"] and cfg.cpt is None


def run_cpt(exp, *overrides, out=None):
    """run() through `launch.sh cpt` (grid --cpt)."""
    out = out or tempfile.mkdtemp()
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
        grid.main(["--cpt", exp, *overrides, f"output_root={out}"])
    dirs = sorted(d for d in os.listdir(out) if os.path.isdir(os.path.join(out, d)))
    return out, dirs, stdout.getvalue()


def fails_cpt(exp, *overrides):
    out = tempfile.mkdtemp()
    try:
        run_cpt(exp, *overrides, out=out)
    except SystemExit:
        return os.listdir(out) == []  # failed before writing anything
    return False


def cpt_path():
    tmp, name = cpt_experiment()
    return os.path.join(tmp, f"{name}.yaml")


def test_cpt_launch_saves_forced_train_keys():
    out, dirs, stdout = run_cpt(cpt_path(), "train=longrun")
    assert dirs == ["sft_lora"] and f"JOBS=({out}/sft_lora)" in stdout
    cfg = saved(out, "sft_lora")
    assert cfg.cpt.bitext_format == "separate" and list(cfg.data.monotext) == ["/abs/mono.fr"]
    assert all(cfg.train[key] == value for key, (value, _) in CPT_FORCED_TRAIN.items())
    assert cfg.train.max_steps == OmegaConf.load(os.path.join(ROOT, "conf/train/longrun.yaml")).max_steps
    assert "KIND" not in stdout                          # launch.sh sets KIND=cpt itself


def test_cpt_and_sft_launches_refuse_each_other():
    assert fails_cpt(SINGLE)                             # an SFT experiment through launch.sh cpt
    assert fails(cpt_path())                             # a CPT experiment through launch.sh sft


def test_cpt_refuses_overrides_of_forced_keys():
    exp = cpt_path()
    assert fails_cpt(exp, "train.packing=false")
    assert fails_cpt(exp, "train.inference_callback_interval=1")
    assert fails_cpt(exp, "train.metric_for_best_model=eval_loss,eval_sp_bleu")   # as a grid axis too
    run_cpt(exp, "train.learning_rate=1e-4")             # other train keys are fine
    run(SINGLE, "train.packing=true")                    # SFT may set them


def test_cpt_sweep():
    out, dirs, _ = run_cpt(cpt_path(), "cpt.bitext_format=separate,concat")
    assert dirs == ["sft_lora-0", "sft_lora-1"]
    assert [saved(out, d).cpt.bitext_format for d in dirs] == ["separate", "concat"]
    assert open(os.path.join(out, "sft_lora.grid.txt")).read().splitlines() == [
        "0: cpt.bitext_format=separate", "1: cpt.bitext_format=concat"]


def test_cpt_resume_and_kind_mismatch():
    out, _, _ = run_cpt(cpt_path())
    run_dir = os.path.join(out, "sft_lora")
    os.makedirs(os.path.join(run_dir, "checkpoint-20"))
    open(os.path.join(run_dir, "checkpoint-20", "trainer_state.json"), "w").close()
    rc = os.path.join(run_dir, "run_config.yaml")
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
        grid.main(["--cpt", rc, "slurm.gres=gpu:l40s:1"])
    assert f"JOBS=({run_dir})" in stdout.getvalue() and "KIND=cpt_resume" in stdout.getvalue()
    for argv in ([rc], ["--cpt", cancelled_run()[1]]):  # sft on a CPT run, cpt on an SFT run
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                grid.main(argv)
            assert False, argv
        except SystemExit:
            pass


def test_inf_and_merge_accept_cpt_runs():
    out, _, _ = run_cpt(cpt_path())
    run_dir = os.path.join(out, "sft_lora")
    os.makedirs(os.path.join(run_dir, "model"))
    rc = os.path.join(run_dir, "run_config.yaml")
    assert f"JOBS=({run_dir}/inf/test)" in inf(rc)
    assert f"JOBS=({run_dir})" in merge(rc)


def test_absolute_checkpoint_must_exist():
    assert fails(SINGLE, "checkpoint=/no/such/model_merged")   # e.g. a CPT run not merged yet
    existing = tempfile.mkdtemp()
    run(SINGLE, f"checkpoint={existing}")                # an existing path and a hub id (the default) pass


def test_configs_saved_before_cpt_still_compose():
    out, _, _ = run(SINGLE)                                       # strip the new keys, as in an old run_config
    rc = os.path.join(out, "sft_lora", "run_config.yaml")
    text = "".join(l for l in open(rc) if not l.startswith(("cpt:", "  monotext:")))
    open(rc, "w").write(text)
    cfg = compose_checked(os.path.dirname(rc), "run_config")
    assert cfg.cpt is None and cfg.data.monotext is None



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


def test_grid_folder_resume_and_inf_jobs_are_named_after_the_folder():
    # So tools/run_status.py can tell one grid folder's queued job from its siblings' (all share run_name).
    out, _, _ = run(SWEEP)
    run_dir = os.path.join(out, "sft_lora_sweep-1")
    rc = os.path.join(run_dir, "run_config.yaml")
    os.makedirs(os.path.join(run_dir, "checkpoint-40"))
    open(os.path.join(run_dir, "checkpoint-40", "trainer_state.json"), "w").close()
    assert "--job-name=sft_lora_sweep-1 " in resume(rc) + " "
    os.makedirs(os.path.join(run_dir, "model"))
    assert "--job-name=sft_lora_sweep-1-inf" in inf(rc)


def test_inf_errors_fail_before_writing():
    run_dir, rc = trained_run()
    for bad in [("decode.num_beams=1,5",),                # comma list
                ("decode.num_bems=5",)]:                  # unknown key
        try:
            inf(rc, *bad)
            assert False, bad
        except SystemExit:
            pass
    try:
        inf(os.path.join(trained_run()[0], "missing.yaml"))   # no such file
        assert False
    except SystemExit:
        pass
    shutil.rmtree(os.path.join(run_dir, "model"))         # no trained model
    try:
        inf(rc)
        assert False
    except SystemExit:
        pass
    assert not os.path.exists(os.path.join(run_dir, "inf"))


def test_inf_on_experiment_decodes_base_checkpoint():
    out = tempfile.mkdtemp()
    stdout = inf(SINGLE, "peft=lora_default", f"output_root={out}")
    inf_dir = os.path.join(out, "sft_lora", "base_ckpt_inf", "test")
    cfg = OmegaConf.load(os.path.join(inf_dir, "inf_config.yaml"))
    assert cfg.model_dir is None and cfg.peft.mode == "lora"
    assert f"JOBS=({inf_dir})" in stdout and "--job-name=sft_lora-base-inf" in stdout
    assert os.listdir(out) == ["sft_lora"] and os.listdir(os.path.join(out, "sft_lora")) == ["base_ckpt_inf"]


def test_inf_on_experiment_refuses_grids():
    for exp, overrides in [(SWEEP, ()), (SINGLE, ("peft.lora.r=8,16",))]:
        out = tempfile.mkdtemp()
        try:
            inf(exp, *overrides, f"output_root={out}")
            assert False, exp
        except SystemExit:
            pass
        assert os.listdir(out) == []


def merge(rc, *overrides):
    """Run grid.main --merge; return stdout. SystemExit propagates."""
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
        grid.main(["--merge", rc, *overrides])
    return stdout.getvalue()


def test_merge_prints_run_folder_and_writes_nothing():
    run_dir, rc = trained_run()
    before = sorted(os.listdir(run_dir))
    stdout = merge(rc, "slurm.gres=gpu:l40s:1")
    assert f"JOBS=({run_dir})" in stdout and "--job-name=sft_lora-merge" in stdout
    assert f"--output={run_dir}/merge-slurm-%A.out" in stdout and "--gres=gpu:l40s:1" in stdout
    assert sorted(os.listdir(run_dir)) == before


def test_merge_refusals():
    run_dir, rc = trained_run()
    for bad in [(rc, "decode.num_beams=5"),              # only slurm.* overrides apply
                (rc, "slurm.gres=a,b"),                  # comma list
                (SINGLE,)]:                              # an experiment, not a run
        try:
            merge(*bad)
            assert False, bad
        except SystemExit:
            pass
    out, _, _ = run(SINGLE, "peft=full_finetune")        # full fine-tune: nothing to merge
    full_dir = os.path.join(out, "sft_lora")
    os.makedirs(os.path.join(full_dir, "model"))
    shutil.rmtree(os.path.join(run_dir, "model"))        # no trained model
    for rc_bad in [os.path.join(full_dir, "run_config.yaml"), rc]:
        try:
            merge(rc_bad)
            assert False, rc_bad
        except SystemExit:
            pass


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
