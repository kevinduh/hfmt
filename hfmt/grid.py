"""Expand an experiment into jobs and prepare their run folders (login node; features 08, 09).

Called by egs/launch.sh sft:  python -m hfmt.grid <experiment.yaml> [overrides...]
     and egs/launch.sh inf:  python -m hfmt.grid --inf <run folder>/run_config.yaml | <experiment.yaml> [overrides...]
     and egs/launch.sh merge: python -m hfmt.grid --merge <run folder>/run_config.yaml [slurm.* overrides...]

Training:
- Grid = the experiment's ``hydra.sweeper.params`` plus command-line comma lists, expanded with
  Hydra's own BasicSweeper rules (a single command-line value pins an axis). No grid -> 1 job.
- Every job's config is composed (so typos fail here, before anything is submitted) and saved
  fully resolved as ``<job folder>/run_config.yaml``, with ``git_rev`` next to it.
- Job folders: ``<output_root>/<run_name>`` for 1 job, ``<output_root>/<run_name>-<idx>`` for
  more; ``output_root`` defaults to ``<experiment folder>/outputs``. Re-running a saved
  ``run_config.yaml`` without a new ``run_name`` uses the folder it's in.

Inference (``--inf``): always one job, one model; the yaml kind decides which model.
- A trained run's ``run_config.yaml``: its saved config + overrides + ``model_dir=<run folder>/model``
  are checked the same way and saved as ``<run folder>/inf/<inf_name>/inf_config.yaml``;
  ``run_config.yaml`` is never written.
- An experiment yaml (not a sweep): the downloaded ``checkpoint`` as is, no adapter (``model_dir``
  unset); saved as ``<output_root>/<run_name>/base_ckpt_inf/<inf_name>/inf_config.yaml``.

Merge (``--merge``): one job on one trained lora/qlora run folder; checks it and prints the job
(the run folder itself), writing nothing. The job writes ``<run folder>/model_merged/``.

Human-readable output goes to stderr; stdout is bash for egs/launch.sh to eval
(``JOBS=(...)`` and ``SBATCH_ARGS=(...)``).

Relies on Hydra internals (OverridesParser, BasicSweeper.split_arguments); tested on 1.3.7.
"""

import os
import shlex
import subprocess
import sys

from hydra import compose, initialize_config_dir
from hydra._internal.core_plugins.basic_sweeper import BasicSweeper
from hydra.core.override_parser.overrides_parser import OverridesParser
from omegaconf import OmegaConf

from hfmt.hydra_config import check_config, register_configs

SAVED = "run_config"
INF_SAVED = "inf_config"


def _log(msg):
    print(msg, file=sys.stderr)


def _compose(exp_dir, exp_name, overrides, with_hydra=False):
    conf_dir = os.path.join(os.environ["HFMT_ROOT"], "conf")
    try:
        with initialize_config_dir(config_dir=exp_dir, version_base=None):
            return compose(exp_name, overrides=overrides + [f"hydra.searchpath=[file://{conf_dir}]"],
                           return_hydra_config=with_hydra)
    except Exception as e:  # Hydra/OmegaConf errors: show the message, not a traceback
        hint = "\n(a key inside a block this config leaves unset, e.g. peft.lora with peft=full_finetune)" \
            if "NoneType" in str(e) else ""
        sys.exit(f"grid: {' '.join(overrides)}\n{e}{hint}")


def _compose_checked(exp_dir, exp_name, job):
    """Compose one job, resolve it and check it; exit with a clear message on any problem."""
    cfg = _compose(exp_dir, exp_name, job)
    try:
        OmegaConf.resolve(cfg)
        check_config(cfg)
    except Exception as e:
        hint = (f"\nCreate {exp_dir}/local_setting.sh (see local_setting.sh.example)"
                if "HFMT_DATA_ROOT" in str(e) else "")
        sys.exit(f"grid: {' '.join(job)}\n{e}{hint}")
    missing = OmegaConf.missing_keys(cfg)
    if missing:
        sys.exit(f"grid: missing values: {sorted(missing)}")
    return cfg


def _git_rev():
    root = os.environ["HFMT_ROOT"]
    rev = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", root, "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    return rev + ("-dirty" if dirty else "")


def _save(job_dir, name, cfg, git_rev):
    """Write ``<job_dir>/<name>.yaml`` (resolved; re-validated by the schema when loaded) and ``git_rev``."""
    if os.path.exists(job_dir):
        _log(f"WARNING: {job_dir} exists; reusing it")
    os.makedirs(job_dir, exist_ok=True)
    with open(os.path.join(job_dir, f"{name}.yaml"), "w") as f:
        f.write("defaults:\n  - base_config\n  - _self_\n\n" + OmegaConf.to_yaml(cfg))
    with open(os.path.join(job_dir, "git_rev"), "w") as f:
        f.write(git_rev + "\n")


def _print_jobs(job_dirs, s, job_name, log):
    """The bash that egs/launch.sh evals: job folders and sbatch options from the slurm: block."""
    sbatch = [f"--array=0-{len(job_dirs) - 1}%{s.array_parallelism}", f"--partition={s.partition}",
              f"--gres={s.gres}", f"--time={s.time}", f"--mem={s.mem_gb}G", f"--cpus-per-task={s.cpus}",
              f"--job-name={job_name}", f"--output={log}"]
    if s.constraint:
        sbatch.append(f"--constraint={s.constraint}")
    if s.exclude:
        sbatch.append(f"--exclude={s.exclude}")
    print("JOBS=(" + " ".join(shlex.quote(d) for d in job_dirs) + ")")
    print("SBATCH_ARGS=(" + " ".join(shlex.quote(a) for a in sbatch) + ")")


def main_inf(exp_dir, exp_name, args, parsed):
    """``--inf``: prepare one inference job. The yaml kind decides which model is decoded:
    a trained run's ``run_config.yaml`` -> its ``model/``; an experiment yaml -> its plain ``checkpoint``."""
    # 1. One model per call: no grids (loop over run folders in the shell instead).
    if any(o.is_sweep_override() for o in parsed):
        sys.exit("grid --inf: comma lists aren't allowed (one model per call; loop in the shell)")

    if exp_name == SAVED:
        # 2a. Trained run: decode <run>/model; output goes inside the run folder.
        model_dir = os.path.join(exp_dir, "model")
        if not os.path.isdir(model_dir):
            sys.exit(f"grid --inf: no trained model in {model_dir}")
        cfg = _compose_checked(exp_dir, exp_name, [f"model_dir={model_dir}"] + args)
        inf_dir = os.path.join(exp_dir, "inf", cfg.inf_name)
        job_name = f"{cfg.run_name}-inf"
    else:
        # 2b. Experiment: decode the checkpoint as downloaded (model_dir stays unset; the peft
        # option only sets the load precision). A sweep file would silently use its base values,
        # so refuse it. Output: <output_root>/<run_name>/base_ckpt_inf/<inf_name>, separate from
        # any trained run's inf/ in the same folder.
        if _compose(exp_dir, exp_name, args, with_hydra=True).hydra.sweeper.params:
            sys.exit(f"grid --inf: {exp_name}.yaml is a sweep; use a single-run experiment")
        cfg = _compose_checked(exp_dir, exp_name, args)
        if cfg.output_root is None:
            cfg.output_root = os.path.join(exp_dir, "outputs")
        inf_dir = os.path.join(cfg.output_root, cfg.run_name, "base_ckpt_inf", cfg.inf_name)
        job_name = f"{cfg.run_name}-base-inf"

    # 3. Save the checked config where the job will read it, then print the bash for launch.sh.
    _save(inf_dir, INF_SAVED, cfg, _git_rev())
    _log(f"  0: {inf_dir}")
    _print_jobs([inf_dir], cfg.slurm, job_name, os.path.join(inf_dir, "slurm-%A.out"))


def main_merge(exp_dir, exp_name, args, parsed):
    """``--merge``: prepare one job that merges a trained run's adapter (``hfmt/merge_model.py``).
    Writes no config: the job reads the run's ``run_config.yaml`` as is."""
    # 1. Only a trained lora/qlora run folder; only slurm.* overrides (the job ignores anything else).
    if exp_name != SAVED:
        sys.exit(f"grid --merge: give a trained run folder's {SAVED}.yaml, not {exp_name}.yaml")
    if any(o.is_sweep_override() for o in parsed):
        sys.exit("grid --merge: comma lists aren't allowed (one run folder per call; loop in the shell)")
    if any(not o.get_key_element().startswith("slurm.") for o in parsed):
        sys.exit("grid --merge: only slurm.* overrides apply (the merge uses the run's saved config)")
    if not os.path.isdir(os.path.join(exp_dir, "model")):
        sys.exit(f"grid --merge: no trained model in {exp_dir}/model")
    cfg = _compose_checked(exp_dir, exp_name, args)
    if cfg.peft.mode == "full":
        sys.exit("grid --merge: peft.mode=full has no adapter; its model/ already holds full weights "
                 "(usable as a checkpoint directly)")

    # 2. Print the bash for launch.sh: the job folder is the run folder itself.
    _log(f"  0: {exp_dir} -> {exp_dir}/model_merged")
    _print_jobs([exp_dir], cfg.slurm, f"{cfg.run_name}-merge", os.path.join(exp_dir, "merge-slurm-%A.out"))


def main(argv):
    # Which flow: sft (default), --inf or --merge (see the module docstring).
    mode = argv[0] if argv[:1] in (["--inf"], ["--merge"]) else None
    if mode:
        argv = argv[1:]
    exp_path, args = os.path.abspath(argv[0]), argv[1:]
    exp_dir, exp_name = os.path.dirname(exp_path), os.path.basename(exp_path).removesuffix(".yaml")
    if exp_name == "config":
        sys.exit(f"grid: {exp_path}: an experiment can't be named config.yaml (clashes with conf/config.yaml)")
    if not os.path.isfile(exp_path):
        sys.exit(f"grid: experiment not found: {exp_path}")
    register_configs()

    parser = OverridesParser.create()
    parsed = parser.parse_overrides(args)
    if mode == "--inf":
        return main_inf(exp_dir, exp_name, args, parsed)
    if mode == "--merge":
        return main_merge(exp_dir, exp_name, args, parsed)

    # Grid: file axes first, command-line overrides after (a later value for a key replaces it).
    plain = [a for a, o in zip(args, parsed) if not o.is_sweep_override()]
    base = _compose(exp_dir, exp_name, plain, with_hydra=True)
    params = [f"{k}={v}" for k, v in (base.hydra.sweeper.params or {}).items()]
    all_overrides = parser.parse_overrides(params + args)
    jobs = BasicSweeper.split_arguments(all_overrides, None)[0]
    swept = {o.get_key_element(): o.is_sweep_override() for o in all_overrides}
    swept = [k for k, is_sweep in swept.items() if is_sweep]
    if any(k.startswith("slurm.") for k in swept):
        sys.exit("grid: slurm.* can't be a grid axis (one sbatch array per launch)")

    # Compose and check every job before writing anything.
    cfgs = [_compose_checked(exp_dir, exp_name, job) for job in jobs]

    n, width = len(cfgs), len(str(len(cfgs) - 1))
    run_name = cfgs[0].run_name
    output_root = cfgs[0].output_root or os.path.join(exp_dir, "outputs")
    reuse_dir = exp_name == SAVED and n == 1 and not any(a.lstrip("+~").startswith("run_name=") for a in args)
    git_rev = _git_rev()
    job_dirs, grid_lines = [], []
    for idx, (job, cfg) in enumerate(zip(jobs, cfgs)):
        name = cfg.run_name if n == 1 else f"{cfg.run_name}-{idx:0{width}d}"
        job_dir = exp_dir if reuse_dir else os.path.join(cfg.output_root or output_root, name)
        if cfg.output_root is None:
            cfg.output_root = output_root
        _save(job_dir, SAVED, cfg, git_rev)
        axes = " ".join(o for o in job if o.split("=", 1)[0] in swept)
        grid_lines.append(f"{idx}: {axes}")
        job_dirs.append(job_dir)
        _log(f"  {idx}: {job_dir}  {axes}")

    if n > 1:
        with open(os.path.join(output_root, f"{run_name}.grid.txt"), "w") as f:
            f.write("\n".join(grid_lines) + "\n")
    _log(f"{n} job(s)")

    log = os.path.join(job_dirs[0], "slurm-%A.out") if n == 1 else \
        os.path.join(output_root, f"{run_name}-%{width}a" if width > 1 else f"{run_name}-%a", "slurm-%A_%a.out")
    _print_jobs(job_dirs, cfgs[0].slurm, run_name, log)


if __name__ == "__main__":
    main(sys.argv[1:])
