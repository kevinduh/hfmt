"""Expand an experiment into jobs and prepare their run folders (login node; feature 08).

Called by egs/sft.sh:  python -m hfmt.grid <experiment.yaml> [overrides...]

- Grid = the experiment's ``hydra.sweeper.params`` plus command-line comma lists, expanded with
  Hydra's own BasicSweeper rules (a single command-line value pins an axis). No grid -> 1 job.
- Every job's config is composed (so typos fail here, before anything is submitted) and saved
  fully resolved as ``<job folder>/run_config.yaml``, with ``git_rev`` next to it.
- Job folders: ``<output_root>/<run_name>`` for 1 job, ``<output_root>/<run_name>-<idx>`` for
  more; ``output_root`` defaults to ``<experiment folder>/outputs``. Re-running a saved
  ``run_config.yaml`` without a new ``run_name`` uses the folder it's in.
- Human-readable output goes to stderr; stdout is bash for sft.sh to eval
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


def _git_rev():
    root = os.environ["HFMT_ROOT"]
    rev = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", root, "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    return rev + ("-dirty" if dirty else "")


def main(argv):
    exp_path, args = os.path.abspath(argv[0]), argv[1:]
    exp_dir, exp_name = os.path.dirname(exp_path), os.path.basename(exp_path).removesuffix(".yaml")
    if exp_name == "config":
        sys.exit(f"grid: {exp_path}: an experiment can't be named config.yaml (clashes with conf/config.yaml)")
    if not os.path.isfile(exp_path):
        sys.exit(f"grid: experiment not found: {exp_path}")
    register_configs()

    # Grid: file axes first, command-line overrides after (a later value for a key replaces it).
    parser = OverridesParser.create()
    parsed = parser.parse_overrides(args)
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
    cfgs = []
    for job in jobs:
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
        cfgs.append(cfg)

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
        if os.path.exists(job_dir):
            _log(f"WARNING: {job_dir} exists; reusing it")
        os.makedirs(job_dir, exist_ok=True)
        with open(os.path.join(job_dir, f"{SAVED}.yaml"), "w") as f:
            f.write("defaults:\n  - base_config\n  - _self_\n\n" + OmegaConf.to_yaml(cfg))
        with open(os.path.join(job_dir, "git_rev"), "w") as f:
            f.write(git_rev + "\n")
        axes = " ".join(o for o in job if o.split("=", 1)[0] in swept)
        grid_lines.append(f"{idx}: {axes}")
        job_dirs.append(job_dir)
        _log(f"  {idx}: {job_dir}  {axes}")

    if n > 1:
        with open(os.path.join(output_root, f"{run_name}.grid.txt"), "w") as f:
            f.write("\n".join(grid_lines) + "\n")
    _log(f"{n} job(s)")

    s = cfgs[0].slurm
    log = os.path.join(job_dirs[0], "slurm-%A.out") if n == 1 else \
        os.path.join(output_root, f"{run_name}-%{width}a" if width > 1 else f"{run_name}-%a", "slurm-%A_%a.out")
    sbatch = [f"--array=0-{n - 1}%{s.array_parallelism}", f"--partition={s.partition}", f"--gres={s.gres}",
              f"--time={s.time}", f"--mem={s.mem_gb}G", f"--cpus-per-task={s.cpus}",
              f"--job-name={run_name}", f"--output={log}"]
    print("JOBS=(" + " ".join(shlex.quote(d) for d in job_dirs) + ")")
    print("SBATCH_ARGS=(" + " ".join(shlex.quote(a) for a in sbatch) + ")")


if __name__ == "__main__":
    main(sys.argv[1:])
