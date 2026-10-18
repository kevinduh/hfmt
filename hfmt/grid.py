"""Turn a yaml + overrides into checked job folders for egs/launch.sh (login node; features 08-10).

egs/launch.sh calls this once per launch, then submits what it prints. Three flows:

  launch.sh sft <experiment.yaml | run_config.yaml> [overrides]  ->  main_sft
      python -m hfmt.grid <yaml> [overrides...]
      Expands the grid, composes + checks every job, writes each job's run folder:
        <output_root>/<run_name>[-<idx>]/run_config.yaml (+ git_rev), and <run_name>.grid.txt for a grid.
  launch.sh inf <run_config.yaml | experiment.yaml> [overrides]  ->  main_inf
      python -m hfmt.grid --inf <yaml> [overrides...]
      One job. The yaml kind decides which model gets decoded:
        run_config.yaml  -> the run's trained model/; writes <run folder>/inf/<inf_name>/inf_config.yaml
        experiment yaml  -> its checkpoint as downloaded;
                            writes <output_root>/<run_name>/base_ckpt_inf/<inf_name>/inf_config.yaml
  launch.sh merge <run_config.yaml> [slurm.* overrides]          ->  main_merge
      python -m hfmt.grid --merge <yaml> [slurm.* overrides...]
      One job on a trained lora/qlora run; checks it and writes nothing (the job reads run_config.yaml
      and writes <run folder>/model_merged/).

What it prints. Human-readable lines go to stderr. stdout is bash that launch.sh evals: the job
folders, and the sbatch options built from the config's slurm: block. For example:
    JOBS=(/.../outputs/sft_lora_sweep-0 /.../outputs/sft_lora_sweep-1)
    SBATCH_ARGS=(--array=0-1%4 --partition=gpu --gres=gpu:a100:1 ... --job-name=sft_lora_sweep ...)
launch.sh then runs `egs/job.sh <kind> "${JOBS[@]}"` as one sbatch array (or LOCAL=1), and array
task i works on JOBS[i].

Training details:
- Grid = the experiment's ``hydra.sweeper.params`` plus command-line comma lists, expanded with
  Hydra's own BasicSweeper rules (a single command-line value pins an axis). No grid -> 1 job.
- Every job is composed and checked before anything is written, so a typo fails here, not on a GPU.
  Saved configs are fully resolved (no ${...} left), so jobs don't need local_setting.sh.
- ``output_root`` defaults to ``<experiment folder>/outputs``. Re-running a saved
  ``run_config.yaml`` without a new ``run_name`` reuses the folder it's in.
- An existing job folder only warns (no timestamps): its files are overwritten.

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

SAVED = "run_config"        # a trained run's saved config: <run folder>/run_config.yaml
INF_SAVED = "inf_config"    # an inference job's saved config: <inference folder>/inf_config.yaml


def _log(msg):
    """Human-readable output: stderr, so stdout stays pure bash for launch.sh."""
    print(msg, file=sys.stderr)


def _compose(exp_dir, exp_name, overrides, with_hydra=False):
    """Compose ``<exp_dir>/<exp_name>.yaml`` + overrides the way ``@hydra.main`` would, with conf/ on
    the search path (for ``- config`` and the groups). ``with_hydra`` also returns the ``hydra:``
    node, needed to read ``hydra.sweeper.params``. Exits with Hydra's message on an error."""
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
    """Compose one job, resolve all ${...} and run the schema's check_config. Exits with a clear
    message on any problem, including keys no yaml sets (MISSING). Used by every flow."""
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
    """Commit of the code that will run, with "-dirty" for uncommitted changes. Saved next to each config."""
    root = os.environ["HFMT_ROOT"]
    rev = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", root, "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    return rev + ("-dirty" if dirty else "")


def _save(job_dir, name, cfg, git_rev):
    """Write ``<job_dir>/<name>.yaml`` and ``git_rev``. The yaml starts with a defaults list naming
    only the schema, so the job loads it as a complete config that the schema checks again."""
    if os.path.exists(job_dir):
        _log(f"WARNING: {job_dir} exists; reusing it")
    os.makedirs(job_dir, exist_ok=True)
    with open(os.path.join(job_dir, f"{name}.yaml"), "w") as f:
        f.write("defaults:\n  - base_config\n  - _self_\n\n" + OmegaConf.to_yaml(cfg))
    with open(os.path.join(job_dir, "git_rev"), "w") as f:
        f.write(git_rev + "\n")


def _print_jobs(job_dirs, s, job_name, log):
    """Print the bash that launch.sh evals (see the module docstring): JOBS = job folders, in array
    order; SBATCH_ARGS = sbatch options from the slurm: block ``s``. ``log`` is sbatch's --output
    pattern (%A = job id, %a = array index)."""
    sbatch = [f"--array=0-{len(job_dirs) - 1}%{s.array_parallelism}", f"--partition={s.partition}",
              f"--gres={s.gres}", f"--time={s.time}", f"--mem={s.mem_gb}G", f"--cpus-per-task={s.cpus}",
              f"--job-name={job_name}", f"--output={log}"]
    if s.constraint:
        sbatch.append(f"--constraint={s.constraint}")
    if s.exclude:
        sbatch.append(f"--exclude={s.exclude}")
    print("JOBS=(" + " ".join(shlex.quote(d) for d in job_dirs) + ")")
    print("SBATCH_ARGS=(" + " ".join(shlex.quote(a) for a in sbatch) + ")")


def main_sft(exp_dir, exp_name, args, parsed, parser):
    """Training: expand the grid, check every job, write one run folder per job."""
    # 1. Read the experiment's own grid. hydra.sweeper.params is only visible after composing, and a
    #    comma list can't be composed, so compose once with the single-value overrides only.
    plain = [a for a, o in zip(args, parsed) if not o.is_sweep_override()]
    base = _compose(exp_dir, exp_name, plain, with_hydra=True)
    params = [f"{k}={v}" for k, v in (base.hydra.sweeper.params or {}).items()]

    # 2. Expand: file axes first, then the command line, so a later value for the same key wins
    #    (a single value pins a file axis; a comma list replaces or adds one). Each job is a list of
    #    single-value overrides. `swept` = keys that vary across jobs (for the grid.txt labels).
    all_overrides = parser.parse_overrides(params + args)
    jobs = BasicSweeper.split_arguments(all_overrides, None)[0]
    swept = {o.get_key_element(): o.is_sweep_override() for o in all_overrides}
    swept = [k for k, is_sweep in swept.items() if is_sweep]
    if any(k.startswith("slurm.") for k in swept):
        sys.exit("grid: slurm.* can't be a grid axis (one sbatch array per launch)")

    # 3. Compose and check every job before writing anything.
    cfgs = [_compose_checked(exp_dir, exp_name, job) for job in jobs]

    # 4. Choose job folders: <run_name> for one job, <run_name>-<idx> for a grid (idx zero-padded so
    #    folders sort in order). Re-running a saved run_config.yaml keeps its folder unless run_name changes.
    n, width = len(cfgs), len(str(len(cfgs) - 1))
    run_name = cfgs[0].run_name
    output_root = cfgs[0].output_root or os.path.join(exp_dir, "outputs")
    reuse_dir = exp_name == SAVED and n == 1 and not any(a.lstrip("+~").startswith("run_name=") for a in args)
    git_rev = _git_rev()
    job_dirs, grid_lines = [], []
    for idx, (job, cfg) in enumerate(zip(jobs, cfgs)):
        name = cfg.run_name if n == 1 else f"{cfg.run_name}-{idx:0{width}d}"
        job_dir = exp_dir if reuse_dir else os.path.join(cfg.output_root or output_root, name)
        if cfg.output_root is None:      # save the resolved root, so a re-run lands next to it
            cfg.output_root = output_root
        # 5. Save each job's config + git_rev; note which grid values it got.
        _save(job_dir, SAVED, cfg, git_rev)
        axes = " ".join(o for o in job if o.split("=", 1)[0] in swept)
        grid_lines.append(f"{idx}: {axes}")
        job_dirs.append(job_dir)
        _log(f"  {idx}: {job_dir}  {axes}")

    if n > 1:                            # idx -> values, next to the job folders
        with open(os.path.join(output_root, f"{run_name}.grid.txt"), "w") as f:
            f.write("\n".join(grid_lines) + "\n")
    _log(f"{n} job(s)")

    # 6. Print the bash for launch.sh. Slurm logs go inside each job folder: %a is the array index,
    #    zero-padded (%2a, ...) to match the folder names.
    log = os.path.join(job_dirs[0], "slurm-%A.out") if n == 1 else \
        os.path.join(output_root, f"{run_name}-%{width}a" if width > 1 else f"{run_name}-%a", "slurm-%A_%a.out")
    _print_jobs(job_dirs, cfgs[0].slurm, run_name, log)


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
    """Parse ``[--inf|--merge] <yaml> [overrides...]`` and hand off to the flow's function."""
    mode = argv[0] if argv[:1] in (["--inf"], ["--merge"]) else None
    if mode:
        argv = argv[1:]
    exp_path, args = os.path.abspath(argv[0]), argv[1:]
    exp_dir, exp_name = os.path.dirname(exp_path), os.path.basename(exp_path).removesuffix(".yaml")
    if exp_name == "config":
        sys.exit(f"grid: {exp_path}: an experiment can't be named config.yaml (clashes with conf/config.yaml)")
    if not os.path.isfile(exp_path):
        sys.exit(f"grid: experiment not found: {exp_path}")
    register_configs()                   # the schema, as Hydra's `base_config`

    parser = OverridesParser.create()
    parsed = parser.parse_overrides(args)
    if mode == "--inf":
        return main_inf(exp_dir, exp_name, args, parsed)
    if mode == "--merge":
        return main_merge(exp_dir, exp_name, args, parsed)
    return main_sft(exp_dir, exp_name, args, parsed, parser)


if __name__ == "__main__":
    main(sys.argv[1:])
