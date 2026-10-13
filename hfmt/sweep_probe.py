"""Compose-only probe that turns an experiment's inline ``sweep`` block into Hydra
multirun overrides for ``egs/run.sh`` (feature 04).

Hydra's basic sweeper has no on/off switch and *rejects* ``hydra.sweeper.params.*``
passed on the command line, so the sweep toggle lives in our typed
``sweep: {enabled, params}`` config (``hfmt/hydra_config.py``) and ``egs/run.sh``
bridges it: this script composes the chosen experiment and, when ``sweep.enabled``,
prints one bare ``<axis>=<csv>`` override per param. The basic sweeper then fans those
comma-lists out into the job grid. Disabled -> prints nothing -> a single job.

It is deliberately compose-only and dependency-light (``hydra`` + ``omegaconf`` + the
config schema, **never** ``transformers``) so ``run.sh`` can call it quickly on a
login node before the heavy training launch.

Confidentiality (CLAUDE.md): it emits only sweep axis names/values -- hyperparameters
or config-file names -- never dataset content.

Usage (normally via ``egs/run.sh``):
    python hfmt/sweep_probe.py <experiment>
"""

import os
import shlex
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

# Sibling import when run as a script (sys.path[0] == hfmt/); package-qualified when
# HFMT_ROOT is on PYTHONPATH. Mirrors hfmt/sft_translation.py.
try:
    from hydra_config import register_configs
except ModuleNotFoundError:
    from hfmt.hydra_config import register_configs


def sweep_overrides(experiment, conf_dir, extra_overrides=None):
    """Return the list of bare ``<axis>=<csv>`` overrides for an experiment.

    ``extra_overrides`` are additional ``sweep.*`` CLI overrides (e.g.
    ``sweep.enabled=true``) forwarded by ``run.sh`` so the toggle can be flipped on
    the command line. Only ``sweep.*`` overrides are passed here -- ad-hoc comma-list
    axis overrides (``lora.lora_r=8,16``) would fail this single-run, type-validated
    compose, and they don't need the probe anyway (the sweeper expands them directly).

    Returns an empty list when the ``sweep`` block is absent or disabled.
    """
    # hfmt_runlabel is defined in sft_translation.py (which imports transformers); we
    # never read hydra.sweep.subdir here, so a no-op stand-in keeps compose() happy
    # without pulling in the training deps.
    OmegaConf.register_new_resolver(
        "hfmt_runlabel", lambda *a, **k: "probe", replace=True
    )
    register_configs()

    overrides = [f"+experiment={experiment}"] + list(extra_overrides or [])
    with initialize_config_dir(config_dir=conf_dir, version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    sweep = cfg.get("sweep", None)
    if not sweep or not sweep.get("enabled", False):
        return []
    return [f"{axis}={values}" for axis, values in sweep.get("params", {}).items()]


def main(argv):
    if len(argv) < 1:
        sys.stderr.write("usage: sweep_probe.py <experiment> [sweep.key=value ...]\n")
        return 2
    experiment = argv[0]
    extra_overrides = argv[1:]

    hfmt_root = os.environ.get("HFMT_ROOT")
    if not hfmt_root:
        sys.stderr.write("sweep_probe.py: HFMT_ROOT is not set\n")
        return 2

    overrides = sweep_overrides(
        experiment, os.path.join(hfmt_root, "conf"), extra_overrides
    )
    if overrides:
        # One line so `EXTRA=$(...)` word-splits into separate args in run.sh.
        print(" ".join(shlex.quote(tok) for tok in overrides))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
