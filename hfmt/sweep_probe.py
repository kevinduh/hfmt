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

It also fast-fails on the login node (before the heavy Slurm submit) when the experiment's
named ``dataset.yaml`` template is missing -- a cheap existence check here, with the full
validation (suffix/basename/inflation) done authoritatively in ``hfmt/sft_translation.py``.

Confidentiality (CLAUDE.md): it emits only sweep axis names/values -- hyperparameters
or config-file names -- never dataset content.

Usage (normally via ``egs/run.sh``):
    python hfmt/sweep_probe.py <group>=<name> [data.dataset_dir=<dir>] [sweep.key=value ...]
e.g. ``python hfmt/sweep_probe.py mmtc/fr-en=sft1 data.dataset_dir=$HFMT_ROOT/egs/mmtc/fr-en``
(feature 06: the experiment is a nested config group selected as ``+<group>=<name>``).
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


class LanguagePairMismatchError(ValueError):
    """The experiment's ``data.src``/``data.trg`` don't match its ``<src>-<trg>`` directory."""


def sweep_overrides(selector, conf_dir, extra_overrides=None):
    """Return the list of bare ``<axis>=<csv>`` overrides for an experiment.

    ``selector`` is the nested experiment group choice ``<group>=<name>`` (feature 06), composed
    as ``+<group>=<name>`` -- e.g. ``mmtc/fr-en=sft1``. ``extra_overrides`` are additional CLI
    overrides forwarded by ``run.sh``: the injected ``data.dataset_dir=<dir>`` and any
    ``sweep.*`` toggle (e.g. ``sweep.enabled=true``). Only these are passed here -- ad-hoc
    comma-list axis overrides (``lora.lora_r=8,16``) would fail this single-run, type-validated
    compose, and they don't need the probe anyway (the sweeper expands them directly).

    Also fast-fails if the experiment's named ``dataset.yaml`` template does not exist, or if its
    ``data.src``/``data.trg`` don't match the ``<src>-<trg>`` directory it lives in (cheap
    login-node checks; full validation lives in ``hfmt/sft_translation.py``).

    Returns an empty list when the ``sweep`` block is absent or disabled.
    """
    # hfmt_runlabel is defined in sft_translation.py (which imports transformers); we
    # never read hydra.sweep.subdir here, so a no-op stand-in keeps compose() happy
    # without pulling in the training deps.
    OmegaConf.register_new_resolver(
        "hfmt_runlabel", lambda *a, **k: "probe", replace=True
    )
    register_configs()

    overrides = [f"+{selector}"] + list(extra_overrides or [])
    with initialize_config_dir(config_dir=conf_dir, version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    # Fast-fail: the named dataset template must exist. Subset of sft_translation's authoritative
    # validation -- just existence, so a missing/mistyped template aborts before the Slurm submit.
    data = cfg.get("data", None)
    if data is not None and not OmegaConf.is_missing(data, "dataset_dir"):
        template = os.path.join(str(data.dataset_dir), str(data.get("dataset_file", "dataset.yaml")))
        if not os.path.isfile(template):
            raise FileNotFoundError(f"dataset template not found: {template}")
        # Fast-fail: the lang pair must match the directory (mirrors resolve_dataset_context in
        # sft_translation.py). It becomes a W&B tag, so a copied yaml must not mislabel runs.
        if not (OmegaConf.is_missing(data, "src") or OmegaConf.is_missing(data, "trg")):
            dataset_dir = os.path.normpath(str(data.dataset_dir))
            lang_pair = f"{data.src}-{data.trg}"
            if os.path.basename(dataset_dir) != lang_pair:
                raise LanguagePairMismatchError(
                    f"data.src/data.trg ({lang_pair!r}) do not match the experiment's directory "
                    f"{dataset_dir!r}; fix them in the experiment yaml."
                )

    sweep = cfg.get("sweep", None)
    if not sweep or not sweep.get("enabled", False):
        return []
    return [f"{axis}={values}" for axis, values in sweep.get("params", {}).items()]


def main(argv):
    if len(argv) < 1:
        sys.stderr.write(
            "usage: sweep_probe.py <group>=<name> [data.dataset_dir=<dir>] [sweep.key=value ...]\n"
        )
        return 2
    selector = argv[0]
    extra_overrides = argv[1:]

    hfmt_root = os.environ.get("HFMT_ROOT")
    if not hfmt_root:
        sys.stderr.write("sweep_probe.py: HFMT_ROOT is not set\n")
        return 2

    try:
        overrides = sweep_overrides(
            selector, os.path.join(hfmt_root, "conf"), extra_overrides
        )
    except (FileNotFoundError, LanguagePairMismatchError) as e:
        # Fast-fail (login node): abort before the heavy submit with a clear message. Returning
        # non-zero makes run.sh's `$(...)` + `set -e` stop the launch.
        sys.stderr.write(f"sweep_probe.py: {e}\n")
        return 2
    if overrides:
        # One line so `EXTRA=$(...)` word-splits into separate args in run.sh.
        print(" ".join(shlex.quote(tok) for tok in overrides))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
