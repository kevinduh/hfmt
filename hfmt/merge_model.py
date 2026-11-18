"""Merge a trained run's lora/qlora adapter into its base model (feature 10).

    python -m hfmt.merge_model <run folder>

Normally run as a Slurm job by ``bash egs/launch.sh merge <run folder>/run_config.yaml``. Reads the
run's saved ``run_config.yaml`` and ``model/`` and writes ``<run folder>/model_merged/``: full bf16
weights + tokenizer + ``merge_info.yaml`` (where it came from). ``model_merged/`` can then be an
experiment's ``checkpoint`` for another round of SFT, or decoded as is with
``launch.sh inf <experiment>.yaml``. qlora: see ``mt_model.merge_trained_model`` (dequantized base).
An existing ``model_merged/`` written by this script is replaced (with a warning).
"""

import logging
import os
import shutil
import sys

from omegaconf import OmegaConf

from hfmt.mt_model import merge_trained_model


def main(run_dir):
    import torch

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
    run_dir = os.path.abspath(run_dir)
    cfg = OmegaConf.load(os.path.join(run_dir, "run_config.yaml"))
    model_dir, out_dir = os.path.join(run_dir, "model"), os.path.join(run_dir, "model_merged")
    if os.path.exists(out_dir):
        if not os.path.isfile(os.path.join(out_dir, "merge_info.yaml")):
            sys.exit(f"merge_model: {out_dir} exists but wasn't written by merge_model; not touching it")
        logging.warning(f"{out_dir} exists; replacing it")
        shutil.rmtree(out_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Merging {model_dir} (peft.mode={cfg.peft.mode}) into {cfg.checkpoint} on {device}")
    warning = merge_trained_model(cfg, model_dir, out_dir, device)

    git_rev_file = os.path.join(run_dir, "git_rev")
    info = {"source_run": run_dir, "checkpoint": cfg.checkpoint, "peft_mode": cfg.peft.mode,
            "run_git_rev": open(git_rev_file).read().strip() if os.path.isfile(git_rev_file) else None,
            "warning": warning}
    OmegaConf.save(OmegaConf.create(info), os.path.join(out_dir, "merge_info.yaml"))
    logging.info(f"Wrote {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m hfmt.merge_model <run folder>")
    main(sys.argv[1])
