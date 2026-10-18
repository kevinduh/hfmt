"""Decode test sets with a trained model or a downloaded checkpoint (Hydra-driven, features 09, 10).

Normally launched by `egs/launch.sh inf`, which saves the job's config as inf_config.yaml:
- a trained run's run_config.yaml -> <run folder>/inf/<inf_name>/; model_dir=<run folder>/model,
  decoded on the base it was trained on, so base model, peft mode, prompt and decode settings match training;
- an experiment yaml -> <output_root>/<run_name>/base_ckpt_inf/<inf_name>/; model_dir unset, the
  checkpoint is decoded as downloaded (zero-shot), at the precision of the experiment's peft option.
Direct run (debugging):
    python -m hfmt.inf_translation --config-dir <run folder> --config-name run_config \\
        model_dir=<run folder>/model hydra.run.dir=<output folder> [overrides]
e.g. decode.num_beams=5, or other test sets: 'data.test.src=[a,b]' 'data.test.trg=[a_ref,null]'.

Each test file is decoded on its own into test<i>.pred.trg (i from 1, in data.test.src order);
scores.json holds each set's paths and, when it has a reference, BLEU/chrF/TER
(mt_model.decode_test_sets, the same code training uses). Everything stays local: no W&B (CLAUDE.md).
"""

import os
import logging

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from hfmt.hydra_config import Config, check_config, register_configs
from hfmt.mt_model import decode_test_sets, load_base_model, load_tokenizer, load_trained_model


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: Config) -> None:
    check_config(cfg)
    if cfg.model_dir is not None and not os.path.isdir(cfg.model_dir):
        raise SystemExit(f"inf_translation: model_dir={cfg.model_dir} is not a trained model folder")
    if cfg.data.test is None or not cfg.data.test.src:
        raise SystemExit("inf_translation: data.test.src is unset or empty; nothing to decode")

    import torch
    import datasets

    outdir = HydraConfig.get().runtime.output_dir
    logging.info(f"Output folder: {outdir}")
    logging.info(f"Config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    datasets.config.HF_DATASETS_OFFLINE = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = load_tokenizer(cfg.checkpoint)
    if cfg.model_dir is None:   # experiment yaml: the checkpoint as downloaded (zero-shot), no adapter
        logging.info(f"No model_dir: decoding the base checkpoint {cfg.checkpoint} (peft.mode={cfg.peft.mode} sets the precision)")
        model, _ = load_base_model(cfg, device)
        model.eval()
    else:                       # trained run: its model/ on the base it was trained on
        model = load_trained_model(cfg, cfg.model_dir, device)

    decode_test_sets(cfg, tokenizer, model, outdir, device)   # test<i>.pred.trg + scores.json


register_configs()

if __name__ == "__main__":
    main()
