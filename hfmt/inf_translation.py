"""Decode a test set with a model trained by sft_translation.py (Hydra-driven, feature 09).

Runs on a trained run folder's saved config, so the base model, fine-tuning mode, prompt and
decode settings match training. Normally launched by `egs/launch.sh inf`, which saves the job's
config as <run folder>/inf/<inf_name>/inf_config.yaml. Direct run (debugging):
    python -m hfmt.inf_translation --config-dir <run folder> --config-name run_config \\
        model_dir=<run folder>/model hydra.run.dir=<output folder> [overrides]
e.g. decode.num_beams=5, or another test set: 'data.test.src=[...]' 'data.test.trg=[...]'.

Writes pred.trg (one line per test line) and, when data.test.trg is given, scores.json
(BLEU/chrF/TER). Everything stays local: no W&B (CLAUDE.md).
"""

import json
import os
import logging

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from hfmt.hydra_config import Config, check_config, register_configs
from hfmt.mt_model import inference_on_eval_data, load_tokenizer, load_trained_model, score


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: Config) -> None:
    check_config(cfg)
    if cfg.model_dir is None or not os.path.isdir(cfg.model_dir):
        raise SystemExit(f"inf_translation: model_dir={cfg.model_dir} is not a trained model folder")
    if cfg.data.test is None or not cfg.data.test.src:
        raise SystemExit("inf_translation: data.test.src is unset or empty; nothing to decode")

    import torch
    import datasets
    from datasets import load_dataset

    outdir = HydraConfig.get().runtime.output_dir
    logging.info(f"Output folder: {outdir}")
    logging.info(f"Config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    datasets.config.HF_DATASETS_OFFLINE = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = load_tokenizer(cfg.checkpoint)
    model = load_trained_model(cfg, cfg.model_dir, device)

    test_data = load_dataset("text", data_files=list(cfg.data.test.src), streaming=False, split="train")
    preds = inference_on_eval_data(tokenizer, model, test_data, os.path.join(outdir, "pred.trg"), device,
                                   cfg.instruction, cfg.decode)

    if cfg.data.test.trg:
        refs = load_dataset("text", data_files=list(cfg.data.test.trg), streaming=False, split="train")["text"]
        scores = score(preds, refs)
        logging.info(f"Test scores: {scores}")
        with open(os.path.join(outdir, "scores.json"), "w") as f:
            json.dump(scores, f, indent=2)


register_configs()

if __name__ == "__main__":
    main()
