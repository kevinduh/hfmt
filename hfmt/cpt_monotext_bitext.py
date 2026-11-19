"""Continued pre-training (CPT) of a decoder-only LM on monotext and bitext (Hydra-driven, feature 16_cpt).

Plain next-token loss on raw text (no chat template), with TRL's SFTTrainer on a `text` dataset, packed
into `cpt.max_length`-token rows. The data comes from the experiment's dataset yaml, as for SFT:
`data.monotext` (one example per non-empty line) and/or `data.train` bitext (each side its own example,
or `src + " " + trg`), mixed by `cpt.monotext_ratio`. Dev loss is on `data.dev`, formatted like the
bitext. CPT never decodes: no dev BLEU, no test decoding; `data.test`, `instruction` and `decode` are
unused. The SFT-only train keys are forced (hydra_config.CPT_FORCED_TRAIN). The saved model/ is a
checkpoint for SFT (lora/qlora: after `launch.sh merge`).

Normally launched by `egs/launch.sh cpt`. Direct single run (debugging; output goes to Hydra's default
folder unless hydra.run.dir=... is given; local_setting.sh provides the data location):
    source egs/synlrl/tuk-eng/local_setting.sh
    python -m hfmt.cpt_monotext_bitext --config-dir egs/synlrl/tuk-eng --config-name cpt_lora [overrides]

Heavy imports (torch, transformers, ...) live inside functions so config-only use stays fast. The data
functions only need `datasets`, so hfmt/test_cpt_monotext_bitext.py runs without torch.
"""

import time
import os
import logging

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from hfmt.hydra_config import Config, apply_cpt_overrides, check_config, register_configs
from hfmt.mt_model import load_tokenizer
from hfmt.train_common import build_model, init_wandb, inspect_first_batch, save_model, train, trainer_args


# Relative eval-loss difference allowed between model/'s fresh dev evaluation and the evaluation at the
# step model/ claims to hold (same weights, same data; only kernel nondeterminism differs). The 16_cpt GPU
# smoke test (Tiny Aya, LoRA, bf16) measured 9e-4, so 5e-3 leaves room for noise.
DEV_LOSS_TOLERANCE = 5e-3
# Examples per source used to estimate token shares and how many examples exceed cpt.max_length.
TOKEN_SAMPLE = 10000


###################################
## Data: module-level functions with their options as arguments (stable `datasets` cache fingerprints)

def bitext_examples(batch, bitext_format):
    """Batched map: line pairs (`src`, `trg`) -> `text` examples. "separate": each non-empty side is its
    own example; "concat": `src + " " + trg` on one line (an empty side is left out)."""
    if bitext_format == "separate":
        texts = [t for s, g in zip(batch["src"], batch["trg"]) for t in (s, g) if t.strip()]
    else:
        texts = [" ".join(t for t in (s, g) if t.strip()) for s, g in zip(batch["src"], batch["trg"])]
        texts = [t for t in texts if t]
    return {"text": texts}


def monotext_examples(batch):
    """Batched map: monotext lines -> `text` examples, dropping empty (whitespace-only) lines."""
    return {"text": [t for t in batch["text"] if t.strip()]}


def load_bitext(bitext):
    """A ``Bitext`` (src/trg file lists, line-aligned) as one dataset with `src` and `trg` columns; each
    side's files are read in list order, as SFT reads them."""
    from datasets import concatenate_datasets, load_dataset

    src = load_dataset("text", data_files=list(bitext.src), split="train").rename_column("text", "src")
    trg = load_dataset("text", data_files=list(bitext.trg), split="train").rename_column("text", "trg")
    if len(src) != len(trg):
        raise ValueError(f"bitext sides differ in length: {len(src)} src lines vs {len(trg)} trg lines "
                         f"({list(bitext.src)} / {list(bitext.trg)})")
    return concatenate_datasets([src, trg], axis=1)


def format_bitext(bitext_ds, bitext_format, num_proc=None):
    return bitext_ds.map(bitext_examples, batched=True, fn_kwargs={"bitext_format": bitext_format},
                         remove_columns=bitext_ds.column_names, num_proc=num_proc)


def load_monotext(files, num_proc=None):
    from datasets import load_dataset

    mono = load_dataset("text", data_files=list(files), split="train")
    return mono.map(monotext_examples, batched=True, remove_columns=mono.column_names, num_proc=num_proc)


def mix(mono, bitext, monotext_ratio, seed):
    """One dataset drawing each example from monotext with probability ``monotext_ratio`` (else bitext)
    until both have been used up at least once; the smaller one is repeated to keep the mix. The ratio
    sets the mix, not how much is trained (train.max_steps does)."""
    from datasets import interleave_datasets

    return interleave_datasets([mono, bitext], probabilities=[monotext_ratio, 1 - monotext_ratio], seed=seed,
                               stopping_strategy="all_exhausted")


def build_datasets(cfg, num_proc=None):
    """``(train, dev, sources)``: `text`-only train and dev datasets from the `cpt` options, and the
    per-source train datasets before mixing ({"monotext": ..., "bitext": ...}, for the logs)."""
    cpt = cfg.cpt
    sources = {}
    if cpt.use_monotext:
        start = time.time()
        sources["monotext"] = load_monotext(cfg.data.monotext, num_proc)
        logging.info(f"monotext: {len(sources['monotext'])} examples from {list(cfg.data.monotext)} "
                     f"({time.time() - start:.1f}s)")
    if cpt.use_bitext:
        start = time.time()
        pairs = load_bitext(cfg.data.train)
        sources["bitext"] = format_bitext(pairs, cpt.bitext_format, num_proc)
        logging.info(f"bitext: {len(pairs)} line pairs -> {len(sources['bitext'])} examples "
                     f"({cpt.bitext_format}) ({time.time() - start:.1f}s)")
    if len(sources) == 2:
        start = time.time()
        train_ds = mix(sources["monotext"], sources["bitext"], cpt.monotext_ratio, cfg.train.seed)
        logging.info(f"mixed with monotext_ratio={cpt.monotext_ratio}: {len(train_ds)} examples "
                     f"(the smaller source repeated; {time.time() - start:.1f}s)")
    else:
        (train_ds,) = sources.values()
    dev_ds = format_bitext(load_bitext(cfg.data.dev), cpt.bitext_format)
    logging.info(f"dev: {len(dev_ds)} examples from data.dev ({cpt.bitext_format})")
    return train_ds, dev_ds, sources


def log_token_stats(sources, tokenizer, max_length, sample=TOKEN_SAMPLE):
    """Log each source's estimated token count and share (the ratio counts lines, not tokens) and the
    share of examples longer than ``max_length`` (bfd packing truncates them), from the first ``sample``
    examples of each source."""
    tokens = {}
    for name, ds in sources.items():
        texts = ds.select(range(min(sample, len(ds))))["text"]
        lengths = [len(ids) for ids in tokenizer(list(texts), add_special_tokens=False)["input_ids"]]
        mean = sum(lengths) / max(len(lengths), 1)
        tokens[name] = mean * len(ds)
        too_long = sum(n > max_length for n in lengths)
        logging.info(f"{name}: ~{mean:.1f} tokens/example, ~{tokens[name]:,.0f} tokens; {too_long} of {len(lengths)} "
                     f"sampled examples exceed max_length={max_length} (truncated by bfd packing)")
    if len(tokens) == 2:
        total = sum(tokens.values())
        logging.info("token share before mixing (estimated): "
                     + ", ".join(f"{name} {100 * n / total:.1f}%" for name, n in tokens.items()))


###################################
## After training

def check_model_dev_loss(log_history, model_global_step, eval_loss, tolerance=DEV_LOSS_TOLERANCE):
    """Does ``model/`` evaluate dev to the loss of the step it claims to hold? ``eval_loss`` is a fresh
    evaluation of the final model; the expectation is that step's ``eval_loss`` in ``log_history``.
    Returns the ``training_info.yaml`` fields ``model_dev_scores`` and ``model_dev_check`` ("ok",
    "unchecked: ..." or "MISMATCH: ..."), as SFT's check_model_dev_scores does with BLEU."""
    expected = next((e["eval_loss"] for e in log_history if e.get("step") == model_global_step and "eval_loss" in e), None)
    if expected is None:
        check = f"unchecked: no eval_loss at step {model_global_step} in log_history"
    elif abs(eval_loss - expected) > tolerance * abs(expected):
        check = (f"MISMATCH: model/ dev eval_loss {eval_loss:.6f} vs {expected:.6f} at step {model_global_step} "
                 f"(relative tolerance {tolerance})")
    else:
        check = "ok"
    return {"model_dev_scores": {"eval_loss": float(eval_loss)}, "model_dev_check": check}


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: Config) -> None:
    import torch
    import datasets
    from transformers import EarlyStoppingCallback
    from trl import SFTTrainer

    check_config(cfg)
    if cfg.cpt is None:
        raise SystemExit("cpt_monotext_bitext: no cpt block in the config; an SFT config runs with sft_translation.py")
    # The SFT-only train keys (already set in a run_config.yaml saved by launch.sh cpt; a direct run sets them here).
    for key, old, new, reason in apply_cpt_overrides(cfg):
        logging.info(f"cpt: train.{key} {old} -> {new} ({reason})")
    # Run folder chosen by Hydra (hydra.run.dir, set by egs/job.sh); everything is written here.
    outdir = HydraConfig.get().runtime.output_dir
    logging.info(f"Run folder: {outdir}")
    logging.info(f"Config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")
    logging.info("Not used by CPT: data.test, instruction, decode (CPT never decodes)")
    os.environ["WANDB_LOG_MODEL"] = "false"  # never upload models to W&B
    os.environ["WANDB_CONSOLE"] = "off"      # never upload console output (it has example text)
    # Resume: egs/job.sh cpt_resume sets HFMT_SFT_RESUME=1 (the same variable as SFT, feature 12).
    resume = os.environ.get("HFMT_SFT_RESUME") == "1"
    logging.info(f"Resume from the last checkpoint: {resume}")

    init_wandb(cfg, outdir, resume)   # config + aggregate metrics only (CLAUDE.md)

    tokenizer = load_tokenizer(cfg.checkpoint)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    datasets.config.HF_DATASETS_OFFLINE = True
    num_proc = cfg.slurm.cpus if cfg.slurm.cpus > 1 else None   # datasets/TRL map workers

    ###################################
    ## Load data
    logging.info(f"======== Loading data ========")
    start_time = time.time()
    train_ds, dev_ds, sources = build_datasets(cfg, num_proc)
    log_token_stats(sources, tokenizer, cfg.cpt.max_length)
    logging.info(f"Loading data - Elapsed time: {time.time() - start_time:.1f}s")

    ###################################
    ## Model and Trainer
    logging.info(f"======== Model Configuration ========")
    model, use_bf16 = build_model(cfg, device, tokenizer)
    training_args = trainer_args(cfg, outdir, use_bf16, dataset_text_field="text", max_length=cfg.cpt.max_length,
                                 packing_strategy=cfg.cpt.packing_strategy, dataset_num_proc=num_proc)

    start_time = time.time()
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        # Our tokenizer, not AutoProcessor (see sft_translation.py); TRL appends its eos_token to each example.
        processing_class=tokenizer,
    )
    logging.info(f"Tokenizing and packing (TRL) - Elapsed time: {time.time() - start_time:.1f}s")
    logging.info(f"Packed: {len(trainer.train_dataset)} train rows, {len(trainer.eval_dataset)} dev rows "
                 f"of up to {cfg.cpt.max_length} tokens; eos_token {tokenizer.eos_token!r}")

    inspect_first_batch(trainer, tokenizer, model)

    # Early stopping on eval_loss at every evaluation; no decoding (CPT never decodes).
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=cfg.train.early_stopping_patience,
                                               early_stopping_threshold=cfg.train.early_stopping_threshold))

    ###################################
    ## Training
    train(trainer, model, cfg, outdir, resume)
    model, info = save_model(trainer, model, tokenizer, outdir)

    # Measure, don't assume: evaluate dev with the final weights and compare with the eval_loss logged at the
    # step model/ claims to hold. A mismatch is logged and recorded, not fatal.
    logging.info(f"======== Dev evaluation of model/ ========")
    log_history = list(trainer.state.log_history)    # before evaluate() adds its own entry
    eval_loss = trainer.evaluate()["eval_loss"]
    info.update(check_model_dev_loss(log_history, info["model_global_step"], eval_loss))
    log = logging.error if info["model_dev_check"].startswith("MISMATCH") else logging.info
    log(f"model/ dev check: {info['model_dev_check']} (dev scores {info['model_dev_scores']})")
    OmegaConf.save(OmegaConf.create(info), os.path.join(outdir, "model", "training_info.yaml"))
    logging.info(f"model/ holds step {info['model_global_step']} "
                 f"({info['metric_for_best_model']} best {info['best_metric']} at step {info['best_global_step']})")
    import wandb
    if wandb.run is not None:                                      # numbers only (CLAUDE.md)
        wandb.run.summary.update({f"dev/model/{k}": v for k, v in info["model_dev_scores"].items()})


register_configs()

if __name__ == "__main__":
    main()
