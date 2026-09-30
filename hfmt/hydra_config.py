"""Structured (typed) Hydra config schema for HFMT experiments.

Registering these dataclasses with Hydra's ``ConfigStore`` makes config
composition *validated*: the YAML group files under ``conf/`` are merged onto
these closed dataclasses, so wrong types and unknown keys are rejected at compose
time rather than surfacing as a confusing error deep inside training.

Scope (feature 00_hydra): the ``sft_translation.py`` QLoRA SFT workflow only.

Confidentiality note (see CLAUDE.md): config carries data *paths*, never data
*content*. Keep it that way when logging config to W&B.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import MISSING


@dataclass
class ModelConfig:
    """Base model + QLoRA adaptation (quantization lives with the model)."""

    checkpoint: str = "Qwen/Qwen2.5-1.5B-Instruct"
    # Fine-tune a pretrained checkpoint. Training a CausalLM from scratch is
    # unsupported in sft_translation.py, so this is effectively always True.
    pretrain: bool = True

    # 4-bit quantization (bitsandbytes / QLoRA).
    load_in_4bit: bool = True
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"  # resolved to a torch dtype in code

    # LoRA adapter.
    lora_r: int = 8
    lora_alpha: float = 32
    lora_dropout: float = 0.1  # D4: previously hardcoded in sft_translation.py
    lora_target: str = "qv"  # one of: qv | all-linear


@dataclass
class DataConfig:
    """Parallel-text data. Paths only — never data content."""

    src_lang: str = "fr"
    trg_lang: str = "en"
    # YAML manifest with train/dev src/trg file lists (see egs/data/*.yaml).
    train_yaml: str = MISSING
    # Eval source text (one sentence per line) to decode at the end.
    evalset: str = MISSING
    # Instruction prefix prepended to the source in the chat prompt.
    instruction: str = ""


@dataclass
class TrainConfig:
    """Trainer hyperparameters (maps onto trl.SFTConfig + the MT callback)."""

    max_steps: int = 800
    learning_rate: float = 2e-4
    train_batch_size: int = 16
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.01
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 50  # code ties save cadence to eval cadence
    save_total_limit: int = 3
    lr_scheduler_type: str = "reduce_lr_on_plateau"
    warmup_steps: int = 30
    label_smoothing_factor: float = 0.0
    seed: int = 42
    optim: str = "adamw_torch_fused"
    completion_only_loss: bool = True
    packing: bool = False
    gradient_checkpointing: bool = True

    # Best-checkpoint selection + early stopping (surfaced from code).
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    early_stopping_patience: int = 100
    early_stopping_threshold: float = 0.05


@dataclass
class DecodeConfig:
    """Generation / eval-decoding knobs (D4: previously hardcoded)."""

    max_length: int = 128  # prompt truncation (chat template)
    max_new_tokens: int = 128
    num_beams: int = 1
    do_sample: bool = False
    batch_size: int = 16  # eval dataloader batch size


@dataclass
class WandbConfig:
    """W&B logging. Only aggregate metrics + data-free config may be logged."""

    enabled: bool = True
    project: str = "hfmt"
    entity: Optional[str] = None
    group: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    # Run name; if null it is derived from the experiment name at runtime (Stage D).
    name: Optional[str] = None


@dataclass
class Config:
    """Top-level experiment config."""

    experiment: str = "default"
    # Root for run outputs; hydra.run.dir composes this with experiment + timestamp.
    output_root: str = "outputs"

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


def register_configs() -> None:
    """Register the top-level schema so config-group YAML is validated on merge."""
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()
    cs.store(name="base_config", node=Config)
