"""Typed Hydra config schema for the SFT workflow (feature 08_hydra_redo).

Registered with Hydra's ConfigStore as ``base_config``; ``conf/config.yaml`` lists it first in
its defaults, so every composed config is validated against these dataclasses: unknown keys
and wrong types fail at compose time. Every field here is read by code (``sft_translation.py``,
or ``grid.py`` for ``slurm`` and ``output_root``); don't add fields nothing reads.

Imports nothing heavy, so ``grid.py`` and ``--cfg job`` stay fast on a login node.
Config carries data *paths*, never data content (CLAUDE.md).
"""

from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import MISSING


@dataclass
class BaseModelConfig:
    """Pretrained model; the 4-bit settings are used only when ``train.finetune_mode=qlora``."""

    checkpoint: str = "Qwen/Qwen2.5-1.5B-Instruct"
    load_in_4bit: bool = True
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"


@dataclass
class LoraConfig:
    """LoRA adapter; used when ``train.finetune_mode`` is ``lora`` or ``qlora``."""

    lora_r: int = 8
    lora_alpha: float = 32
    lora_dropout: float = 0.1
    lora_target: str = "qv"  # qv | all-linear | attention | mlp


@dataclass
class DataSplit:
    """Parallel file lists for one split; ``src[i]`` and ``trg[i]`` are line-aligned files."""

    src: List[str] = field(default_factory=list)
    trg: List[str] = field(default_factory=list)


@dataclass
class DataConfig:
    """Parallel text. ``train``/``dev``/``test`` come from the experiment's ``dataset.yaml``,
    whose paths refer to ``data_root``/``src``/``trg``. No ``test`` -> no final test decoding."""

    data_root: str = MISSING
    src: str = MISSING
    trg: str = MISSING
    instruction: str = ""  # prepended to the source in the chat prompt
    train: DataSplit = field(default_factory=DataSplit)
    dev: DataSplit = field(default_factory=DataSplit)
    test: Optional[DataSplit] = None


@dataclass
class TrainConfig:
    """Trainer settings (mapped onto ``trl.SFTConfig`` and the MT early-stopping callback)."""

    finetune_mode: str = "qlora"  # full | lora | qlora
    max_steps: int = 800
    learning_rate: float = 2e-4
    train_batch_size: int = 16
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.01
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 50
    save_total_limit: int = 3
    lr_scheduler_type: str = "reduce_lr_on_plateau"
    warmup_steps: int = 30
    label_smoothing_factor: float = 0.0
    seed: int = 42
    optim: str = "adamw_torch_fused"
    completion_only_loss: bool = True
    packing: bool = False
    gradient_checkpointing: bool = True
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    early_stopping_patience: int = 100
    early_stopping_threshold: float = 0.05


@dataclass
class DecodeConfig:
    """Generation settings for dev decoding (during training) and final test decoding."""

    max_length: int = 128  # prompt truncation
    max_new_tokens: int = 128
    num_beams: int = 1
    do_sample: bool = False
    batch_size: int = 16


@dataclass
class WandbConfig:
    """W&B logging: config and aggregate metrics only, never data (CLAUDE.md).
    Run name and group are derived from the run folder and ``run_name``."""

    enabled: bool = True
    project: str = "hfmt"
    entity: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class SlurmConfig:
    """sbatch resources, read by ``grid.py``/``egs/sft.sh`` (not by training code)."""

    partition: str = "gpu"
    gres: str = "gpu:a100:1"
    time: str = "4:00:00"
    mem_gb: int = 25
    cpus: int = 1
    array_parallelism: int = 4


@dataclass
class Config:
    """Top-level config. ``run_name`` names the run folder(s) and the W&B group."""

    run_name: str = MISSING
    # Folder holding run folders; grid.py defaults it to <experiment folder>/outputs.
    output_root: Optional[str] = None
    basemodel: BaseModelConfig = field(default_factory=BaseModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    slurm: SlurmConfig = field(default_factory=SlurmConfig)


def register_configs() -> None:
    """Register the schema as ``base_config`` (idempotent)."""
    from hydra.core.config_store import ConfigStore

    ConfigStore.instance().store(name="base_config", node=Config)
