"""Typed Hydra config schema for the SFT workflow (features 08_hydra_redo, 09_hydra_detailed_reorg).

Registered with Hydra's ConfigStore as ``base_config``; ``conf/config.yaml`` lists it first in
its defaults, so every composed config is validated against these dataclasses: unknown keys
and wrong types fail at compose time.

**Types only, no default values** (feature 09). Value fields are ``MISSING``; optional blocks and
keys are ``Optional[...] = None`` (absent). Every default lives in exactly one yaml:
``conf/config.yaml`` (project-wide: slurm, wandb, inf_name), ``conf/<group>/<option>.yaml``
(train, peft, decode), or the experiment folder (experiment yaml, ``dataset.yaml``). A key that no
yaml sets fails ``grid.py``'s missing-values check.

Every field is read by code (``sft_translation.py``/``inf_translation.py``, or ``grid.py`` for
``slurm``, ``output_root``, ``inf_name``), except the optional ``data.data_root/src/trg``, which a
``dataset.yaml`` may use to build file paths. Don't add fields nothing reads.

Imports nothing heavy, so ``grid.py`` and ``--cfg job`` stay fast on a login node.
Config carries data *paths*, never data content (CLAUDE.md).
"""

from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import MISSING


@dataclass
class LoraConfig:
    """LoRA adapter; required when ``peft.mode`` is ``lora`` or ``qlora``."""

    lora_r: int = MISSING
    lora_alpha: float = MISSING
    lora_dropout: float = MISSING
    lora_target: str = MISSING  # qv | all-linear | attention | mlp


@dataclass
class QuantConfig:
    """4-bit (bitsandbytes) loading of the base model; required when ``peft.mode`` is ``qlora``."""

    load_in_4bit: bool = MISSING
    bnb_4bit_use_double_quant: bool = MISSING
    bnb_4bit_quant_type: str = MISSING


@dataclass
class PeftConfig:
    """Fine-tuning method: the ``peft`` group (``conf/peft/``)."""

    mode: str = MISSING  # full | lora | qlora
    lora: Optional[LoraConfig] = None
    quant: Optional[QuantConfig] = None


@dataclass
class Bitext:
    """Parallel file lists for one split; ``src[i]`` and ``trg[i]`` are line-aligned files."""

    src: List[str] = MISSING
    trg: List[str] = MISSING


@dataclass
class DataConfig:
    """Parallel text, from the experiment folder's ``dataset.yaml``. ``data_root``/``src``/``trg`` are
    optional helpers for building the file paths (``data_root`` usually from ``$HFMT_DATA_ROOT``, set
    in the untracked ``local_setting.sh``); a dataset yaml may instead list absolute paths and omit
    them. No ``test`` -> no test decoding."""

    data_root: Optional[str] = None
    src: Optional[str] = None
    trg: Optional[str] = None
    train: Bitext = MISSING
    dev: Bitext = MISSING
    test: Optional[Bitext] = None


@dataclass
class TrainConfig:
    """Trainer settings (mapped onto ``trl.SFTConfig`` and the MT early-stopping callback)."""

    max_steps: int = MISSING
    learning_rate: float = MISSING
    train_batch_size: int = MISSING
    eval_batch_size: int = MISSING
    gradient_accumulation_steps: int = MISSING
    weight_decay: float = MISSING
    logging_steps: int = MISSING
    eval_steps: int = MISSING
    save_steps: int = MISSING
    save_total_limit: int = MISSING
    lr_scheduler_type: str = MISSING
    warmup_steps: int = MISSING
    label_smoothing_factor: float = MISSING
    seed: int = MISSING
    optim: str = MISSING
    completion_only_loss: bool = MISSING
    packing: bool = MISSING
    gradient_checkpointing: bool = MISSING
    load_best_model_at_end: bool = MISSING
    metric_for_best_model: str = MISSING
    greater_is_better: bool = MISSING
    early_stopping_patience: int = MISSING
    early_stopping_threshold: float = MISSING


@dataclass
class DecodeConfig:
    """Generation settings for dev decoding (during training), test decoding and inference."""

    max_length: int = MISSING  # prompt truncation
    max_new_tokens: int = MISSING
    num_beams: int = MISSING
    do_sample: bool = MISSING
    batch_size: int = MISSING


@dataclass
class WandbConfig:
    """W&B logging: config and aggregate metrics only, never data (CLAUDE.md).
    Run name and group are derived from the run folder and ``run_name``."""

    enabled: bool = MISSING
    project: str = MISSING
    entity: Optional[str] = MISSING
    tags: List[str] = MISSING


@dataclass
class SlurmConfig:
    """sbatch resources, read by ``grid.py``/``egs/launch.sh`` (not by training code)."""

    partition: str = MISSING
    gres: str = MISSING
    time: str = MISSING
    mem_gb: int = MISSING
    cpus: int = MISSING
    array_parallelism: int = MISSING
    constraint: Optional[str] = None  # e.g. "--constraint=80gb"
    exclude: Optional[str] = None  # e.g. "--exclude=node[01-05]"


@dataclass
class Config:
    """Top-level config. ``run_name`` names the run folder(s) and the W&B group."""

    run_name: str = MISSING
    checkpoint: str = MISSING  # pretrained model id or path
    instruction: str = MISSING  # prepended to the source in the chat prompt
    # Folder holding run folders; unset -> grid.py uses <experiment folder>/outputs.
    output_root: Optional[str] = None
    inf_name: str = MISSING  # inference output subfolder: <run folder>/inf/<inf_name>/
    model_dir: Optional[str] = None  # trained model to decode; set by grid.py --inf
    peft: PeftConfig = field(default_factory=PeftConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    slurm: SlurmConfig = field(default_factory=SlurmConfig)


def check_config(cfg) -> None:
    """Fail if ``peft.mode`` disagrees with which ``peft`` blocks are present."""
    mode, has_lora, has_quant = cfg.peft.mode, cfg.peft.lora is not None, cfg.peft.quant is not None
    need = {"full": (False, False), "lora": (True, False), "qlora": (True, True)}
    if mode not in need:
        raise ValueError(f"peft.mode={mode}: must be one of {' | '.join(need)}")
    if (has_lora, has_quant) != need[mode]:
        state = lambda lora, quant: f"peft.lora {'set' if lora else 'unset'}, peft.quant {'set' if quant else 'unset'}"
        raise ValueError(f"peft.mode={mode} needs {state(*need[mode])}; got {state(has_lora, has_quant)}")


def register_configs() -> None:
    """Register the schema as ``base_config`` (idempotent)."""
    from hydra.core.config_store import ConfigStore

    ConfigStore.instance().store(name="base_config", node=Config)
