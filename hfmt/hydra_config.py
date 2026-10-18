"""Structured (typed) Hydra config schema for HFMT experiments.

Registering these dataclasses with Hydra's ``ConfigStore`` makes config
composition *validated*: the YAML group files under ``conf/`` are merged onto
these closed dataclasses, so wrong types and unknown keys are rejected at compose
time rather than surfacing as a confusing error deep inside training.

Scope (feature 00_hydra): the ``sft_translation.py`` QLoRA SFT workflow only.

Feature 04 reshaped the groups into atomistic, mix-and-match components:
``basemodel`` (the pretrained model + quantization), ``lora`` (the adapter, with an
``enabled`` toggle for full fine-tuning), ``data``, ``train``, ``decode``, plus
``wandb`` and an inline ``sweep`` block. Experiments compose these and live under
``egs/<dataset>/<src>-<trg>/`` (feature 06; reached via ``hydra.searchpath``).

Confidentiality note (see CLAUDE.md): config carries data *paths*, never data
*content*. Keep it that way when logging config to W&B.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from omegaconf import MISSING


@dataclass
class BaseModelConfig:
    """The pretrained base model we fine-tune, plus its (optional) quantization.

    LoRA lives in its own ``LoraConfig`` group now (feature 04) so the base model
    and the adaptation method are independent, mix-and-match components. When LoRA
    is disabled (full fine-tune), set ``load_in_4bit=false`` -- a 4-bit frozen base
    cannot be fully fine-tuned (enforced in sft_translation.py).
    """

    checkpoint: str = "Qwen/Qwen2.5-1.5B-Instruct"
    # Fine-tune a pretrained checkpoint. Training a CausalLM from scratch is
    # unsupported in sft_translation.py, so this is effectively always True.
    pretrain: bool = True

    # 4-bit quantization (bitsandbytes / QLoRA). Required for QLoRA; turn OFF for a
    # full fine-tune.
    load_in_4bit: bool = True
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"  # quantized compute dtype (torch dtype in code)

    # Load dtype used only when NOT quantized (full fine-tune path); resolved to a
    # torch dtype in code. Ignored when load_in_4bit=true.
    dtype: str = "bfloat16"


@dataclass
class LoraConfig:
    """LoRA adapter settings (feature 04: split out of the old ModelConfig).

    ``enabled=false`` skips adapter injection entirely to fully fine-tune the base
    model (requires ``basemodel.load_in_4bit=false``). Named ``LoraConfig`` only in
    this schema module; peft's own ``LoraConfig`` is imported separately inside
    sft_translation.py, so there is no clash.
    """

    enabled: bool = True
    lora_r: int = 8
    lora_alpha: float = 32
    lora_dropout: float = 0.1  # D4: previously hardcoded in sft_translation.py
    lora_target: str = "qv"  # one of: qv | all-linear | attention | mlp


@dataclass
class DataSplit:
    """Source/target file lists for one split (post-inflation absolute paths).

    Mirrors one ``train``/``dev``/``test`` block of the ``dataset.yaml`` template once its
    ``$DATA_ROOT``/``$SRC``/``$TRG`` placeholders are inflated (feature 06). Lists (not scalars)
    because ``datasets.load_dataset("text", data_files=...)`` takes a list of files per split.
    """

    src: List[str] = field(default_factory=list)
    trg: List[str] = field(default_factory=list)


@dataclass
class DataConfig:
    """Parallel-text data. Paths only — never data content.

    Feature 06: the old ``conf/data`` group (with a ``train_yaml`` manifest + a flat
    ``evalset``) is gone. An experiment now declares ``data_root``/``src``/``trg`` and names a
    per-lang-pair ``dataset.yaml`` template (``dataset_file`` in ``dataset_dir``); the template
    is inflated in code into the ``train``/``dev``/``test`` path lists below. ``dataset_dir`` is
    injected by ``egs/run.sh`` (the experiment's own directory). ``test`` is optional — when its
    source files are absent the end-of-run inference pass is skipped.

    ``dataset_name`` identifies the dataset (e.g. ``mmtc``). When unset it is filled in code from
    the hierarchy (the parent of ``dataset_dir``: ``egs/<dataset>/<src>-<trg>/``). Together with
    ``src``/``trg`` it carries the dataset/lang-pair context into W&B as tags + config, so run
    names don't need to repeat it (feature 06 naming schema).
    """

    src: str = MISSING  # source language code, e.g. "fr"
    trg: str = MISSING  # target language code, e.g. "en"
    # Absolute dir where the data FILES live; the inflation input for $DATA_ROOT.
    data_root: str = MISSING
    # Template file NAME (basename, including the .yaml suffix); default dataset.yaml.
    dataset_file: str = "dataset.yaml"
    # Directory holding dataset_file (== the experiment's own dir); injected by egs/run.sh.
    dataset_dir: str = MISSING
    # Dataset identifier, e.g. "mmtc"; filled in code from dataset_dir's parent when unset.
    # An identifier only, never data content -- it becomes a W&B tag.
    dataset_name: Optional[str] = None
    # Instruction prefix prepended to the source in the chat prompt.
    instruction: str = ""

    # Inflated absolute paths, filled in code from dataset_file (feature 06). train/dev are
    # required; test is optional (None -> skip the final held-out inference pass).
    train: DataSplit = field(default_factory=DataSplit)
    dev: DataSplit = field(default_factory=DataSplit)
    test: Optional[DataSplit] = None


@dataclass
class TrainConfig:
    """Trainer hyperparameters (maps onto trl.SFTConfig + the MT callback)."""

    # Fine-tuning mode selected in sft_translation.py's model-loading block:
    #   "full"  -- full-parameter fine-tune of the (unquantized) base model
    #   "lora"  -- LoRA adapters on an unquantized base
    #   "qlora" -- LoRA adapters on a 4-bit (bitsandbytes) quantized base
    # "qlora" requires basemodel.load_in_4bit=true; "full"/"lora" load the base
    # unquantized.
    finetune_mode: str = "qlora"
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
    # If null, the group is the experiment name (the file stem). The dataset and lang pair are
    # NOT part of names or groups: derive_run_identity prepends them to tags (feature 06).
    group: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    # Run name; if null it is derived from the experiment name at runtime (Stage D).
    name: Optional[str] = None


@dataclass
class SweepConfig:
    """Inline, toggleable parameter sweep (feature 04).

    Lives in the experiment config (``egs/<dataset>/<src>-<trg>/<name>.yaml``) so a sweep is a
    self-contained part of the experiment. ``enabled`` is the on/off switch;
    ``params`` maps each sweep axis to a comma-separated list of choices. Axes are
    ordinary Hydra task overrides -- either a config-group choice (sweep over whole
    component files, e.g. ``basemodel: "qwen2.5-7b,qwen2.5-1.5b"``) or a leaf value
    (e.g. ``train.learning_rate: "2e-5,2e-4"``).

    Hydra itself does not act on this block: ``egs/run.sh`` runs ``sweep_probe.py``,
    which reads it and, only when ``enabled``, forwards the axes as bare
    comma-list overrides to the ``--multirun`` launch (passing them under
    ``hydra.sweeper.params`` on the CLI is rejected by Hydra). Confidentiality
    (CLAUDE.md): axis names/values are hyperparameters or config-file names, never
    data content, so this is safe to log to W&B.
    """

    enabled: bool = False
    params: Dict[str, str] = field(default_factory=dict)


@dataclass
class Config:
    """Top-level experiment config."""

    experiment: str = "default"
    # Root for run outputs; hydra.run.dir composes this with experiment + timestamp.
    output_root: str = "outputs"

    basemodel: BaseModelConfig = field(default_factory=BaseModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)


def register_configs() -> None:
    """Register the top-level schema so config-group YAML is validated on merge."""
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()
    cs.store(name="base_config", node=Config)
