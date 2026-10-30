"""Prompt formats for causal-LM MT (feature 13_translategemma_prompt).

A *prompt format* says how a source sentence becomes the user turn of a chat prompt for one family
of models. Formats live in the ``PROMPT_FORMATS`` registry; an experiment picks one with the
``prompt_format`` key next to ``checkpoint``. The key is explicit rather than derived from the
checkpoint's chat template, because a model's required format may be documented only in its model
card; the explicit key is visible in ``run_config.yaml``/W&B and checked at launch (see
``decisions.md``).

Adding a model with special prompt requirements = one more ``PromptFormat`` entry.

Imports nothing heavy (no torch/transformers), so the launch-time config check
(``hfmt/grid.py`` -> ``hydra_config.check_config`` -> ``check_prompt_format``) can use it. Building a
formatter against a loaded tokenizer's chat template is phase B (``make_prompt_formatter``).
"""

from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass
class PromptFormat:
    """One model family's prompt format.

    ``build(cfg, text)`` returns the user-turn messages for one source sentence. ``check(cfg)``
    validates the config at launch and raises ``ValueError`` whose message states the fix.
    ``marker``, when set, is a string the checkpoint's chat template must contain (checked at job
    start in phase B); ``None`` means no template requirement.
    """

    build: Callable[[object, str], List[dict]]
    check: Callable[[object], None]
    marker: Optional[str]


def _build_chat(cfg, text):
    """The plain instruction-prefixed user turn used by every model so far."""
    return [{"role": "user", "content": f"{cfg.instruction}\n{text}"}]


def _check_chat(cfg):
    if cfg.instruction is None:
        raise ValueError("prompt_format=chat needs a non-null instruction "
                         "(the text prefixed to each source sentence)")


def _build_translategemma(cfg, text):
    """TranslateGemma's user turn: a one-element list whose mapping carries the language codes. The
    model's chat template builds the instruction from those codes, so there is no instruction string."""
    return [{"role": "user", "content": [{"type": "text",
                                          "source_lang_code": cfg.data.src,
                                          "target_lang_code": cfg.data.trg,
                                          "text": text}]}]


def _check_translategemma(cfg):
    if cfg.instruction is not None:
        raise ValueError("prompt_format=translategemma builds the instruction from data.src/data.trg; "
                         "set instruction: null")
    if not cfg.data.src or not cfg.data.trg:
        raise ValueError("prompt_format=translategemma needs data.src and data.trg (the BCP-47 "
                         "language codes, e.g. fr/en); set them in the dataset yaml")


PROMPT_FORMATS = {
    "chat": PromptFormat(build=_build_chat, check=_check_chat, marker=None),
    "translategemma": PromptFormat(build=_build_translategemma, check=_check_translategemma,
                                   marker="source_lang_code"),
}


def check_prompt_format(cfg) -> None:
    """Config-only check, run at launch and again at job start (from ``hydra_config.check_config``):
    the name must be a known format, then the format's own ``check`` runs."""
    name = cfg.prompt_format
    if name not in PROMPT_FORMATS:
        raise ValueError(f"prompt_format={name!r}: unknown; choose one of {sorted(PROMPT_FORMATS)}")
    PROMPT_FORMATS[name].check(cfg)
