"""Prompt formats for causal-LM MT (feature 13_translategemma_prompt).

A *prompt format* says how a source sentence becomes the user turn of a chat prompt for one family
of models. Formats live in the ``PROMPT_FORMATS`` registry; an experiment picks one with the
``prompt_format`` key next to ``checkpoint``. The key is explicit rather than derived from the
checkpoint's chat template, because a model's required format may be documented only in its model
card; the explicit key is visible in ``run_config.yaml``/W&B and checked at launch (see
``decisions.md``).

Adding a model with special prompt requirements = one more ``PromptFormat`` entry.

Imports nothing heavy (no torch/transformers), so both the launch-time config check
(``hfmt/grid.py`` -> ``hydra_config.check_config`` -> ``check_prompt_format``) and the job-start
``make_prompt_formatter`` can use it. Nothing here inspects a checkpoint's chat template: the
``prompt_format`` key is the single source of truth, so a format that doesn't fit the checkpoint
surfaces when the template is applied, not through any guessing here.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, List


@dataclass
class PromptFormat:
    """One model family's prompt format.

    ``build(cfg, text)`` returns the user-turn messages for one source sentence. ``check(cfg)``
    validates the config at launch and raises ``ValueError`` whose message states the fix.
    ``template_kwargs`` go to every ``apply_chat_template`` call that renders these messages
    (training via TRL's ``chat_template_kwargs`` column, decoding, ``tools/inspect_model.py``).
    """

    build: Callable[[object, str], List[dict]]
    check: Callable[[object], None]
    template_kwargs: dict = field(default_factory=dict)


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
    # enable_thinking=False: Qwen3.5-4B's template thinks by default (generation prompt `<think>\n`,
    # so the model writes its reasoning as the translation); 0.8B/2B already default to off. Off, the
    # generation prompt ends with the empty `<think>\n\n</think>\n\n` that the training targets start
    # with. Templates without the variable (Gemma) ignore it, so their prompts are unchanged.
    "chat": PromptFormat(build=_build_chat, check=_check_chat,
                         template_kwargs={"enable_thinking": False}),
    "translategemma": PromptFormat(build=_build_translategemma, check=_check_translategemma),
}


def check_prompt_format(cfg) -> None:
    """Config-only check, run at launch and again at job start (from ``hydra_config.check_config``):
    the name must be a known format, then the format's own ``check`` runs."""
    name = cfg.prompt_format
    if name not in PROMPT_FORMATS:
        raise ValueError(f"prompt_format={name!r}: unknown; choose one of {sorted(PROMPT_FORMATS)}")
    PROMPT_FORMATS[name].check(cfg)


def chat_template_kwargs(cfg) -> dict:
    """Extra ``apply_chat_template`` arguments for ``cfg.prompt_format`` (a copy; may be empty)."""
    return dict(PROMPT_FORMATS[cfg.prompt_format].template_kwargs)


def make_prompt_formatter(cfg):
    """Return ``format_prompt(text) -> user-turn messages`` for ``cfg.prompt_format``. The format is
    the experiment's explicit choice, already validated at launch by ``check_prompt_format``; nothing
    here inspects the checkpoint's chat template. A format that doesn't fit the checkpoint surfaces
    when the template is applied during training/decoding, not through any guessing here."""
    fmt = PROMPT_FORMATS[cfg.prompt_format]

    def format_prompt(text):
        return fmt.build(cfg, text)

    logging.info(f"prompt_format={cfg.prompt_format}: example built prompt (placeholder source): "
                 f"{format_prompt('x')!r}")
    return format_prompt
