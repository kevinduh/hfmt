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
``make_prompt_formatter`` can use it. Nothing here inspects a checkpoint's chat template to choose
a format: the ``prompt_format`` key is the single source of truth, so a format that doesn't fit the
checkpoint surfaces when the template is applied, not through any guessing here. A format may carry
a ``fix_template`` that repairs a known defect in its model's shipped template (TranslateGemma,
below); that edits the template, it never picks the format.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class PromptFormat:
    """One model family's prompt format.

    ``build(cfg, text)`` returns the user-turn messages for one source sentence. ``check(cfg)``
    validates the config at launch and raises ``ValueError`` whose message states the fix.
    ``template_kwargs`` go to every ``apply_chat_template`` call that renders these messages
    (training via TRL's ``chat_template_kwargs`` column, decoding, ``tools/inspect_model.py``).
    ``fix_template(template) -> template``, if set, is applied to the checkpoint's chat template
    wherever hfmt loads the tokenizer (``fix_chat_template``, called by ``mt_model.load_tokenizer``).
    """

    build: Callable[[object, str], List[dict]]
    check: Callable[[object], None]
    template_kwargs: dict = field(default_factory=dict)
    fix_template: Optional[Callable[[str], str]] = None


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


# TranslateGemma's shipped chat template (google/translategemma-4b-it, checked 2026-11-04) writes an
# assistant turn with two output tags on separate lines:
#         {{ '<start_of_turn>model\n'}}
#         {{ message["content"] | trim }}
# Neither tag strips whitespace, so the line break and 8-space indentation between them are output:
# "<start_of_turn>model\n\n        <target><end_of_turn>\n". The generation prompt ends at
# "<start_of_turn>model\n", so training (prompt + reply) and decoding (prompt) disagreed: the tokenizer
# merges the two newlines into one token, the prompt was not a token prefix of prompt + reply (TRL
# warned on every example), and the model trained on a reply start it never sees when decoding. The
# whitespace-control dash on the content tag drops that template text; the "\n" inside the header
# string is output, not template text, so it stays. Only assistant turns change: prompts (and so all
# decoding) render exactly as before.
_TRANSLATEGEMMA_CONTENT_TAG = '{{ message["content"] | trim }}'
_TRANSLATEGEMMA_CONTENT_TAG_FIXED = '{{- message["content"] | trim }}'


def _fix_translategemma_template(template):
    """TranslateGemma's chat template with the assistant-turn whitespace removed (see above). A template
    that already has the fix (a tokenizer hfmt saved: a full fine-tune's ``model/``, a ``model_merged/``)
    is returned unchanged. Anything else raises: a changed upstream template has to be re-checked, not
    trained on silently."""
    if _TRANSLATEGEMMA_CONTENT_TAG_FIXED in template:
        return template
    found = template.count(_TRANSLATEGEMMA_CONTENT_TAG)
    if found != 1:
        raise ValueError(
            f"prompt_format=translategemma: expected the assistant-content tag {_TRANSLATEGEMMA_CONTENT_TAG!r} "
            f"once in the checkpoint's chat template, found it {found} times. The template has changed: check "
            "how it renders an assistant turn (the generation prompt must be a token prefix of prompt + reply) "
            "and update _fix_translategemma_template in hfmt/prompt_formats.py.")
    return template.replace(_TRANSLATEGEMMA_CONTENT_TAG, _TRANSLATEGEMMA_CONTENT_TAG_FIXED)


PROMPT_FORMATS = {
    # enable_thinking=False: Qwen3.5-4B's template thinks by default (generation prompt `<think>\n`,
    # so the model writes its reasoning as the translation); 0.8B/2B already default to off. Off, the
    # generation prompt ends with the empty `<think>\n\n</think>\n\n` that the training targets start
    # with. Templates without the variable (Gemma) ignore it, so their prompts are unchanged.
    "chat": PromptFormat(build=_build_chat, check=_check_chat,
                         template_kwargs={"enable_thinking": False}),
    "translategemma": PromptFormat(build=_build_translategemma, check=_check_translategemma,
                                   fix_template=_fix_translategemma_template),
}


def check_prompt_format(cfg) -> None:
    """Config-only check, run at launch and again at job start (from ``hydra_config.check_config``):
    the name must be a known format, then the format's own ``check`` runs."""
    name = cfg.prompt_format
    if name not in PROMPT_FORMATS:
        raise ValueError(f"prompt_format={name!r}: unknown; choose one of {sorted(PROMPT_FORMATS)}")
    PROMPT_FORMATS[name].check(cfg)


def fix_chat_template(tokenizer, prompt_format):
    """Apply ``prompt_format``'s ``fix_template`` (if any) to ``tokenizer.chat_template`` in place and
    return the tokenizer. Called by ``mt_model.load_tokenizer``, so training, dev/test decoding,
    inference and merge all render with the same template, and a tokenizer hfmt saves keeps the fix."""
    fix = PROMPT_FORMATS[prompt_format].fix_template
    if fix is None:
        return tokenizer
    template = tokenizer.chat_template
    if not isinstance(template, str):
        raise ValueError(f"prompt_format={prompt_format}: expected the tokenizer to carry one chat template "
                         f"string, got {type(template).__name__}")
    fixed = fix(template)
    if fixed != template:
        logging.info(f"prompt_format={prompt_format}: patched the checkpoint's chat template ({fix.__name__})")
    tokenizer.chat_template = fixed
    return tokenizer


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
