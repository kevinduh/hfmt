"""Tests for the chat-template fix in hfmt/prompt_formats.py (TranslateGemma's assistant-turn whitespace).
String-level only: no transformers, torch or checkpoint needed. The rendered-token check (the prompt is
a token prefix of prompt + reply) needs the gated tokenizer and runs on the cluster.

Run standalone (pytest isn't installed in the hfmt env):
    python hfmt/test_prompt_formats.py
"""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hfmt.prompt_formats import PROMPT_FORMATS, fix_chat_template  # noqa: E402

# The assistant branch of google/translategemma-4b-it's chat template as shipped (2026-11-04).
SHIPPED = """{%- for message in messages -%}
    {%- if (message['role'] == 'assistant') -%}
        {{ '<start_of_turn>model\\n'}}
        {{ message["content"] | trim }}
    {%- elif (message['role'] == 'user') -%}
        {{ '<start_of_turn>user\\n' + message["content"][0]["text"] | trim }}
    {%- endif -%}
    {{ '<end_of_turn>\\n' }}
{%- endfor -%}
"""

fix = PROMPT_FORMATS["translategemma"].fix_template


def raises(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return True
    return False


def test_fix_dashes_only_the_assistant_content_tag():
    fixed = fix(SHIPPED)
    assert '{{- message["content"] | trim }}' in fixed
    assert '{{ message["content"] | trim }}' not in fixed
    assert fixed == SHIPPED.replace('{{ message["content"]', '{{- message["content"]')


def test_fix_is_idempotent_for_saved_tokenizers():
    # A full fine-tune's model/ and a model_merged/ save the already-fixed template.
    assert fix(fix(SHIPPED)) == fix(SHIPPED)


def test_fix_refuses_a_changed_template():
    assert raises(fix, "{{ bos_token }}{% for m in messages %}{{ m['content'] }}{% endfor %}")
    doubled = SHIPPED + '{{ message["content"] | trim }}'
    assert raises(fix, doubled)


def test_fix_chat_template_patches_translategemma_only():
    tok = SimpleNamespace(chat_template=SHIPPED)
    assert fix_chat_template(tok, "chat") is tok and tok.chat_template == SHIPPED
    assert fix_chat_template(tok, "translategemma") is tok and tok.chat_template == fix(SHIPPED)


def test_fix_chat_template_needs_one_template_string():
    for template in (None, {"default": SHIPPED}):
        assert raises(fix_chat_template, SimpleNamespace(chat_template=template), "translategemma")
    # A format without a fix never looks at the template.
    assert fix_chat_template(SimpleNamespace(chat_template=None), "chat").chat_template is None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
