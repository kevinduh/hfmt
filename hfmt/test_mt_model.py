"""Tests for hfmt/mt_model.py's load_checkpoint_weights (the full fine-tune checkpoint reload). No GPU
or training deps needed: a stand-in model class plays the PreTrainedModel.

Run standalone (pytest isn't installed in the hfmt env):
    python hfmt/test_mt_model.py
"""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hfmt.mt_model import load_checkpoint_weights  # noqa: E402


class FakeModel:
    """Just enough of a PreTrainedModel: weights by name, a dtype, from_pretrained from a registry."""
    checkpoints = {}   # checkpoint_dir -> weights, standing in for the files on disk
    calls = []         # from_pretrained calls

    def __init__(self, weights, dtype="float32"):
        self.weights, self.dtype = dict(weights), dtype

    def parameters(self):
        return iter(SimpleNamespace(dtype=self.dtype) for _ in self.weights)

    def state_dict(self):
        return dict(self.weights)

    def load_state_dict(self, state_dict, strict=False):
        if strict and set(state_dict) != set(self.weights):
            raise RuntimeError("key mismatch")
        self.weights.update(state_dict)

    @classmethod
    def from_pretrained(cls, checkpoint_dir, dtype):
        cls.calls.append((checkpoint_dir, dtype))
        return cls(cls.checkpoints[checkpoint_dir], dtype)


def test_weights_come_from_the_checkpoint_in_the_models_dtype():
    FakeModel.checkpoints["/run/checkpoint-50"] = {"model.embed_tokens.weight": 5, "lm_head.weight": 5}
    model = FakeModel({"model.embed_tokens.weight": 0, "lm_head.weight": 0}, dtype="bfloat16")
    load_checkpoint_weights(model, "/run/checkpoint-50")
    assert model.weights == {"model.embed_tokens.weight": 5, "lm_head.weight": 5}
    assert FakeModel.calls[-1] == ("/run/checkpoint-50", "bfloat16")


def test_a_name_mismatch_fails_instead_of_warning():
    FakeModel.checkpoints["/run/checkpoint-60"] = {"model.language_model.embed_tokens.weight": 5}
    model = FakeModel({"model.embed_tokens.weight": 0})
    try:
        load_checkpoint_weights(model, "/run/checkpoint-60")
    except RuntimeError:
        assert model.weights == {"model.embed_tokens.weight": 0}
    else:
        raise AssertionError("expected the strict load_state_dict to fail")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
