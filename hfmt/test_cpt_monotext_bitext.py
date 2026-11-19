"""Tests for the data preparation and the model/ check of hfmt/cpt_monotext_bitext.py (feature 16_cpt).
Needs `datasets`, not torch or a GPU. Inputs are made-up sentences written to a temp dir, never dataset content.

Run standalone (pytest isn't installed in the hfmt env):
    python hfmt/test_cpt_monotext_bitext.py
"""

import logging
import os
import sys
import tempfile
from collections import Counter

from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hfmt.cpt_monotext_bitext import (bitext_examples, build_datasets, check_model_dev_loss, mix,  # noqa: E402
                                      monotext_examples)

logging.disable(logging.INFO)


def write(lines):
    path = tempfile.mktemp(suffix=".txt")
    with open(path, "w") as f:
        f.write("".join(f"{line}\n" for line in lines))
    return path


def cfg(use_monotext=True, use_bitext=True, bitext_format="separate", monotext_ratio=0.5, n_mono=4, n_pairs=3):
    """A config with only what build_datasets reads; data files are made-up sentences."""
    src = [f"source sentence {i}" for i in range(n_pairs)]
    trg = [f"target sentence {i}" for i in range(n_pairs)]
    data = {"train": {"src": [write(src)], "trg": [write(trg)]},
            "dev": {"src": [write(["dev source"])], "trg": [write(["dev target"])]},
            "monotext": [write([f"mono line {i}" for i in range(n_mono)])]}
    return OmegaConf.create({"data": data, "train": {"seed": 42},
                             "cpt": {"use_monotext": use_monotext, "use_bitext": use_bitext, "bitext_format": bitext_format,
                                     "monotext_ratio": monotext_ratio}})


def test_bitext_examples():
    batch = {"src": ["s1", "s2", "  "], "trg": ["t1", "", "t3"]}
    assert bitext_examples(batch, "separate") == {"text": ["s1", "t1", "s2", "t3"]}     # empty sides dropped
    assert bitext_examples(batch, "concat") == {"text": ["s1 t1", "s2", "t3"]}          # space-joined
    assert bitext_examples({"src": [""], "trg": [" "]}, "concat") == {"text": []}


def test_monotext_examples():
    assert monotext_examples({"text": ["a b", "", "   ", "c"]}) == {"text": ["a b", "c"]}


def test_bitext_only_and_dev_format():
    train, dev, sources = build_datasets(cfg(use_monotext=False))
    assert list(sources) == ["bitext"] and train.column_names == ["text"]
    assert sorted(train["text"]) == sorted([f"source sentence {i}" for i in range(3)] + [f"target sentence {i}" for i in range(3)])
    assert dev["text"] == ["dev source", "dev target"]
    train, dev, _ = build_datasets(cfg(use_monotext=False, bitext_format="concat"))
    assert train["text"] == [f"source sentence {i} target sentence {i}" for i in range(3)]
    assert dev["text"] == ["dev source dev target"]


def test_monotext_only_dev_still_from_bitext():
    train, dev, sources = build_datasets(cfg(use_bitext=False))
    assert list(sources) == ["monotext"] and train["text"] == [f"mono line {i}" for i in range(4)]
    assert dev["text"] == ["dev source", "dev target"]                                 # dev follows bitext_format


def test_mix_keeps_every_example_and_the_ratio():
    c = cfg(monotext_ratio=0.25, n_mono=200, n_pairs=300)                               # 200 mono vs 600 bitext examples
    train, _, sources = build_datasets(c)
    counts = Counter(train["text"])
    assert set(counts) == set(sources["monotext"]["text"]) | set(sources["bitext"]["text"])   # nothing dropped
    mono_share = sum(n for t, n in counts.items() if t.startswith("mono")) / len(train)
    assert 0.2 < mono_share < 0.3, mono_share
    again = mix(sources["monotext"], sources["bitext"], 0.25, 42)
    assert again["text"] == train["text"]                                             # same seed, same order
    assert mix(sources["monotext"], sources["bitext"], 0.25, 7)["text"] != train["text"]


def test_check_model_dev_loss():
    history = [{"loss": 2.0, "step": 10}, {"eval_loss": 1.5, "step": 20}, {"eval_loss": 1.2, "step": 40}]
    assert check_model_dev_loss(history, 40, 1.2000001) == {"model_dev_scores": {"eval_loss": 1.2000001},
                                                             "model_dev_check": "ok"}
    assert check_model_dev_loss(history, 40, 1.5)["model_dev_check"].startswith("MISMATCH")   # e.g. last step's weights
    assert check_model_dev_loss(history, 30, 1.2)["model_dev_check"].startswith("unchecked")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
