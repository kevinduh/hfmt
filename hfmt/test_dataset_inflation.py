"""Tests for the dataset.yaml template inflation + validation helpers in sft_translation.py
(feature 06_hydra_config_directory_hierarchy).

Pure functions -- no GPU/torch/datasets needed (those imports live inside main()). Hydra/
OmegaConf + yaml are imported at module load, so the test env needs them. Each test builds its
own temp dataset template + split files, so they run under pytest or standalone.

Run with pytest, or standalone: `python hfmt/test_dataset_inflation.py`.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omegaconf import OmegaConf  # noqa: E402

import sft_translation as S  # noqa: E402
from hydra_config import Config  # noqa: E402


# A comment carrying a Hydra-interpolation lookalike (${...}) + a bare '$' -- must be harmless
# because inflation parses YAML first (dropping comments) and only substitutes string values.
TEMPLATE = """\
# comment with a ${hydra.lookalike} and a bare $ sign -- must not break inflation
train:
  src: [$DATA_ROOT/train.$SRC-$TRG.$SRC]
  trg: [$DATA_ROOT/train.$SRC-$TRG.$TRG]
dev:
  src: [$DATA_ROOT/dev.$SRC-$TRG.$SRC]
  trg: [$DATA_ROOT/dev.$SRC-$TRG.$TRG]
test:
  src: [$DATA_ROOT/test.$SRC-$TRG.$SRC]
  trg: [$DATA_ROOT/test.$SRC-$TRG.$TRG]
"""


def _make(tmp, make_test_files=False, dataset_file="dataset.yaml", template=TEMPLATE):
    """Write a dataset template + train/dev (and optionally test) split files; return a cfg."""
    with open(os.path.join(tmp, dataset_file), "w") as f:
        f.write(template)
    data_root = os.path.join(tmp, "data")
    os.makedirs(data_root, exist_ok=True)
    splits = ["train", "dev"] + (["test"] if make_test_files else [])
    for split in splits:
        for lang in ("fr", "en"):
            with open(os.path.join(data_root, f"{split}.fr-en.{lang}"), "w") as f:
                f.write("x\n")
    cfg = OmegaConf.structured(Config)
    cfg.data.src = "fr"
    cfg.data.trg = "en"
    cfg.data.data_root = data_root
    cfg.data.dataset_dir = tmp
    cfg.data.dataset_file = dataset_file
    cfg.data.instruction = "Translate French to English"
    return cfg, data_root


# --- inflation ------------------------------------------------------------------

def test_inflate_train_dev_paths_and_src_ne_trg():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, root = _make(tmp)
        S.populate_data_paths(cfg)
        assert list(cfg.data.train.src) == [os.path.join(root, "train.fr-en.fr")]
        assert list(cfg.data.train.trg) == [os.path.join(root, "train.fr-en.en")]
        assert list(cfg.data.dev.src) == [os.path.join(root, "dev.fr-en.fr")]
        assert list(cfg.data.dev.trg) == [os.path.join(root, "dev.fr-en.en")]
        # typo regression (request template had train.trg=$SRC): src and trg must differ.
        assert list(cfg.data.train.src) != list(cfg.data.train.trg)


def test_comment_with_interp_syntax_is_harmless():
    # a ${...} / bare $ in a template COMMENT must not break substitution (YAML parsed first).
    with tempfile.TemporaryDirectory() as tmp:
        cfg, root = _make(tmp)  # TEMPLATE has ${hydra.lookalike} + a bare $ in a comment
        S.populate_data_paths(cfg)
        assert list(cfg.data.train.src) == [os.path.join(root, "train.fr-en.fr")]


# --- optional test split --------------------------------------------------------

def test_test_present_when_files_exist():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, root = _make(tmp, make_test_files=True)
        S.populate_data_paths(cfg)
        assert cfg.data.test is not None
        assert list(cfg.data.test.src) == [os.path.join(root, "test.fr-en.fr")]
        assert list(cfg.data.test.trg) == [os.path.join(root, "test.fr-en.en")]


def test_test_none_when_files_missing():
    # test block present in the template, but the files don't exist -> skip (test is None).
    with tempfile.TemporaryDirectory() as tmp:
        cfg, root = _make(tmp, make_test_files=False)
        S.populate_data_paths(cfg)
        assert cfg.data.test is None


def test_test_none_when_block_absent():
    tmpl = ("train:\n  src: [$DATA_ROOT/train.$SRC-$TRG.$SRC]\n  trg: [$DATA_ROOT/train.$SRC-$TRG.$TRG]\n"
            "dev:\n  src: [$DATA_ROOT/dev.$SRC-$TRG.$SRC]\n  trg: [$DATA_ROOT/dev.$SRC-$TRG.$TRG]\n")
    with tempfile.TemporaryDirectory() as tmp:
        cfg, root = _make(tmp, template=tmpl)
        S.populate_data_paths(cfg)
        assert cfg.data.test is None


# --- validation (R10) -----------------------------------------------------------

def test_unknown_placeholder_in_value_errors():
    bad = ("train:\n  src: [$DATA_ROOT/x.$BOGUS]\n  trg: [$DATA_ROOT/y.$TRG]\n"
           "dev:\n  src: [$DATA_ROOT/d.fr]\n  trg: [$DATA_ROOT/d.en]\n")
    with tempfile.TemporaryDirectory() as tmp:
        cfg, root = _make(tmp, template=bad)
        try:
            S.populate_data_paths(cfg)
            assert False, "expected ValueError on unknown placeholder"
        except ValueError as e:
            assert "BOGUS" in str(e) or "cannot inflate" in str(e)


def test_missing_dataset_dir_errors():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, root = _make(tmp)
        cfg.data.dataset_dir = "???"  # OmegaConf MISSING sentinel
        try:
            S.populate_data_paths(cfg)
            assert False, "expected ValueError on missing dataset_dir"
        except ValueError as e:
            assert "dataset_dir" in str(e)


def test_dataset_file_without_yaml_suffix_errors():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, root = _make(tmp)
        cfg.data.dataset_file = "dataset"  # no .yaml suffix
        try:
            S.resolve_dataset_template(cfg.data)
            assert False, "expected ValueError on missing .yaml suffix"
        except ValueError as e:
            assert ".yaml" in str(e)


def test_dataset_file_with_path_errors():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, root = _make(tmp)
        cfg.data.dataset_file = "sub/dataset.yaml"  # must be a bare basename
        try:
            S.resolve_dataset_template(cfg.data)
            assert False, "expected ValueError on path-bearing dataset_file"
        except ValueError as e:
            assert "bare filename" in str(e) or "path" in str(e)


def test_nonexistent_template_errors():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, root = _make(tmp)
        cfg.data.dataset_file = "nope.yaml"
        try:
            S.resolve_dataset_template(cfg.data)
            assert False, "expected FileNotFoundError on missing template"
        except FileNotFoundError as e:
            assert "not found" in str(e)


def test_missing_data_root_errors():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, root = _make(tmp)
        cfg.data.data_root = "???"  # MISSING
        try:
            S.populate_data_paths(cfg)
            assert False, "expected ValueError on missing data_root"
        except ValueError as e:
            assert "data_root" in str(e)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
