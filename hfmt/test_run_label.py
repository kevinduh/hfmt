"""Tests for the run-label helpers in sft_translation.py (feature 03_output_wandb_align).

Pure functions -- no GPU, no torch/transformers needed (those imports are guarded or live
inside main()). Hydra/OmegaConf are imported at module load, so the test env needs them.

Run with pytest, or standalone: `python hfmt/test_run_label.py`.
"""

import os
import re
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sft_translation as S  # noqa: E402


def _cfg(experiment="mmtc_fr-en_sft1", group="mmtc_fr-en_sft1", tags=("sft", "fr-en"), name=None):
    """Minimal stand-in for the Hydra cfg fields derive_run_identity reads."""
    return SimpleNamespace(
        experiment=experiment,
        wandb=SimpleNamespace(group=group, tags=list(tags), name=name),
    )


SWEEP_OUT = "/x/outputs/mmtc_fr-en_sft1/sweep-2026-10-08_15-47-06/lora_r-8__lora_target-qv__seed-37"
SINGLE_OUT = "/x/outputs/mmtc_fr-en_sft1/2026-10-08_15-47-06"

# Hydra's default override_dirname separators (pairs ',', key/value '=').
GRID = "model.lora_r=16,model.lora_target=all-linear,train.learning_rate=0.0002,train.seed=37"
SHELL_SAFE = re.compile(r"\A[A-Za-z0-9._-]*\Z")


# --- build_run_label ------------------------------------------------------------

def test_descriptive_grid_label():
    # Group prefixes stripped; leaf keys verbatim (lora kept); pairs joined by '__'.
    assert S.build_run_label(GRID) == (
        "lora_r-16__lora_target-all-linear__learning_rate-0.0002__seed-37"
    )


def test_keeps_lora_prefix_and_underscores():
    # 'lora' must survive and leaf underscores are preserved (not collapsed to '-').
    label = S.build_run_label("model.lora_r=8,model.lora_dropout=0.1")
    assert label == "lora_r-8__lora_dropout-0.1"
    assert "lora_r" in label and "lora_dropout" in label


def test_order_is_preserved_from_override_dirname():
    assert S.build_run_label("train.seed=37,model.lora_r=8") == "seed-37__lora_r-8"


def test_data_key_dropped_for_confidentiality():
    # A data *path* must never enter the label (== dir name == W&B run name).
    label = S.build_run_label("model.lora_r=8,data.train_yaml=/proprietary/corpus.yaml")
    assert label == "lora_r-8"
    assert "proprietary" not in label and "corpus" not in label


def test_bare_data_group_dropped():
    assert S.build_run_label("data=mmtc_fr-en,model.lora_r=8") == "lora_r-8"


def test_slash_bearing_value_dropped():
    # Any '/'-bearing value is dropped (would fracture the dir) even on a non-data key.
    assert S.build_run_label("model.checkpoint=/path/to/model,model.lora_r=8") == "lora_r-8"


def test_structural_selectors_dropped():
    tok = "experiment=mmtc_fr-en_sft1,sweep=coarse,hydra.launcher=slurm,model.lora_r=8"
    assert S.build_run_label(tok) == "lora_r-8"


def test_append_and_tilde_prefixes_stripped():
    # '+'/'~' are stripped before matching, so '+experiment' is still recognised and dropped,
    # while a real '~'-deleted param override is kept.
    assert S.build_run_label("+experiment=x,+model.lora_r=8") == "lora_r-8"
    assert S.build_run_label("~train.optim=adamw") == "optim-adamw"


def test_empty_falls_back_to_job_num():
    assert S.build_run_label("") == "job0"
    assert S.build_run_label("", job_num="3") == "job3"
    assert S.build_run_label("experiment=x", job_num=5) == "job5"


def test_value_underscores_preserved():
    assert S.build_run_label("train.optim=adamw_torch_fused") == "optim-adamw_torch_fused"


def test_label_is_shell_safe():
    # Even with hostile characters in a value, the label stays within [A-Za-z0-9._-].
    label = S.build_run_label("train.instruction=a b;c|d,model.lora_r=8")
    assert SHELL_SAFE.match(label), label


# --- _strict --------------------------------------------------------------------

def test_strict_keeps_safe_chars():
    assert S._strict("all-linear") == "all-linear"
    assert S._strict("adamw_torch_fused") == "adamw_torch_fused"
    assert S._strict("0.0002") == "0.0002"


def test_strict_collapses_and_trims():
    assert S._strict("a b") == "a-b"
    assert S._strict("x=1,y=2") == "x-1-y-2"
    assert S._strict("--lead trail--") == "lead-trail"


# --- derive_run_identity --------------------------------------------------------

def test_identity_sweep_name_is_dir_basename():
    rid, name, group, tags = S.derive_run_identity(_cfg(), SWEEP_OUT, is_multirun=True)
    # The W&B run name IS the run-dir basename (single source of truth).
    assert name == "lora_r-8__lora_target-qv__seed-37"
    assert name == os.path.basename(SWEEP_OUT)


def test_identity_sweep_group_from_parent():
    _, _, group, _ = S.derive_run_identity(_cfg(), SWEEP_OUT, is_multirun=True)
    assert group == "mmtc_fr-en_sft1-sweep-2026-10-08_15-47-06"


def test_identity_sweep_tags_split_from_label():
    _, _, _, tags = S.derive_run_identity(_cfg(), SWEEP_OUT, is_multirun=True)
    assert tags == ["sft", "fr-en", "sweep", "lora_r-8", "lora_target-qv", "seed-37"]


def test_identity_sweep_run_id_unique_and_stable():
    rid1, *_ = S.derive_run_identity(_cfg(), SWEEP_OUT, is_multirun=True)
    rid2, *_ = S.derive_run_identity(_cfg(), SWEEP_OUT, is_multirun=True)
    assert rid1 == rid2  # stable across a requeue of the same job (same path)
    assert rid1 == "mmtc_fr-en_sft1_sweep-2026-10-08_15-47-06_lora_r-8__lora_target-qv__seed-37"


def test_identity_single_run():
    rid, name, group, tags = S.derive_run_identity(_cfg(group=None), SINGLE_OUT, is_multirun=False)
    assert group == "mmtc_fr-en_sft1"  # falls back to experiment
    assert name == "mmtc_fr-en_sft1-2026-10-08_15-47-06"
    assert rid == "mmtc_fr-en_sft1_2026-10-08_15-47-06"
    assert tags == ["sft", "fr-en"]  # no sweep tag


def test_identity_respects_wandb_name_override():
    _, name, _, _ = S.derive_run_identity(_cfg(name="my-run"), SWEEP_OUT, is_multirun=True)
    assert name == "my-run"


def test_identity_no_slashes_or_data_anywhere():
    # Labels are data-free by construction; identity must stay path-safe / content-free.
    rid, name, group, tags = S.derive_run_identity(_cfg(), SWEEP_OUT, is_multirun=True)
    for s in [rid, name, group, *tags]:
        assert "/" not in s
        assert SHELL_SAFE.match(s), s


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
