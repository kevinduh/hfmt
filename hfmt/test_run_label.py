"""Tests for the run-label helpers in sft_translation.py (feature 03_output_wandb_align;
label hardening in feature 04_config_structure_change).

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


def _cfg(experiment="sft1", group=None, tags=("sft",), name=None,
         dataset_name="mmtc", src="fr", trg="en"):
    """Minimal stand-in for the Hydra cfg fields derive_run_identity reads.

    Feature 06 naming: ``experiment`` is the yaml's file stem (injected by run.sh) and yamls no
    longer set ``wandb.group`` (it falls back to the experiment); the dataset/lang-pair context
    comes from ``data.dataset_name``/``src``/``trg``.
    """
    return SimpleNamespace(
        experiment=experiment,
        data=SimpleNamespace(dataset_name=dataset_name, src=src, trg=trg),
        wandb=SimpleNamespace(group=group, tags=list(tags), name=name),
    )


OUTPUTS = "/x/egs/mmtc/fr-en/outputs"
SWEEP_OUT = f"{OUTPUTS}/sft1/sweep-2026-10-08_15-47-06/lora_r-8__lora_target-qv__seed-37"
SINGLE_OUT = f"{OUTPUTS}/sft1/2026-10-08_15-47-06"
# Plain run via run.sh: --multirun (so under sweep-<ts>/) but no swept params, so the dir
# leaf is the experiment name (build_run_label fallback) -- feature 04.
PLAIN_OUT = f"{OUTPUTS}/sft1_fullft/sweep-2026-10-08_15-47-06/sft1_fullft"

# Hydra's default override_dirname separators (pairs ',', key/value '=').
GRID = "lora.lora_r=16,lora.lora_target=all-linear,train.learning_rate=0.0002,train.seed=37"
SHELL_SAFE = re.compile(r"\A[A-Za-z0-9._-]*\Z")


# --- build_run_label ------------------------------------------------------------

def test_descriptive_grid_label():
    # Group prefixes stripped; leaf keys verbatim (lora kept); pairs joined by '__'.
    assert S.build_run_label(GRID) == (
        "lora_r-16__lora_target-all-linear__learning_rate-0.0002__seed-37"
    )


def test_keeps_lora_prefix_and_underscores():
    # 'lora' must survive and leaf underscores are preserved (not collapsed to '-').
    label = S.build_run_label("lora.lora_r=8,lora.lora_dropout=0.1")
    assert label == "lora_r-8__lora_dropout-0.1"
    assert "lora_r" in label and "lora_dropout" in label


def test_order_is_preserved_from_override_dirname():
    assert S.build_run_label("train.seed=37,lora.lora_r=8") == "seed-37__lora_r-8"


def test_group_choice_labels_by_file_name():
    # Feature 04: sweeping whole component files labels by the selected YAML option
    # (the file name), not the value inside it. The group name is kept (no dot to strip).
    assert S.build_run_label("basemodel=qwen2.5-7b") == "basemodel-qwen2.5-7b"
    assert S.build_run_label("basemodel=qwen2.5-1.5b,lora=aggressive") == (
        "basemodel-qwen2.5-1.5b__lora-aggressive"
    )


def test_data_path_dropped_for_confidentiality():
    # A data *path* must never enter the label (== dir name == W&B run name).
    label = S.build_run_label("lora.lora_r=8,data.train_yaml=/proprietary/corpus.yaml")
    assert label == "lora_r-8"
    assert "proprietary" not in label and "corpus" not in label


def test_bare_data_group_kept():
    # Feature 04: a bare data=<name> group choice is a config-file NAME (not a path or
    # content), so it IS kept -- sweeping across data configs is labelled by file name.
    assert S.build_run_label("data=mmtc_fr-en,lora.lora_r=8") == "data-mmtc_fr-en__lora_r-8"


def test_slash_bearing_value_dropped():
    # Any '/'-bearing value is dropped (would fracture the dir) even on a non-data key.
    assert S.build_run_label("basemodel.checkpoint=/path/to/model,lora.lora_r=8") == "lora_r-8"


def test_structural_selectors_dropped():
    # experiment/sweep/sweep.enabled/hydra* carry no hyperparameter info.
    tok = "experiment=sft1,sweep=coarse,sweep.enabled=true,hydra.launcher=slurm,lora.lora_r=8"
    assert S.build_run_label(tok) == "lora_r-8"


def test_nested_experiment_selector_dropped():
    # Feature 06: the experiment is chosen as a nested group +<dataset>/<lang>=<name>, whose KEY
    # contains '/'. It is a structural choice, not a hyperparameter, so it must be dropped (else
    # it would leak into the run-dir / W&B name and flip is_sweep for a plain run).
    assert S._label_pairs("+mmtc/fr-en=sft1") == []
    assert S._label_pairs("+mmtc/fr-en=sft1,train.seed=37,lora.lora_r=8") == ["seed-37", "lora_r-8"]
    # dropped at any nesting depth
    assert S._label_pairs("+mmtc/fr-en/variant=sft1,train.learning_rate=2e-4") == ["learning_rate-2e-4"]


def test_nested_selector_plain_run_falls_back_to_experiment():
    # A plain run via run.sh (nested selector only, no swept axes) has no informative tokens,
    # so the label falls back to the experiment name -- dir/W&B name stay meaningful.
    assert S.build_run_label("+mmtc/fr-en=sft1", experiment="sft1") == "sft1"


def test_append_and_tilde_prefixes_stripped():
    # '+'/'~' are stripped before matching, so '+experiment' is still recognised and dropped,
    # while a real '~'-deleted param override is kept.
    assert S.build_run_label("+experiment=x,+lora.lora_r=8") == "lora_r-8"
    assert S.build_run_label("~train.optim=adamw") == "optim-adamw"


def test_empty_falls_back_to_job_num():
    assert S.build_run_label("") == "job0"
    assert S.build_run_label("", job_num="3") == "job3"
    assert S.build_run_label("experiment=x", job_num=5) == "job5"


def test_fallback_uses_experiment_when_given():
    # Feature 04: a plain run (no informative tokens) is named by the experiment, not jobN,
    # so run.sh single runs get a meaningful dir/W&B name instead of "job0".
    assert S.build_run_label("", experiment="sft1_fullft") == "sft1_fullft"
    # Tokens that all get dropped (data path) still fall back to the experiment.
    assert S.build_run_label("data.train_yaml=/x/y.yaml", experiment="exp") == "exp"
    # An informative token still wins over the experiment fallback.
    assert S.build_run_label("lora.lora_r=8", experiment="exp") == "lora_r-8"


def test_value_underscores_preserved():
    assert S.build_run_label("train.optim=adamw_torch_fused") == "optim-adamw_torch_fused"


def test_label_is_shell_safe():
    # Even with hostile characters in a value, the label stays within [A-Za-z0-9._-].
    label = S.build_run_label("train.instruction=a b;c|d,lora.lora_r=8")
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

def test_identity_sweep_name_prefixes_experiment():
    rid, name, group, tags = S.derive_run_identity(_cfg(), SWEEP_OUT, is_multirun=True, is_sweep=True)
    # Feature 05: the W&B run name prefixes the experiment onto the label so flat run lists
    # stay unambiguous across experiments that sweep the same axis. The run *directory* leaf
    # keeps the bare label, so name and dir basename differ by exactly that prefix.
    assert name == "sft1-lora_r-8__lora_target-qv__seed-37"
    assert name == f"{_cfg().experiment}-{os.path.basename(SWEEP_OUT)}"


def test_identity_sweep_group_from_parent():
    _, _, group, _ = S.derive_run_identity(_cfg(), SWEEP_OUT, is_multirun=True, is_sweep=True)
    assert group == "sft1-sweep-2026-10-08_15-47-06"


def test_identity_sweep_tags_split_from_label():
    _, _, _, tags = S.derive_run_identity(_cfg(), SWEEP_OUT, is_multirun=True, is_sweep=True)
    # Feature 06: dataset + lang pair always lead the tags (names no longer carry them).
    assert tags == ["mmtc", "fr-en", "sft", "sweep", "lora_r-8", "lora_target-qv", "seed-37"]


def test_identity_sweep_run_id_unique_and_stable():
    rid1, *_ = S.derive_run_identity(_cfg(), SWEEP_OUT, is_multirun=True, is_sweep=True)
    rid2, *_ = S.derive_run_identity(_cfg(), SWEEP_OUT, is_multirun=True, is_sweep=True)
    assert rid1 == rid2  # stable across a requeue of the same job (same path)
    assert rid1 == "mmtc_fr-en_sft1_sweep-2026-10-08_15-47-06_lora_r-8__lora_target-qv__seed-37"


def test_identity_single_run():
    rid, name, group, tags = S.derive_run_identity(_cfg(group=None), SINGLE_OUT, is_multirun=False)
    assert group == "sft1"  # falls back to experiment
    assert name == "sft1-2026-10-08_15-47-06"
    assert rid == "mmtc_fr-en_sft1_2026-10-08_15-47-06"
    assert tags == ["mmtc", "fr-en", "sft"]  # no sweep tag


def test_identity_plain_multirun_run():
    # A plain run via run.sh is --multirun but has no sweep params: named by the experiment
    # (== dir leaf, so disk/W&B match), grouped under the base group with NO -sweep-<ts>
    # suffix, and with no 'sweep' tag (feature 04).
    cfg = _cfg(experiment="sft1_fullft", tags=("sft", "fullft"))
    rid, name, group, tags = S.derive_run_identity(cfg, PLAIN_OUT, is_multirun=True, is_sweep=False)
    assert name == "sft1_fullft"
    assert name == os.path.basename(PLAIN_OUT)        # dir basename == W&B name (invariant holds)
    assert group == "sft1_fullft"                     # base group, no date appended
    assert "sweep" not in tags
    assert tags == ["mmtc", "fr-en", "sft", "fullft"]
    # run_id carries the sweep-<ts> parent so separate launches stay distinct + requeue-stable,
    # and the dataset/lang-pair prefix so it stays unique across lang pairs (feature 06).
    assert rid == "mmtc_fr-en_sft1_fullft_sweep-2026-10-08_15-47-06_sft1_fullft"


def test_identity_respects_wandb_name_override():
    _, name, _, _ = S.derive_run_identity(_cfg(name="my-run"), SWEEP_OUT, is_multirun=True, is_sweep=True)
    assert name == "my-run"


def test_identity_no_slashes_or_data_anywhere():
    # Labels are data-free by construction; identity must stay path-safe / content-free.
    rid, name, group, tags = S.derive_run_identity(_cfg(), SWEEP_OUT, is_multirun=True, is_sweep=True)
    for s in [rid, name, group, *tags]:
        assert "/" not in s
        assert SHELL_SAFE.match(s), s


# --- feature 06 naming schema ---------------------------------------------------

def test_identity_names_carry_no_dataset_or_lang_pair():
    # The outputs path already says mmtc/fr-en; W&B names/groups must not repeat it.
    sweep = S.derive_run_identity(_cfg(experiment="sft1_refine1"), SWEEP_OUT,
                                  is_multirun=True, is_sweep=True)
    plain = S.derive_run_identity(_cfg(experiment="sft1_fullft"), PLAIN_OUT,
                                  is_multirun=True, is_sweep=False)
    for _, name, group, tags in (sweep, plain):
        for s in (name, group):
            assert "mmtc" not in s and "fr-en" not in s, s
        assert tags[:2] == ["mmtc", "fr-en"]


def test_identity_context_tags_deduplicated():
    # A hand-written fr-en tag left in an experiment yaml is not doubled.
    _, _, _, tags = S.derive_run_identity(_cfg(tags=("sft", "fr-en")), SINGLE_OUT,
                                          is_multirun=False)
    assert tags == ["mmtc", "fr-en", "sft"]


def test_identity_run_id_unique_across_lang_pairs():
    # Same experiment stem, same sweep-<ts>, same label, different lang pair: with
    # resume='allow' a shared id would silently attach one job to the other's run (R11).
    fr_en, *_ = S.derive_run_identity(_cfg(), SWEEP_OUT, is_multirun=True, is_sweep=True)
    zh_en, *_ = S.derive_run_identity(_cfg(src="zh"), SWEEP_OUT, is_multirun=True, is_sweep=True)
    other, *_ = S.derive_run_identity(_cfg(dataset_name="other"), SWEEP_OUT,
                                      is_multirun=True, is_sweep=True)
    assert len({fr_en, zh_en, other}) == 3
    assert zh_en.startswith("mmtc_zh-en_sft1_") and other.startswith("other_fr-en_sft1_")


def test_identity_without_dataset_name_keeps_lang_pair():
    # Defensive: if dataset_name was never filled in, the lang pair still tags/prefixes.
    rid, _, _, tags = S.derive_run_identity(_cfg(dataset_name=None), SINGLE_OUT,
                                            is_multirun=False)
    assert tags == ["fr-en", "sft"]
    assert rid == "fr-en_sft1_2026-10-08_15-47-06"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
