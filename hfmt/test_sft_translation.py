"""Tests for the training-record helpers in hfmt/sft_translation.py (dev scores in log_history,
model/training_info.yaml). No GPU or training deps needed: the Trainer state and args are stand-ins.

Run standalone (pytest isn't installed in the hfmt env):
    python hfmt/test_sft_translation.py
"""

import os
import sys
from types import SimpleNamespace

from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hfmt.sft_translation import add_dev_scores_to_log_history, check_model_dev_scores, training_info  # noqa: E402

SCORES = {"bleu": 41.2, "bleu_13a_lowercase": 43.0, "chrf": 65.1, "ter": 48.3}


def test_dev_scores_join_the_evaluation_entry():
    log_history = [{"loss": 1.2, "step": 40}, {"eval_loss": 0.9, "eval_runtime": 3.0, "epoch": 1.3, "step": 50}]
    add_dev_scores_to_log_history(log_history, 50, SCORES)
    assert len(log_history) == 2
    assert log_history[-1] == {"eval_loss": 0.9, "eval_runtime": 3.0, "epoch": 1.3, "step": 50,
                               "eval_bleu": 41.2, "eval_bleu_13a_lowercase": 43.0, "eval_chrf": 65.1, "eval_ter": 48.3}


def test_dev_scores_get_their_own_entry_without_one():
    for log_history in ([], [{"loss": 1.2, "step": 50}], [{"eval_loss": 0.9, "step": 40}]):
        before = [dict(entry) for entry in log_history]
        add_dev_scores_to_log_history(log_history, 50, SCORES)
        assert log_history[:-1] == before
        assert log_history[-1] == {"eval_bleu": 41.2, "eval_bleu_13a_lowercase": 43.0, "eval_chrf": 65.1,
                                   "eval_ter": 48.3, "step": 50}


def state(global_step, best_global_step, best_metric, max_steps=400):
    checkpoint = f"/runs/sweep-0/checkpoint-{best_global_step}" if best_global_step else None
    return SimpleNamespace(global_step=global_step, max_steps=max_steps, best_metric=best_metric,
                           best_global_step=best_global_step, best_model_checkpoint=checkpoint)


def args(load_best_model_at_end=True):
    return SimpleNamespace(load_best_model_at_end=load_best_model_at_end, metric_for_best_model="eval_bleu")


def test_training_info_early_stopped_best_loaded():
    assert training_info(state(350, 200, 41.2), args()) == {
        "model_global_step": 200, "metric_for_best_model": "eval_bleu", "best_metric": 41.2,
        "best_global_step": 200, "global_step": 350, "max_steps": 400, "stopped_early": True}


def test_training_info_last_step_kept():
    info = training_info(state(400, 200, 41.2), args(load_best_model_at_end=False))
    assert info["model_global_step"] == 400 and info["best_global_step"] == 200 and not info["stopped_early"]
    info = training_info(state(400, None, None), args())                 # never evaluated: nothing to load
    assert info["model_global_step"] == 400 and info["best_metric"] is None


def test_training_info_saves_as_yaml():
    info = training_info(state(350, 200, 41.2), args())
    info.update(check_model_dev_scores([{"eval_bleu": 41.2, "step": 200}], 200, SCORES))
    assert OmegaConf.to_container(OmegaConf.create(OmegaConf.to_yaml(OmegaConf.create(info)))) == info


def test_model_dev_check_against_the_steps_dev_bleu():
    history = [{"eval_loss": 0.9, "eval_bleu": 34.33, "step": 50}, {"eval_loss": 1.1, "step": 100}]
    close = {**SCORES, "bleu": 34.0}
    assert check_model_dev_scores(history, 50, close) == {"model_dev_scores": close, "model_dev_check": "ok"}
    check = check_model_dev_scores(history, 50, {**SCORES, "bleu": 24.74})["model_dev_check"]   # the Qwen3.5 2B zh-en case
    assert check.startswith("MISMATCH") and "24.74" in check and "34.33" in check and "step 50" in check
    assert check_model_dev_scores(history, 100, SCORES)["model_dev_check"].startswith("unchecked")   # no dev decode there
    assert check_model_dev_scores(history, 50, close, tolerance=0.1)["model_dev_check"].startswith("MISMATCH")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
