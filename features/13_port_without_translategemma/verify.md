# Verifying the port on the cluster

The port is a refactor that must change nothing for Qwen3.5, so the strongest tests compare against
runs that already exist from `hydra3-translategemma` rather than judging a new BLEU on its own.
Steps in order of cost; stop at the first mismatch.

## 1. Reproduce an existing run's test decoding (no training)

Pick a finished Qwen3.5-4B run folder from the feature branch: 4B is the model whose template thinks
by default, so this is where `CHAT_TEMPLATE_KWARGS` (`enable_thinking=False`) matters. Its saved
config carries a `prompt_format` line the schema no longer has; remove it, decode into a new inference
folder, and compare byte for byte with the predictions training wrote at its end:

```bash
rc=egs/mmtc/fr-en/outputs/<a finished 4B run>/run_config.yaml
sed -i '/^prompt_format: chat$/d' $rc
bash egs/launch.sh inf $rc inf_name=port_check
# when the job is done:
run=egs/mmtc/fr-en/outputs/<run>
cmp $run/test1.pred.trg $run/inf/port_check/test1.pred.trg
diff <(python -m json.tool $run/scores.json) <(python -m json.tool $run/inf/port_check/scores.json)
```

Identical files prove tokenizer loading, prompt building, the chat-template kwargs, adapter loading
and scoring all behave as before. `<think>` tags in the predictions mean the thinking switch is broken
(predictions are local files: inspect them locally, never upload them). Repeat for one finished full
fine-tune run to cover `load_trained_model`'s full-weights path.

## 2. One short training rerun, compared against its original

Rerun a feature-branch run that already exists, with a new `run_name` and the same seed, then compare
the early `eval_loss` values and the dev BLEU curve in the two `trainer_state.json` files. The anneal
sweep pinned to one cell is the cheapest 4B option that still exercises everything (about 50 steps,
dev decoding every 10):

```bash
bash egs/launch.sh sft egs/mmtc/fr-en/sft_lora_qwen-3.5-4b_all-linear_anneal.sweep.yaml \
    train.learning_rate=1e-4 train.max_steps=50 run_name=port_check_lora_4b
```

Losses at steps 5 to 20 should match the original run to a few thousandths; dev BLEU within noise.

Also launch the fr-en 0.8B full fine-tune with a new `run_name` (150 steps, small disk):

```bash
bash egs/launch.sh sft egs/mmtc/fr-en/sft_qwen-3.5-0.8b.yaml run_name=port_check_full_0.8b
```

This is the only cheap way to exercise `load_checkpoint_weights` (the best-checkpoint reload) and the
end-of-training dev check of `model/`.

## 3. Read the results with the new tooling

```bash
python tools/run_status.py egs/mmtc/fr-en/outputs
```

For the two new runs, check:

- `model/training_info.yaml` has `model_dev_check: ok`, and `run_status.py` reports no failed checks.
- The training log has zero occurrences of TRL's "Mismatch between tokenized prompt" warning.
- The log line `apply_chat_template kwargs (training and decoding): {'enable_thinking': False}` is present.
- The `dev.step_*.pred` files hold translations, not reasoning text.

## What these tests don't cover

Gemma 3: the all-linear adapter now also covers the vision tower's `q/k/v_proj` (the vision freeze
was dropped with feature 13, see `porting.md` §1). A Gemma 3 run shows a larger trainable-parameter
count in its log but the same scores, since the tower never runs without images.
