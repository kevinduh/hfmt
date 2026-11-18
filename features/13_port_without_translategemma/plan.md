# Plan: port `hydra3-translategemma` onto `hydra3-removetranslategemma` without TranslateGemma

> Status 2026-11-07: executed (phases A-F done, nothing committed). Results and every deviation from the
> feature branch are in `porting.md`; the full patch against the feature branch is `port_vs_branch.patch`.

Branch state (2026-11-07): `hydra3-removetranslategemma` is at `5402fc5`, which is exactly the
merge base of `hydra3-translategemma` (21 commits ahead, 70 files, +3747/-165). So the port is
"take every change on the feature branch, minus what exists only to support
`google/translategemma-4b-it`". Nothing is committed; the developer commits.

## What the feature branch contains (by theme)

| Theme | Commits | Verdict |
|---|---|---|
| Prompt-format registry (`hfmt/prompt_formats.py`, `prompt_format` schema key, launch checks, `fix_chat_template`, `load_tokenizer(checkpoint, prompt_format)`) | 6b40b39, ce24bac, a0f9027 | **Drop.** Only `chat` survives; hardcode its one effect (`enable_thinking=False`). |
| Language-model-only training (`NON_LANGUAGE_MODULES`, `resolve_excluded_modules`, PEFT `exclude_modules`, vision freeze, `inspect_model` "not trained" section) | 3743e17, 9c66f6a | **Drop** (judgment call, see Risks). Qwen3.5 under `AutoModelForCausalLM` has none of these modules; built for TranslateGemma. |
| TranslateGemma experiments (8 yamls), feature 13 docs (`features/13_translategemma_prompt/`), feature 13 test cases | 17140af, 8ac5f9a, 83941d1, … | **Drop.** |
| Qwen `enable_thinking=False` for training + decoding (TRL `chat_template_kwargs` column) | c03a1c3 | **Keep, hardcoded** as a constant in `mt_model.py`. |
| `processing_class=tokenizer` for `SFTTrainer` | ce24bac | **Keep.** Qwen3.5 ships a VL processor; without it TRL takes its multimodal path. |
| `config_token_ids` (`text_config` fallback for the pad/bos/eos diagnostic log) | ce24bac | **Keep.** Gemma 3 4B (a kept experiment) has a composite config. Reword docstring. |
| Full fine-tune checkpoint reload through `from_pretrained` (`load_checkpoint_weights`), `training_info.yaml`, `trainer_state.json`, dev check of `model/` (`check_model_dev_scores`, `DEV_SCORE_TOLERANCE`), dev scores in `log_history` | 2eeed12, 030cead, d63852b | **Keep** ("proper model saving"). |
| `bleu_13a_lowercase` metric everywhere (`score`, `DECODE_METRICS`, W&B, `run_sacrebleu.py`) | 0cc394f | **Keep.** |
| W&B privacy (`wandb_privacy_settings`, `train_seq2seq.py` env vars), `.gitignore` for `_results/`, `sync.sh` | 397f60a | **Keep.** |
| Slurm: `--requeue --open-mode=append`, `job.sh` GPU wait + requeue→resume, per-folder job names for resume/inf, `sleep 3` in the inf loop | 9aa8fb3 | **Keep.** |
| `tools/run_status.py` + tests, `tools/delete_checkpoints.sh`, `gpu_free.sh`/`run_on_free_gpu.sh` wording | 43e4fb1, 390f84d, 9f000fc, 030cead | **Keep** unchanged (no TranslateGemma references). |
| `conf/train/{shortrun,midrun,lora_anneal_short,full_finetune_short}.yaml`, `conf/peft/lora_qwen3_5_all_linear.yaml`, `conf/peft/lora_gemma3_all_linear.yaml` | 17140af, 78ac136, 0036b28, e0bf36d | **Keep**; strip TranslateGemma/`NON_LANGUAGE_MODULES` from the two comment headers. |
| All Qwen3.5 / Gemma 3 experiments (fr-en, zh-en), `egs/mmtc/zh-en/dataset.yaml` + `local_setting.sh.example`, deletion of the stale `zh-en/{sft1,inf}.sh` | 17140af, 8ac5f9a, 94fd4d7, 78ac136, e0bf36d | **Keep**; remove the `prompt_format: chat` line (28 files) and TranslateGemma mechanism references in the two Gemma 3 headers. |
| Docs: `CLAUDE.md`, `README.md`, `decisions.md` | several | **Keep** every non-TranslateGemma addition; rewrite the prompting paragraph; drop the two TranslateGemma `decisions.md` sections, keep a short record of what was kept from feature 13 and why. |

## Resulting design of the prompt path (replaces the registry)

- `mt_model.format_input_prompt(instruction, text)` stays the single prompt builder (as on this branch).
- New constant `mt_model.CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}`, passed to every
  `apply_chat_template` call: dev decoding, test decoding, inference, and training via TRL's
  per-example `chat_template_kwargs` column (so prompt and prompt+completion render identically).
  Templates without the variable (Gemma) ignore it.
- `load_tokenizer(checkpoint)` keeps its one-argument signature; no template patching.
- Schema: `instruction: str = MISSING` (back to required, non-null); no `prompt_format` key, so
  saved `run_config.yaml`s from this branch stay loadable and the "not backward compatible" caveat
  in `CLAUDE.md` is not triggered by this port.

## Code-quality improvements, in scope only

1. `inference_on_eval_data` takes `instruction` again (not a closure) and no longer needs the
   `template_kwargs=None` default: the kwargs are a module constant.
2. `sft_translation.py`: the LoRA/full branches lose the excluded-module bookkeeping; the
   full-fine-tune log keeps the trainable/total count format of the LoRA branch (uniform).
3. `DECODE_METRICS` in `sft_translation.py` and `tools/run_status.py` are both derived from
   the metric names `score()` returns, documented once.
4. `test_grid.py`: port the two new tests (`--requeue`, per-folder job names); drop `test_prompt_format`.

## Risks and open questions (surfaced, decided as stated)

- **Dropping the vision-tower freeze changes Gemma 3 4B LoRA runs** (`sft_lora_gemma-3-4b-it_all-linear`,
  both pairs): the adapter will also wrap the vision tower's `q_proj/k_proj/v_proj`. With no
  `pixel_values` the tower never runs, so those adapters get no gradient and the translation result is
  unchanged, but the trainable-parameter count and adapter size grow and `tools/inspect_model.py`
  lists the vision Linear layers as candidates again. Existing run folders are unaffected
  (`PeftModel.from_pretrained` rebuilds the wrapping from `adapter_config.json`). If this matters,
  cherry-pick `3743e17` + `9c66f6a` back; they are self-contained.
- **`prompt_format` removal vs. saved configs.** Runs saved on the feature branch carry
  `prompt_format: chat` (or `translategemma`) in `run_config.yaml`. On this branch the schema has no
  such key, so `launch.sh inf|merge|sft <run_config.yaml>` on those folders fails Hydra's struct
  check. Fix: delete the line by hand (documented in `porting.md`). TranslateGemma run folders
  cannot be used on this branch at all.
- **Untracked working-tree files** `hfmt/test_prompt_formats.py` and `test.py` import the removed
  module and will break `pytest` collection. They are untracked on both branches; not deleted here
  (developer's call), flagged in `porting.md`.
- **Tests that need torch** (`hfmt/test_mt_model.py`, `hfmt/test_sft_translation.py`, both
  untracked) can't run in this sandbox; `test_grid.py` and `tools/test_run_status.py` can.
- `features/13_translategemma_prompt/` is not ported; this plan reuses index 13.

## Execution order

A. Wholesale copies from `hydra3-translategemma` (no edits): tools, conf/train, `conf/peft/lora_qwen3_5_all_linear.yaml`,
   `egs/job.sh`, `egs/launch.sh`, `.gitignore`, `analysis/run_sacrebleu.py`, `hfmt/train_seq2seq.py`,
   `egs/mmtc/zh-en/{dataset.yaml,local_setting.sh.example}`, delete `egs/mmtc/zh-en/{sft1,inf}.sh`,
   all non-TranslateGemma experiment yamls.
B. Hand-ported Python: `hfmt/mt_model.py`, `hfmt/sft_translation.py`, `hfmt/hydra_config.py`,
   `hfmt/grid.py`, `hfmt/inf_translation.py`, `tools/inspect_model.py`, `hfmt/test_grid.py`.
C. Yaml cleanup: strip `prompt_format: chat`; fix the Gemma 3 and peft comment headers.
D. Docs: `CLAUDE.md`, `README.md`, `decisions.md`.
E. Verify: `pytest hfmt/test_grid.py tools/test_run_status.py`, `python -m pyflakes`/compile on
   edited files, `git grep` for leftovers (`prompt_format`, `translategemma`, `NON_LANGUAGE`, `fix_chat_template`).
F. Write `porting.md` with the diffs highlighted.
