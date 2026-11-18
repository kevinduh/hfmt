# Porting document: `hydra3-translategemma` → `hydra3-removetranslategemma`, without TranslateGemma

Date: 2026-11-07. Source: `hydra3-translategemma` at `e0bf36d` (21 commits past the common base
`5402fc5`, which is this branch's HEAD). Nothing is committed; everything below is in the working
tree for the developer to review and commit (suggested commit messages at the end).

Three kinds of diff matter here, and each is shown in its own section:

1. **What the port adds to this branch** (HEAD → working tree): §2, with `git diff HEAD --stat`.
2. **What differs from the feature branch** (feature branch → working tree): §3 is the highlight.
   Every hunk there is a deliberate deviation; the complete patch is `port_vs_branch.patch`
   (35 files, 1210 lines) next to this file.
3. **What was left behind** on the feature branch: §4.

## 1. Summary

**Kept (everything that isn't TranslateGemma):** all Qwen3.5 and Gemma 3 experiments for fr-en and
zh-en (28 yamls), the zh-en `dataset.yaml` + `local_setting.sh.example`, the four `conf/train`
schedules and two `conf/peft` all-linear options, `tools/run_status.py` (+ tests) and
`tools/delete_checkpoints.sh`, the full fine-tune checkpoint reload (`load_checkpoint_weights`),
`trainer_state.json` + `model/training_info.yaml` + the measured dev check of `model/`, the
`bleu_13a_lowercase` metric, W&B privacy settings, the Slurm `--requeue`/GPU-wait/resume-on-requeue
changes and per-folder job names, `.gitignore` additions, the docs for all of it.

**Dropped (TranslateGemma support):** the `prompt_format` key and `hfmt/prompt_formats.py` registry,
the chat-template whitespace patch (`fix_chat_template`), the language-model-only training
(`NON_LANGUAGE_MODULES`, `resolve_excluded_modules`, PEFT `exclude_modules`, vision freeze), the
`[prompt_format]` argument and "not trained" sections of `tools/inspect_model.py`, 8 TranslateGemma
experiment yamls, `features/13_translategemma_prompt/`, `test_prompt_format` in `test_grid.py`, and
the two TranslateGemma sections of `decisions.md`.

**Hardcoded instead:** `mt_model.CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}`, passed to every
`apply_chat_template` call (training through TRL's per-example `chat_template_kwargs` column, dev and
test decoding, inference, `inspect_model.py`). Qwen3.5-4B's template thinks by default; 0.8B/2B and
Gemma are unaffected.

**Judgment calls (flagged, reversible):**

- *Vision-tower freeze removed.* It was feature 13-C, written for TranslateGemma. Qwen3.5 under
  `AutoModelForCausalLM` has no vision modules, so no kept Qwen run changes. The kept
  `sft_lora_gemma-3-4b-it_all-linear` experiments (fr-en, zh-en) do change: the all-linear adapter
  now also wraps the vision tower's `q_proj/k_proj/v_proj`. Without `pixel_values` the tower never
  runs, so those adapters get no gradient and the translation result is unchanged; the logged
  trainable count and the adapter size grow, and `inspect_model.py` lists vision Linear layers as
  candidates again. Existing run folders load fine (`PeftModel.from_pretrained` rebuilds the wrapping
  from `adapter_config.json`). To restore: cherry-pick `3743e17` and `9c66f6a` from the feature branch.
- *`config_token_ids` kept.* Its `text_config` fallback was written for TranslateGemma's composite
  config, but Gemma 3 4B has the same config class and it is a 10-line diagnostic helper; docstring reworded.
- *`processing_class=tokenizer` kept.* Qwen3.5 ships a VL processor; without this TRL takes its
  multimodal path. Comment reworded.

## 2. What the port adds to this branch (HEAD → working tree)

`git diff HEAD --stat` (tracked files only): 17 files changed, 437 insertions, 123 deletions.
Plus 37 new files and 2 deletions. Identical to the feature branch unless marked.

| Area | Files | vs. feature branch |
|---|---|---|
| Python | `hfmt/grid.py`, `hfmt/train_seq2seq.py` | identical |
| | `hfmt/mt_model.py`, `hfmt/sft_translation.py`, `hfmt/test_grid.py`, `tools/inspect_model.py` | **edited** (§3) |
| | `hfmt/hydra_config.py`, `hfmt/inf_translation.py` | unchanged from HEAD (their only feature-branch change was `prompt_format`) |
| Tools | `tools/run_status.py`, `tools/test_run_status.py`, `tools/delete_checkpoints.sh` (new); `tools/gpu_free.sh`, `tools/run_on_free_gpu.sh` | identical |
| Launch | `egs/job.sh`, `egs/launch.sh`, `.gitignore`, `analysis/run_sacrebleu.py` | identical |
| Config groups | `conf/train/{shortrun,midrun,lora_anneal_short,full_finetune_short}.yaml`, `conf/peft/lora_qwen3_5_all_linear.yaml` (new) | identical, except one comment word in `full_finetune_short.yaml` |
| | `conf/peft/lora_gemma3_all_linear.yaml` (new) | **edited** comment header (no `NON_LANGUAGE_MODULES`) |
| Experiments | 28 yamls under `egs/mmtc/{fr-en,zh-en}/` (26 new, `sft_lora.yaml` and `sft_lora.sweep.yaml` modified) | 28 lose the line `prompt_format: chat`; the two Gemma 3 headers, `sft_lora_qwen-3.5-9b_all-linear.yaml` and both `sft_qwen-3.5-4b.yaml` lose TranslateGemma mechanism references in comments |
| zh-en folder | `dataset.yaml`, `local_setting.sh.example` (new); `inf.sh`, `sft1.sh` deleted | identical |
| Docs | `CLAUDE.md`, `README.md`, `decisions.md` | **edited** (§3.6) |
| Feature dir | `features/13_port_without_translategemma/{request,plan,porting}.md`, `port_vs_branch.patch` | new here |

## 3. Highlighted diffs against the feature branch (`-` = feature branch, `+` = this port)

### 3.1 `hfmt/mt_model.py`

The registry import, the vision-module list/resolver and the tokenizer patch go; one constant and the
original `format_input_prompt` come back.

```diff
-from hfmt.prompt_formats import chat_template_kwargs, fix_chat_template, make_prompt_formatter
-
-NON_LANGUAGE_MODULES = ["vision_tower", "multi_modal_projector"]
-
-def resolve_excluded_modules(model, target_modules):
-    ...  (23 lines)
-
-def load_tokenizer(checkpoint, prompt_format):
+# Extra arguments for every ``apply_chat_template`` call that renders a prompt (training via TRL's
+# per-example ``chat_template_kwargs`` column, dev/test decoding, inference), so prompts render the same
+# everywhere. enable_thinking=False: Qwen3.5-4B's template thinks by default ...
+CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}
+
+def format_input_prompt(instruction_prefix, content):
+    return [{"content": instruction_prefix + "\n" + content, "role": "user"}]
+
+def load_tokenizer(checkpoint):
     tokenizer = AutoTokenizer.from_pretrained(checkpoint)
-    fix_chat_template(tokenizer, prompt_format)
```

```diff
-    load_tokenizer(cfg.checkpoint, cfg.prompt_format).save_pretrained(out_dir)
+    load_tokenizer(cfg.checkpoint).save_pretrained(out_dir)

-def inference_on_eval_data(tokenizer, model, eval_data, predictions_file, device, format_prompt, decode, text_field='text',
-                           template_kwargs=None):
-    template_kwargs = template_kwargs or {}
+def inference_on_eval_data(tokenizer, model, eval_data, predictions_file, device, instruction_prefix, decode, text_field='text'):
     ...
-            prompts = [format_prompt(s) for s in eval_batch[text_field]]
+            prompts = [format_input_prompt(instruction_prefix, s) for s in eval_batch[text_field]]
             test_inputs = tokenizer.apply_chat_template(prompts, ..., return_dict=True,
-                                                        **template_kwargs).to(device)
+                                                        **CHAT_TEMPLATE_KWARGS).to(device)
     ...  (same substitution in the truncation-check render)

 def decode_test_sets(cfg, tokenizer, model, outdir, device):
-    format_prompt = make_prompt_formatter(cfg)
     ...
-                                       format_prompt, cfg.decode, template_kwargs=chat_template_kwargs(cfg))
+                                       cfg.instruction, cfg.decode)
```

Unchanged from the feature branch: `load_checkpoint_weights`, `load_trained_model`,
`merge_trained_model`, `score` with `bleu_13a_lowercase`, the `scores.json` docstring.

### 3.2 `hfmt/sft_translation.py`

```diff
-from hfmt.mt_model import (decode_test_sets, inference_on_eval_data, load_base_model, load_checkpoint_weights, load_tokenizer,
-                           resolve_excluded_modules, score)
-from hfmt.prompt_formats import chat_template_kwargs, make_prompt_formatter
+from hfmt.mt_model import (CHAT_TEMPLATE_KWARGS, decode_test_sets, format_input_prompt, inference_on_eval_data, load_base_model,
+                           load_checkpoint_weights, load_tokenizer, score)

-DECODE_METRICS = ("eval_bleu", "eval_bleu_13a_lowercase", "eval_chrf", "eval_ter")  # added by the MT callback, only on decode steps
+# The metrics score() returns, as the MT callback adds them to the eval metrics (only on decode steps).
+DECODE_METRICS = ("eval_bleu", "eval_bleu_13a_lowercase", "eval_chrf", "eval_ter")
```

```diff
-def make_mt_callback(..., device, format_prompt, decode, interval,
-                     template_kwargs):
+def make_mt_callback(..., device, instruction_prefix, decode, interval):
     ...
-            preds = inference_on_eval_data(tokenizer, model, data, ..., device, format_prompt, decode, 'src',
-                                           template_kwargs=template_kwargs)
+            preds = inference_on_eval_data(tokenizer, model, data, ..., device, instruction_prefix, decode, 'src')
```

```diff
     ## User settings
-    tokenizer = load_tokenizer(cfg.checkpoint, cfg.prompt_format)
-    format_prompt = make_prompt_formatter(cfg)   # builder for cfg.prompt_format (validated at launch)
-    template_kwargs = chat_template_kwargs(cfg)  # e.g. enable_thinking=False; same for training and decoding
-    logging.info(f"prompt_format={cfg.prompt_format}: apply_chat_template kwargs {template_kwargs}")
+    instruction_prefix = cfg.instruction
+    logging.info(f"instruction: '{instruction_prefix}'")
+    logging.info(f"apply_chat_template kwargs (training and decoding): {CHAT_TEMPLATE_KWARGS}")
+
+    tokenizer = load_tokenizer(cfg.checkpoint)

     def preprocess_fn(samples):
-        prompt = [format_prompt(s) for s in samples["src"]]
+        prompt = [format_input_prompt(instruction_prefix, s) for s in samples["src"]]
         completion = [[{"content":t, "role":"assistant"}] for t in samples["trg"]]
-        out = {"prompt": prompt, "completion": completion}
-        # Only added when non-empty: Arrow can't store an empty struct.
-        if template_kwargs:
-            out["chat_template_kwargs"] = [dict(template_kwargs) for _ in prompt]
-        return out
+        # TRL passes an example's chat_template_kwargs to both of its renders (prompt, prompt + completion),
+        # so training prompts render as our decoding does (CHAT_TEMPLATE_KWARGS: enable_thinking=False).
+        return {"prompt": prompt, "completion": completion,
+                "chat_template_kwargs": [dict(CHAT_TEMPLATE_KWARGS) for _ in prompt]}
```

The whole excluded-modules block after `load_base_model` is gone (14 lines), as is
`exclude_modules=` in `LoraConfig` and the freeze loop in the full-fine-tune branch:

```diff
-    target_modules = list(cfg.peft.lora.target_modules) if finetune_mode in ("lora", "qlora") else []
-    excluded_subtrees, excluded_leaves = resolve_excluded_modules(model, target_modules)
-    if excluded_subtrees: ... else: logging.info("Not trained (outside the language model): none")
     ...
-            exclude_modules=excluded_leaves or None, # keep the adapter out of the vision tower
     ...
-    else: # full fine-tune of the language model (vision modules frozen just above)
-        for module in excluded_subtrees.values():
-            module.requires_grad_(False)
+    else: # full fine-tune of all parameters
         model.config.use_cache = False
         trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
         total = sum(p.numel() for p in model.parameters())
-        logging.info(f"Full fine-tuning of the language model: trainable params {trainable:,} of "
-                     f"{total:,} ({100 * trainable / total:.4f}%)")
+        logging.info(f"Full fine-tuning of the base model: trainable params {trainable:,} of {total:,} ({100 * trainable / total:.4f}%)")
```

Code-quality change in scope: the test-score W&B loop derives its keys from `DECODE_METRICS`
instead of repeating the tuple:

```diff
-                for k in ("bleu", "bleu_13a_lowercase", "chrf", "ter"):
+                for k in (m.removeprefix("eval_") for m in DECODE_METRICS):
```

(`str.removeprefix` needs Python ≥ 3.9; `install/install_hf_custom.sh` pins 3.13.)

Unchanged from the feature branch: `wandb_privacy_settings`, `add_dev_scores_to_log_history`,
`training_info`, `check_model_dev_scores`, `DEV_SCORE_TOLERANCE`, `config_token_ids` (docstring
reworded), `processing_class=tokenizer` (comment reworded), the resume/best-checkpoint
`load_checkpoint_weights` calls, the end-of-training `trainer.save_state()`, `training_info.yaml`,
`dev.model.pred` check and W&B summary, the commented-out `logging.info(D)` lines.

### 3.3 `hfmt/hydra_config.py` (back to HEAD's version)

```diff
-from omegaconf import MISSING, OmegaConf
-from hfmt.prompt_formats import check_prompt_format
+from omegaconf import MISSING
 ...
-    prompt_format: str = MISSING
-    instruction: Optional[str] = MISSING
+    instruction: str = MISSING  # prepended to the source in the chat prompt
 ...
 def check_config(cfg) -> None:
-    if not OmegaConf.is_missing(cfg, "prompt_format"):
-        check_prompt_format(cfg)
```

Consequence: `run_config.yaml` / `inf_config.yaml` files written on the feature branch carry a
`prompt_format:` line the schema has no key for. On this branch `launch.sh inf|merge|sft` on such a
folder is refused by `grid.py` (`RETIRED_KEYS`, added 2026-11-10), which prints the `sed` command that
removes the line. One-liner for a tree of runs (review the list first):

```bash
grep -l '^prompt_format: chat$' egs/mmtc/*/outputs/*/run_config.yaml | xargs sed -i '/^prompt_format: chat$/d'
```

TranslateGemma run folders (`prompt_format: translategemma`, `instruction: null`) cannot be used on
this branch.

### 3.4 `hfmt/inf_translation.py` (back to HEAD's version)

```diff
-    tokenizer = load_tokenizer(cfg.checkpoint, cfg.prompt_format)
+    tokenizer = load_tokenizer(cfg.checkpoint)
```

### 3.5 `tools/inspect_model.py`

The `[prompt_format]` CLI argument, the `PROMPT_FORMATS` lookup, the `excluded_subtrees` filtering of
section 3 and the "not trained" lines of section 4 are removed. Section 7 now measures hfmt's actual
prompt (instruction + source, `CHAT_TEMPLATE_KWARGS`) rather than a bare one-word message, so a
Qwen3.5 count includes the empty think block:

```diff
-from hfmt.mt_model import NON_LANGUAGE_MODULES, resolve_excluded_modules  # noqa: E402
-from hfmt.prompt_formats import PROMPT_FORMATS  # noqa: E402
+from hfmt.mt_model import CHAT_TEMPLATE_KWARGS, format_input_prompt  # noqa: E402

-def main(checkpoint, prompt_format="chat"):
+def main(checkpoint):
 ...
-    excluded_subtrees, _ = resolve_excluded_modules(model, [])
-    groups = linear_groups(model, lm_head_name, excluded_subtrees)
+    groups = linear_groups(model, lm_head_name)
 ...
-    stub = types.SimpleNamespace(instruction="Translate French to English", data=types.SimpleNamespace(src="fr", trg="en"))
-    try:
-        messages = PROMPT_FORMATS[prompt_format].build(stub, word)
-        templated = tok.apply_chat_template(messages, ..., **PROMPT_FORMATS[prompt_format].template_kwargs)["input_ids"]
-    except Exception as e: ...
+    messages = format_input_prompt("Translate French to English", word)
+    templated = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
+                                        **CHAT_TEMPLATE_KWARGS)["input_ids"]
```

### 3.6 Tests and docs

- `hfmt/test_grid.py`: `test_prompt_format` (40 lines, including the three TranslateGemma experiment
  composes) removed; the two new tests `--requeue --open-mode=append` assertion and
  `test_grid_folder_resume_and_inf_jobs_are_named_after_the_folder` kept. 25 tests pass.
- `CLAUDE.md`: project tree no longer lists `prompt_formats.py`; `inspect_model.py` usage back to one
  argument; the experiment yaml sets `run_name`, `checkpoint`, `instruction`; the **Prompting** bullet
  is rewritten around `format_input_prompt` + `CHAT_TEMPLATE_KWARGS` + `processing_class=tokenizer`,
  keeping the TRL prefix-warning rule; the **Only the language model is trained** bullet is removed.
  Everything else from the feature branch (run_status, requeue, dev check, `training_info.yaml`,
  `bleu_13a_lowercase`, zh-en no longer stale) is kept verbatim.
- `README.md`: the **Prompt format** section becomes a three-line **Prompt** paragraph; the
  experiment bullet loses `prompt_format`. Run-folder contents, the pre-2026-11-05 full fine-tune
  warning, `tools/run_status.py` usage and the `sleep 3` inference loop are kept verbatim.
- `decisions.md`: the sections *TranslateGemma and the prompt-format key* and *TranslateGemma
  chat-template whitespace fix* are replaced by one section, *Prompt rendering: one builder,
  hardcoded `enable_thinking=False` (port of 2026-11-07)*, recording what was kept from feature 13
  (`processing_class`), what was dropped (registry, template patch, vision freeze) and the saved-config
  caveat. *GPU startup race and requeue* and *Full fine-tunes: the best checkpoint is reloaded through
  `from_pretrained`* are kept; two TranslateGemma mentions in the latter are reworded.

### 3.7 Yaml comment edits

- 28 experiment yamls: only the line `prompt_format: chat` removed.
- `egs/mmtc/{fr-en,zh-en}/sft_lora_gemma-3-4b-it_all-linear.yaml`: header rewritten. Kept: the
  motivation (Gemma 3 4B is TranslateGemma's base; the analysis numbers), memory, inspection and
  template-prefix facts, recipe and launch lines. Removed: `NON_LANGUAGE_MODULES`, `fix_template`,
  the `chat` format wording and the pointer to the deleted TranslateGemma sweep yaml. Added: one
  sentence that the all-linear adapter also lands on the vision tower's q/k/v_proj, which get no gradient.
- `egs/mmtc/{fr-en,zh-en}/sft_qwen-3.5-4b.yaml`: "as in sft_translategemma-4b-it.yaml" → "(the sweeps' 32 x 2)".
- `egs/mmtc/fr-en/sft_lora_qwen-3.5-9b_all-linear.yaml`: zero-shot sentence no longer compares to
  TranslateGemma; "The `chat` format adds" → "The prompt adds".
- `conf/peft/lora_gemma3_all_linear.yaml`: examples and the vision-tower sentence reworded for gemma-3-4b-it.
- `conf/train/full_finetune_short.yaml`: "(0.8B/2B/4B/TranslateGemma, lr" → "(Qwen3.5 0.8B/2B/4B, lr".

## 4. Left behind on `hydra3-translategemma`

| Path | Why |
|---|---|
| `hfmt/prompt_formats.py` | the registry; its only surviving effect is `CHAT_TEMPLATE_KWARGS` |
| `features/13_translategemma_prompt/{request,plan,todo}.md` | TranslateGemma feature |
| `egs/mmtc/fr-en/sft_translategemma-4b-it.yaml`, `sft_lora_translategemma-4b-it.sweep.yaml`, `sft_lora_translategemma-4b-it_all-linear.sweep.yaml`, `sft_lora_translategemma-4b-it_q-v-proj.sweep.yaml`, `sft_qlora_translategemma-4b-it.sweep.yaml` | TranslateGemma experiments |
| `egs/mmtc/zh-en/sft_translategemma-4b-it.yaml`, `sft_lora_translategemma-4b-it_all-linear.sweep.yaml`, `sft_lora_translategemma-4b-it_q-v-proj.sweep.yaml` | TranslateGemma experiments |
| `hfmt/test_grid.py::test_prompt_format` | tests the removed key |
| vision-tower freeze (`3743e17`, `9c66f6a`) | see §1 judgment call |

## 5. Verification

Done in this sandbox (no torch here, per the project's no-install rule):

- `python -m pytest hfmt/test_grid.py tools/test_run_status.py`: 44 passed (25 + 19).
- All 28 kept experiment yamls composed through `grid.main` (the `test_grid.run` helper, dummy
  `HFMT_DATA_ROOT`, temp `output_root`): 28 OK, 0 failed; no saved config carries `prompt_format`, every
  one has a non-empty string `instruction`. Job counts match the feature branch (e.g. 4 for the
  all-linear sweeps, 3 for the seeds sweep, 1 for the single runs).
- `py_compile` on every edited Python file; an AST scan found no unused imports.
- `grep` over the ported tree for `prompt_format|translategemma|NON_LANGUAGE|fix_chat_template|
  resolve_excluded|exclude_modules|make_prompt_formatter`: no hits in code, conf or experiment yamls
  except the intentional historical mentions of TranslateGemma as a model in the two Gemma 3 headers
  and in `decisions.md`.

Not verifiable here, to check on the cluster:

- A training job: the `chat_template_kwargs` column is now always written (the feature branch wrote
  it only when non-empty; it was always non-empty for `chat`, so behavior is identical in practice).
- `tools/inspect_model.py Qwen/Qwen3.5-4B` section 7 (needs transformers).
- `hfmt/test_mt_model.py` and `hfmt/test_sft_translation.py` (untracked, need torch) should still
  pass: the functions they test are unchanged.

## 6. Developer follow-ups

- **Files that import the removed module** and break `pytest` collection: `hfmt/test_prompt_formats.py`
  (tests the TranslateGemma template fix) was tracked and got committed with the port in `ef4ac04`; it was
  `git rm`'d on 2026-11-10. `test.py` (an untracked scratch script defaulting to a TranslateGemma run) is
  still in the working tree; delete it if it is no longer wanted. `benchmark_plan.md`, `config_reference.md`,
  `inference_overview.md` and `results.md` (also untracked) still describe `prompt_format`.
- **Saved configs** from the feature branch: remove the `prompt_format:` line before resuming,
  merging or decoding them (§3.3).
- **Suggested commits**, in the project's `<feature_dir>-<phase>: summary` form (`todo.md` phases
  map to the plan's execution order):

```
13_port_without_translategemma-A: port the TranslateGemma-free files from hydra3-translategemma

run_status/delete_checkpoints tools, Slurm requeue + GPU wait, conf/train schedules and
all-linear peft options, Qwen3.5 and Gemma 3 experiments for fr-en and zh-en (zh-en dataset
yaml replaces the stale sft1.sh/inf.sh), W&B privacy env vars, .gitignore.

13_port_without_translategemma-B: port model saving, dev check and BLEU-lc without prompt_format

Hardcode CHAT_TEMPLATE_KWARGS (enable_thinking=False) in mt_model and use it in training,
decoding, inference and inspect_model; keep processing_class=tokenizer and the full fine-tune
checkpoint reload; drop the prompt-format registry and the vision-tower freeze.

13_port_without_translategemma-C: docs for the port (CLAUDE.md, README, decisions, porting.md)
```
