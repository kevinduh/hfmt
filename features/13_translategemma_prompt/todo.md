# 13_translategemma_prompt — todo

Generated from `plan.md` (v3.1, final). Groups follow the plan's design sections and are ordered by
dependency; after every phase the Qwen experiments still train as before. Commit once per phase,
subject `13_translategemma_prompt-<phase>: <summary>`; Claude presents the message and the
developer commits. Each phase ends with its "Done when" checks. Items marked *(review)* came from
checking the plan against the code (2026-10-30), not from the plan itself.

Ground rules for every phase:
- **No dataset content to W&B** (CLAUDE.md). This feature adds no W&B calls; the new log lines
  (example built prompt, "Not trained" list) stay local and hold no data.
- **Change as little as possible.** Keep TRL/PEFT defaults.
- **Don't touch the untracked leftovers or the developer's runs** (`code_review.md`, `sync.sh`,
  `analysis/wandb_export.json`, `tools/delete_checkpoints.sh`, the modified
  `local_setting.sh.example`). Stage files by name.
- **Tests:** `HFMT_ROOT=$(pwd) python hfmt/test_grid.py`.
- **Login-node checks** use placeholder sentences, never data; scratch scripts live in the
  scratchpad, not the repo. `DRYRUN=1` launches use `output_root=<scratchpad dir>`.
- **GPU runs** go to `egs/mmtc/fr-en/outputs/p13<phase>…` (`run_name=p13…`), smoke settings
  `train.max_steps=20 train.eval_steps=10 wandb.enabled=false`, and are deleted once confirmed.

## 0. Before any code changes

- [ ] Resume or finish any cancelled run that may still need it. After A, resuming or merging a run
      saved earlier needs `prompt_format: chat` added to its `run_config.yaml` by hand; `inf` takes
      `prompt_format=chat` on the command line (plan Risk 3).
- [ ] *(review)* Reference for the Qwen regression (Risk 1), at `5402fc5`: `p13ref_qwen`, a smoke run
      of `sft_lora_qwen-3.5-0.8b.sweep.yaml` with both axes pinned (`train.learning_rate=1e-4
      peft.lora.r=8`). Keep its "Inspecting batch before training" dump, `adapter_config.json` and
      `test1.pred.trg`. (Reuse an existing Qwen run/log from `run_logs/` at `5402fc5` if one is
      already around, instead of launching this.)

## A. Prompt formats + config, launch side (plan §1, §2, §6)

### A1. `hfmt/prompt_formats.py` (new; imports nothing heavy)
- [x] Module docstring: what a prompt format is, how to add one (one entry), why the key is
      explicit (see decisions.md).
- [x] `@dataclass PromptFormat`: `build(cfg, text) -> messages`, `check(cfg) -> None` (config only;
      `ValueError` that says the fix).
- [x] `chat`: build `[{"role": "user", "content": f"{cfg.instruction}\n{text}"}]` (the same string as
      `format_input_prompt` today); check requires `instruction` not null.
- [x] `translategemma`: build `[{"role": "user", "content": [{"type": "text", "source_lang_code":
      cfg.data.src, "target_lang_code": cfg.data.trg, "text": text}]}]`; check requires `instruction:
      null` ("TranslateGemma's template writes the instruction from data.src/trg") and
      `data.src`/`data.trg` set.
- [x] `PROMPT_FORMATS = {"chat": ..., "translategemma": ...}`.
- [x] `check_prompt_format(cfg)`: unknown name → `ValueError` listing the known names; otherwise run
      the format's `check`.

### A2. Schema (`hfmt/hydra_config.py`)
- [x] `prompt_format: str = MISSING` next to `checkpoint`, commented (a `PROMPT_FORMATS` key, set in
      the experiment yaml next to `checkpoint`).
- [x] `instruction: Optional[str] = MISSING` (null allowed, must still be set); comment: required by
      `chat`, `null` for `translategemma`.
- [x] `check_config` calls `check_prompt_format(cfg)`; its docstring says so.
- [x] No default in `conf/config.yaml`.

### A3. Experiments (`egs/mmtc/fr-en/`)
- [x] `prompt_format: chat` next to `checkpoint` in `sft_lora.yaml`, `sft_lora.sweep.yaml` and the
      6 Qwen3.5 files.
- [x] `prompt_format: translategemma` + `instruction: null` (comment: the template builds the
      instruction from `data.src`/`data.trg`) in the 3 TranslateGemma files.
- [x] `train: {train_batch_size: 16, gradient_accumulation_steps: 4}` in
      `sft_lora_translategemma-4b-it.sweep.yaml` and `sft_qlora_translategemma-4b-it.sweep.yaml`,
      commented (effective batch 64; fits the 40 GB A100s; a job that still OOMs is resumed on a
      bigger GPU) (Q4).
- [ ] *(review, optional — unrelated cleanup)* The copied header comments of
      `sft_lora_qwen-3.5-2b.sweep.yaml`, `sft_qlora_qwen-3.5-2b.sweep.yaml` and `sft_qwen-3.5-2b.yaml`
      still say 0.8B (title, launch line, output folder). Fix in a separate commit, not inside a
      feature-13 phase.
- [ ] *(review, developer decides)* The 9 new experiment yamls and `conf/train/midrun.yaml` are
      untracked, and A4's test composes the TranslateGemma ones. Stage them with A (or commit them
      first) so the test passes on a clean checkout.

### A4. Tests (`hfmt/test_grid.py`)
- [x] New test, on `SINGLE` with overrides:
  - `prompt_format=nope` fails before writing anything; the message lists `chat` and `translategemma`;
  - `prompt_format=translategemma` fails with the instruction still set, passes with `instruction=null`;
  - `prompt_format=chat instruction=null` fails;
  - *(review, optional)* `prompt_format=translategemma instruction=null` fails on a dataset yaml
    without `src`/`trg` (the absolute-path dataset of
    `test_dataset_with_absolute_paths_needs_no_data_root`) — the three cases above already cover the
    behavior; add only if cheap;
  - the 3 TranslateGemma experiments compose: saved `prompt_format: translategemma`,
    `instruction` null; the two sweeps have batch 16 × 4.
- [x] *(review)* An experiment without `prompt_format` fails (like `test_missing_value_fails`): there
      is deliberately no default.
- [x] Module docstring: add feature 13.

### Done when
- [x] Tests pass (25 passed, incl. `test_prompt_format`).
- [x] `grid.py` on each TranslateGemma experiment writes its jobs and prints the sbatch args (ran the
      step `launch.sh sft` calls, with `output_root` in the scratchpad; `launch.sh` itself sources the
      cluster-only `install/path.sh`); `instruction=...` on a TranslateGemma experiment fails before
      writing anything.
- [x] `todo.md` A ticked; commit message presented.

## B. Prompts at job start: training and decoding (plan §1 `make_prompt_formatter`, §3, §4)

### B1. `hfmt/prompt_formats.py`
- [x] `make_prompt_formatter(cfg)` (no tokenizer, no heavy imports): look up `cfg.prompt_format` in
      the registry and return `format_prompt(text) -> messages` bound to `cfg`; log one
      placeholder-built example (local log; no data). No chat-template inspection — the explicit key
      is the source of truth; a format that doesn't fit the checkpoint surfaces when the template is
      applied (one the template rejects crashes there; one that renders anyway is the user's choice).

### B2. `hfmt/mt_model.py`
- [x] Remove `format_input_prompt`.
- [x] `inference_on_eval_data(..., format_prompt, ...)` in place of `instruction_prefix`, for both the
      batch prompts and the cut-off length check.
- [x] `decode_test_sets` builds `make_prompt_formatter(cfg)` and passes it.
      (`inf_translation.py` needs no change.)
- [x] Module docstring: prompts come from `prompt_formats.py`.

### B3. `hfmt/sft_translation.py`
- [x] `format_prompt = make_prompt_formatter(cfg)` right after `load_tokenizer`; drop
      `instruction_prefix` and its log line.
- [x] `preprocess_fn`: `prompt = [format_prompt(s) for s in samples["src"]]`; completion unchanged
      (assistant string).
- [x] `SFTTrainer(..., processing_class=tokenizer)`, commented: keeps TRL off `AutoProcessor`
      (TranslateGemma and Qwen3.5 ship VL processors); one tokenizer for training and decoding.
- [x] `make_mt_callback(..., format_prompt, ...)` in place of `instruction_prefix`.
- [x] Imports: drop `format_input_prompt`, add `make_prompt_formatter`.

### Done when
- [x] Tests pass (25); `grep -rn "format_input_prompt\|instruction_prefix" hfmt/mt_model.py
      hfmt/sft_translation.py tools/` is empty; modules import with no torch (heavy imports stay in
      functions); `make_prompt_formatter` build logic verified (chat + translategemma message
      structures) with a small script.
- [ ] *(cluster; not runnable in this sandbox — no hfmt env, TranslateGemma is gated)* Login node,
      scratch script (plan Validation 1), TranslateGemma tokenizer: applying its template to the
      translategemma prompt and prompt+completion renders; prompt tokens are a prefix of
      prompt+completion; exactly one BOS; the completion ends with `<end_of_turn>`. (A `chat` prompt or
      an unsupported code raises — ordinary transformers behavior, no longer pre-checked.)
      *2026-11-04: run after the fact; the prefix check FAILED (template whitespace leaked into the
      assistant turn). Fixed by `fix_template` in `hfmt/prompt_formats.py`; see `decisions.md`
      "TranslateGemma chat-template whitespace fix". Re-tick after `test.py` passes with the fix.*
- [ ] *(cluster)* Same script, Qwen2.5-1.5B-Instruct and Qwen3.5-0.8B tokenizers: `chat` renders the
      same string as `format_input_prompt` did. (This is B's Qwen gate; the GPU batch-dump regression
      runs once at the end of C, since C is a no-op for Qwen.)
- [x] `todo.md` B ticked; commit message presented.

## C. Only the language model is trained (plan §3 `NON_LANGUAGE_MODULES`, §4 Q3)

### C1. `hfmt/mt_model.py`
- [x] `NON_LANGUAGE_MODULES = ["vision_tower", "multi_modal_projector"]`, commented: applied to
      every model whatever its `prompt_format`; a multimodal model with other names (`visual`,
      `audio_tower`, …) → add them here (Risk 8).
- [x] `resolve_excluded_modules(model, target_modules) -> (subtrees, leaf_full_names)`:
  - `subtrees`: the outermost modules whose last name component is in `NON_LANGUAGE_MODULES` (any
    depth);
  - `leaf_full_names`: full names of modules inside those subtrees whose last name component is in
    `target_modules`;
  - nothing matched → `({}, [])`; no heavy imports (also works on a meta-device model).

### C2. `hfmt/sft_translation.py`, after `load_base_model`
- [x] Log "Not trained (outside the language model): model.vision_tower <n> params,
      model.multi_modal_projector <n> params" ("none" when nothing matches).
- [x] `full`: `requires_grad_(False)` on each subtree; the log line becomes "trainable X of Y".
- [x] `lora`/`qlora`: `LoraConfig(exclude_modules=leaf_full_names or None)`. *(review)* `None` when
      empty, so a Qwen `adapter_config.json` keeps `exclude_modules: null` (not `[]`).
- [x] Comment: saved weights keep the unchanged vision tower; `PeftModel.from_pretrained` rebuilds
      the wrapping from `adapter_config.json`, so `load_trained_model`/`merge_trained_model` don't
      change and older adapters still load.

### Done when
- [x] Tests pass (25); modules import with no torch; `resolve_excluded_modules` verified on a
      fake-model structure (translategemma-like → 2 subtrees + the vision q/v_proj leaves, LM leaves
      kept; full-FT `[]` targets → subtrees only; qwen-like → `({}, [])`).
- [ ] *(cluster; needs transformers/meta-device)* `resolve_excluded_modules` on TranslateGemma built
      on the meta device (as `inspect_model.py` builds it) with `[q_proj, v_proj]` → the 2 subtrees
      and the SigLIP leaf names per target; on Qwen3.5-0.8B → `({}, [])`.
- [ ] *(cluster)* GPU smoke runs (plan Validation 2): `p13c_lora` and `p13c_qlora` (one point of each sweep, its
      16 × 4 batch, 40 GB A100), `p13c_full` (the full-FT experiment, 80 GB):
  - no "Mismatch between tokenized prompt" warning;
  - the batch dump shows the TranslateGemma prompt, with the prompt labels at -100;
  - the "Not trained" line lists `model.vision_tower` and `model.multi_modal_projector`;
  - lora/qlora: "LoRA wraps: q_proj x34, v_proj x34"; full: trainable ≈ 3.88B, and the Trainer's
    count agrees;
  - `generation_config` eos includes 106, and dev predictions stop;
  - peak allocated memory logged; note it next to `inspect_model.py`'s estimate.
- [ ] *(cluster)* Qwen regression, `p13c_qwen` (same config as `p13ref_qwen`): the batch dump (decoded text,
      `input_ids`, `labels`) and `test1.pred.trg` match the reference (Risk 1, covers B's
      `processing_class` change); "Not trained: none"; LoRA wraps unchanged; `adapter_config.json`
      identical to the reference's (`exclude_modules: null`).
- [x] `todo.md` C ticked; commit message presented.

## D. `tools/inspect_model.py` (plan §5)

- [x] *(review)* Make `hfmt` importable: `python tools/inspect_model.py` puts `tools/`, not the repo
      root, on `sys.path`, and `install/path.sh` sets no `PYTHONPATH`. Insert the repo root (as
      `test_grid.py` does) before the `hfmt` imports.
- [x] Section 3: leave modules inside `resolve_excluded_modules`' subtrees out of `linear_groups`, so
      the candidate table and the paste-ready `target_modules` lines show language-model modules
      only. (This also corrects section 5's lora/qlora adapter counts; its full row still counts
      the vision params, inside ±30%, left for later.)
- [x] Section 4: a line listing the excluded subtrees with their param counts ("not trained; see
      `NON_LANGUAGE_MODULES`"), no `get_decoder` walk.
- [x] Section 7: build the message with an explicit `prompt_format` (CLI flag, default `chat`; pass
      `translategemma` for a TranslateGemma-style checkpoint) and a stand-in cfg (`data.src=fr`,
      `data.trg=en`, `instruction="Translate French to English"` for `chat`). Print the chosen
      `prompt_format` and the real per-example overhead (saying it includes the instruction) instead
      of crashing (a format that doesn't render prints a one-line hint). No auto-detection.
- [x] Module docstring: mention the excluded modules and the suggested `prompt_format`.

### Done when (plan Validation 1)
- [x] `python3 -m py_compile` passes; the `hfmt` imports resolve and section 7's stub-cfg `build`
      produces the right messages for `chat` and `translategemma` (checked without torch). The runs
      below need the hfmt env + weights, so they're cluster.
- [ ] *(cluster)* `inspect_model.py google/translategemma-4b-it translategemma`: section 7 renders
      (no crash); LoRA candidates show `q_proj` 34 (vision dropped); the vision tower and projector
      are listed as not trained.
- [ ] *(cluster)* `inspect_model.py Qwen/Qwen3.5-0.8B` (default `chat`): section 7 renders; nothing
      listed as excluded; the LoRA-candidate counts are unchanged; the overhead now includes the
      instruction.
- [ ] *(cluster)* `inspect_model.py google/gemma-3-4b-it` (default `chat`): section 7 renders; the
      vision tower and projector are still listed as not trained (exclusion is independent of the
      prompt format).
- [x] `todo.md` D ticked; commit message presented.

## E. Inference and merge on GPU (plan Validation 3–4)

No code expected here; a fix is committed under the phase whose code it changes.
- [ ] `launch.sh inf <p13c_lora>/run_config.yaml` reproduces its `test1.pred.trg` (`cmp`).
- [ ] Zero-shot: `launch.sh inf egs/mmtc/fr-en/sft_translategemma-4b-it.yaml run_name=p13e_zeroshot
      slurm.constraint=null` gives sensible BLEU (our prompt matches the model card's).
- [ ] `launch.sh merge <p13c_lora>/run_config.yaml`; the vision tower and projector weights in
      `model_merged/` equal the original checkpoint's.
- [ ] Decode the merged model: `launch.sh inf egs/mmtc/fr-en/sft_translategemma-4b-it.yaml
      checkpoint=<p13c_lora>/model_merged peft=lora_default run_name=p13e_merged
      slurm.constraint=null` scores like `p13c_lora`.
- [ ] Delete the `p13*` runs (and `p13ref_qwen`).
- [ ] `todo.md` E ticked.

## F. Docs (plan §7)

- [x] `CLAUDE.md`:
  - project structure: a `prompt_formats.py` line (and `[prompt_format]` on the inspect_model line);
  - Configuration bullet: an experiment sets `run_name`, `checkpoint`, `prompt_format`, `instruction`;
  - Prompting bullet: causal-LM prompts come from `hfmt/prompt_formats.py`; adding a format = one
    entry;
  - only the language model is trained (`NON_LANGUAGE_MODULES` in `mt_model.py`); when adding a
    multimodal model, look for a vision/audio tower (`inspect_model.py` model print) and extend the list.
- [x] README "Running SFT experiments": the `prompt_format` key in the experiment description; a
      short TranslateGemma note (`prompt_format: translategemma`, `instruction: null`, language codes
      from `data.src`/`data.trg`); old run configs need `prompt_format=chat` for `inf`, or the line
      added by hand for `merge`/resume.
- [x] `decisions.md`, feature 13 entry:
  - explicit `prompt_format` key, not derived from the template (a model's format may be documented
    only in its card; the key shows in `run_config.yaml`/W&B and is checked at launch); notes the
    dropped job-start template check;
  - `processing_class=tokenizer`;
  - only the language model is trained, via one list of module names applied to every model (not
    per prompt format: Gemma 3 4B uses `chat` but has a vision tower); plus the Q4 sweep batch note.
- [x] Re-read the `hydra_config.py` comments for `prompt_format` and `instruction` (A2) — accurate,
      no change needed.
- [x] Tests pass (25). `todo.md` F ticked; commit message presented.

## G. Developer validation

- [ ] Launch the two TranslateGemma sweeps and the full fine-tune; resume any job that OOMs on a
      bigger GPU (`launch.sh sft <run folder>/run_config.yaml slurm.gres=…`).
