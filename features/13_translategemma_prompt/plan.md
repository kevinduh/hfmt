# 13_translategemma_prompt — plan (v3.1, simplified)

Status: **v3.1.** All questions decided.
- Q1: the prompt format is an **explicit key**, `prompt_format`, set in the experiment yaml next to
  `checkpoint`. A small registry of formats keeps adding the next model to one entry.
- Q2: TranslateGemma experiments set `instruction: null`; a string is an error.
- Q3: **only the language model is trained.** We never train on images, and logs +
  `tools/inspect_model.py` must not count vision parameters as trainable.
- Q4: the two TranslateGemma sweeps use `train_batch_size: 16`, `gradient_accumulation_steps: 4`
  (effective batch 64), which fits the 40 GB A100 pool. A job that still OOMs is resumed on a bigger
  GPU (feature 12).

Since v2: the registry is a **dict of plain records** (not a class hierarchy); the job-start template
check is **two checks** (not a bidirectional marker cross-check); the modules outside the language
model are **named in one shared list** (`NON_LANGUAGE_MODULES` in `mt_model.py`), applied to every
model and not discovered via `get_decoder()` (dropping the old Risk 6) — architecture, not the prompt
format, decides them (Gemma 3 4B/12B/27B use `chat` but have TranslateGemma's vision tower); and
`inspect_model.py` only corrects the reported counts, leaving the estimate's vision over-count for
later (inside its stated ±30%).

## Request (summary)

Change `format_input_prompt` and the other places that build prompts so `google/translategemma-4b-it`
can be fine-tuned with `full`, `lora` and `qlora`, then decoded (dev, test, `launch.sh inf`) and
merged like any other model.
- **Out of scope:** image translation (`"type": "image"`).
- The model card uses `AutoProcessor` + `AutoModelForImageTextToText`; we keep `AutoTokenizer` +
  `AutoModelForCausalLM`. For text-only use they give the same class and the same template. The
  processor is only needed for images.

## Starting point (`hydra3` @ 5402fc5)

**Prompts today.** One helper, used by training and every decode:
`mt_model.py:14` `format_input_prompt(instruction, text)` → `[{"role": "user", "content":
instruction + "\n" + text}]`. Training builds `prompt` with it (`sft_translation.py:146`) and
`completion` as `[{"role": "assistant", "content": target}]`. Decoding calls it in
`inference_on_eval_data` (`mt_model.py:110`), reached by the dev callback and `decode_test_sets`.

**Two blockers for TranslateGemma:**

1. **The chat template** (`chat_template.jinja`, pasted 2026-10-27) needs the **user turn** as a list
   with exactly one mapping `{"type": "text", "source_lang_code": ..., "target_lang_code": ...,
   "text": ...}` (a string raises); the **assistant turn** as a **string** (ours already is). The
   instruction is built into the template from the two codes ("You are a professional French (fr) to
   English (en) translator…"), so `cfg.instruction` has nowhere to go. Codes are BCP-47 keys of the
   template's `languages` dict and match our `data.src`/`data.trg` (`fr`/`en`); an unknown code
   raises a Jinja `UndefinedError`. The template emits its own `bos_token` and ends each turn with
   `<end_of_turn>\n`.
2. **TRL 1.3.0 loads a processor.** With no `processing_class`, `SFTTrainer`
   (`sft_translation.py:250`) calls `AutoProcessor.from_pretrained` → a `Gemma3Processor`, sets
   `_is_vlm = True`, and `prepare_multimodal_messages` rewrites the assistant string into
   `[{"type": "text", ...}]`, which the template rejects. **So training fails even with the user turn
   fixed, unless we pass the tokenizer.**

**The vision tower (Q3).** `AutoModelForCausalLM` maps `gemma3` to `Gemma3ForConditionalGeneration`,
which holds `model.vision_tower` (SigLIP, 27 layers) and `model.multi_modal_projector` (~0.42B of
4.30B params). With no `pixel_values` the vision tower never runs, so **it gets no gradients and
never changes regardless of what we do** — Q3 is about correct *reporting*, not making training work.
Today it misleads: in `full` the vision params are reported trainable; in `lora`/`qlora`
`[q_proj, v_proj]` also matches the 27 SigLIP attention layers (61 wrapped modules, not 34), bloating
the adapter and the reported count. The multimodal Gemma 3 sizes (4B/12B/27B, also
`Gemma3ForConditionalGeneration`) have the same issue under the plain `chat` format; the text-only
sizes (1B/270M, `gemma3_text` → `Gemma3ForCausalLM`) don't. Qwen2.5/Qwen3.5 have no vision tower in
our runs (the text-only class drops `model.visual.*` on load), so nothing is excluded for them.

**Checked, no change needed:** no double BOS (`apply_chat_template(tokenize=True)` uses
`add_special_tokens=False`); `token_type_ids` only used when given; the completion ends with
`<end_of_turn>` so the model learns to stop (generation stops there if `eos_token_id` includes id
106 — verify, see Validation); TRL masks the prompt prefix and warns on a mismatch; merge /
second-round / inference reload the saved tokenizer+template and re-set `prompt_format`.
`inspect_model.py` section 7 currently crashes on the plain-string user message.

## Design

### 1. Prompt formats (`hfmt/prompt_formats.py`, new)

A dict registry of plain records — **no class hierarchy**. The module imports nothing heavy, so
`check_config` can use it at launch.

```
@dataclass
class PromptFormat:
    build:   Callable        # (cfg, text) -> the user-turn messages
    check:   Callable        # (cfg) -> None; config-only, raises ValueError with the fix
    marker:  str | None      # must appear in the checkpoint's chat_template, or None

PROMPT_FORMATS = {"chat": PromptFormat(...), "translategemma": PromptFormat(...)}
```

- **`chat`** (every model so far): `build` → `[{"role": "user", "content": f"{cfg.instruction}\n{text}"}]`;
  `check` requires `instruction` set (not null); `marker=None`.
- **`translategemma`**: `build` → `[{"role": "user", "content": [{"type": "text", "source_lang_code":
  cfg.data.src, "target_lang_code": cfg.data.trg, "text": text}]}]`; `check` requires
  `instruction: null` ("TranslateGemma's template writes the instruction from data.src/trg") and
  `data.src`/`data.trg` set; `marker="source_lang_code"`.
- `check_prompt_format(cfg)`: unknown name → raise listing the known names; else run the format's
  `check`. Called from `check_config`, so it runs at launch (`grid.py:84`) and again at job start
  (`sft_translation.py:101`, `inf_translation.py:31`).
- `make_prompt_formatter(cfg, tokenizer)` (job start; takes the tokenizer as an argument, imports
  nothing heavy). Returns `format_prompt(text) -> messages` after **two checks**:
  1. if the format has a `marker`, it must be in `tokenizer.chat_template`, else raise naming the
     format to use. This is the only *silent* failure: `translategemma`'s list content renders as a
     bare repr on a plain-chat template instead of raising.
  2. render a placeholder prompt (`build(cfg, "x")`) through the template; any exception re-raises
     with a clear message. This fails within seconds and catches `chat` on a TranslateGemma
     checkpoint (loud Jinja error) and an unsupported language code. Log the rendered placeholder.

Adding a model with special requirements later = one more entry.

### 2. Schema and experiments

- `hydra_config.py`: `prompt_format: str = MISSING` next to `checkpoint`; `instruction:
  Optional[str] = MISSING` (null allowed, still required to be set). `check_config` calls
  `check_prompt_format`.
- **No default in `conf/config.yaml`** — each experiment sets `prompt_format` next to `checkpoint`.
- The 11 fr-en experiments:
  - `prompt_format: chat`: `sft_lora.yaml`, `sft_lora.sweep.yaml`, and the 6 Qwen3.5 files.
  - `prompt_format: translategemma` + `instruction: null` (with a comment): the 3 TranslateGemma
    files. The two sweeps also get `train_batch_size: 16`, `gradient_accumulation_steps: 4` (Q4).
- **Old `run_config.yaml`s lack the key** (decisions.md:411): `launch.sh inf` → pass
  `prompt_format=chat`; `launch.sh merge`/resume take only `slurm.*`, so add `prompt_format: chat`
  to the saved config by hand first (Risk 3).

### 3. `mt_model.py`

- Remove `format_input_prompt`. `inference_on_eval_data` takes `format_prompt` in place of
  `instruction_prefix`; `decode_test_sets` builds it with `make_prompt_formatter(cfg, tokenizer)`.
  `inf_translation.py` needs no change.
- New constant `NON_LANGUAGE_MODULES = ["vision_tower", "multi_modal_projector"]`: module names that
  are never trained, applied to every model whatever its `prompt_format`. A model that has none of
  them (Qwen, text-only Gemma 3) is unaffected. A future model with differently named ones
  (`visual`, `audio_tower`, …) means one more name here (Risk 8).
- New helper `resolve_excluded_modules(model, target_modules)` → `(subtrees, leaf_full_names)`:
  `subtrees` = the outermost modules whose last name component is in `NON_LANGUAGE_MODULES` (e.g.
  `model.vision_tower`; matched by name, so the nesting depth doesn't matter), for freezing/counting;
  `leaf_full_names` = the full names of modules under those subtrees whose last name component is in
  `target_modules` (for PEFT's `exclude_modules`). Explicit names, so no `get_decoder()` best-effort.
  No match → `([], [])`. No heavy module-level imports, so `inspect_model.py` can use it.

### 4. Training (`sft_translation.py`)

- **Prompts:** build `format_prompt = make_prompt_formatter(cfg, tokenizer)` right after
  `load_tokenizer` (fail-fast; replaces the `instruction: '...'` log line). `preprocess_fn`:
  `prompt = [format_prompt(s) for s in samples["src"]]`. `SFTTrainer(..., processing_class=tokenizer)`
  keeps TRL off the processor path and uses one tokenizer object for training and decoding. The MT
  callback takes `format_prompt` in place of `instruction_prefix`.
- **Language model only (Q3),** after `load_base_model`, driven by `NON_LANGUAGE_MODULES`:
  - log the excluded subtrees with param counts ("Not trained (outside the language model):
    model.vision_tower …, model.multi_modal_projector …").
  - `full`: `requires_grad_(False)` on each subtree; the log line becomes "trainable X of Y" and the
    Trainer's count agrees.
  - `lora`/`qlora`: `LoraConfig(exclude_modules=<leaf_full_names>)`. Exact full names, so PEFT's
    matching rule doesn't matter. Empty for Qwen → `adapter_config.json` unchanged. TranslateGemma's
    "LoRA wraps" line then shows `q_proj x34, v_proj x34`.
  - The saved weights still include the (unchanged) vision tower. Inference and merge need no change:
    `PeftModel.from_pretrained` rebuilds the wrapping from `adapter_config.json` (including
    `exclude_modules`), and adapters saved before this change still load.

### 5. `tools/inspect_model.py` (counts only)

Imports the registry + `resolve_excluded_modules` from `hfmt/` (the exclusion needs no cfg, so it
matches training exactly). For the prompt it has no cfg, so it picks the
format whose `marker` is in the template (else `chat`) — section 7 already needs this.
- **Sections 3–4:** drop the excluded subtrees from the LoRA-candidate table and the paste-ready
  `target_modules` lines, and add a line listing the excluded subtrees with their param counts (via
  `resolve_excluded_modules`; no `get_decoder` walk). A new multimodal model whose modules aren't in
  `NON_LANGUAGE_MODULES` simply won't appear on that line — the section-1 model dump and section 3's
  parent column (e.g. a `…VisionAttention` parent on `q_proj`) make the omission visible.
- **Section 5 (memory estimate):** unchanged for now. It still counts the vision params in the
  full-FT estimate (~5 GiB high for TranslateGemma), which is inside its stated ±30% band; revisit
  if it matters.
- **Section 7:** build the message via the chosen format's `build` (placeholder codes `fr`→`en`),
  print the `prompt_format` to use, and report the real per-example overhead instead of crashing.

### 6. Tests (`hfmt/test_grid.py`)

One new test, on `sft_lora.yaml` with overrides:
- `prompt_format=nope` fails before writing anything (message lists the known names);
- `prompt_format=translategemma` fails with the instruction still set, and passes with
  `instruction=null`;
- `prompt_format=chat instruction=null` fails;
- the 3 TranslateGemma experiments compose.

### 7. Docs

- `CLAUDE.md`: an experiment sets `run_name`, `checkpoint`, `prompt_format`, `instruction`; the
  prompting bullet points to `hfmt/prompt_formats.py` and how to add a format; only the language
  model is trained (`NON_LANGUAGE_MODULES` in `mt_model.py`; check for a vision/audio tower when
  adding a multimodal model and extend the list).
- README: the `prompt_format` key and a short TranslateGemma note (`instruction: null`, codes from
  `data.src/trg`).
- `decisions.md`: (a) explicit `prompt_format` key over deriving it from the template (a model's
  format may live only in its card; the key is visible in `run_config.yaml`/W&B and checked at
  launch); (b) `processing_class=tokenizer`; (c) only the language model is trained, via one list
  of module names applied to every model (not per prompt format: Gemma 3 4B uses `chat` but has a
  vision tower).
- `hydra_config.py` comments for `prompt_format` and `instruction`.

Expected size: `prompt_formats.py` ~55 lines; ~25 changed in `mt_model.py`, ~20 in
`sft_translation.py`, 3 in `hydra_config.py`, ~20 in `inspect_model.py`, ~20 in `test_grid.py`; plus
11 yamls and docs.

## Risks

1. **Qwen3.5 moves onto TRL's tokenizer path.** Its checkpoint ships a VL processor (we load the
   text-only model), so passing the tokenizer changes how its data is tokenized. Qwen templates
   render string and `[{"type": "text"}]` content the same way, so tokens should be identical —
   verify with the "Inspecting batch before training" dump (Validation 5). Qwen2.5 is unaffected
   (`AutoProcessor` already returns its tokenizer).
2. **What fails where.** At launch: unknown/missing `prompt_format`, an instruction that doesn't fit
   the format, missing `data.src/trg`. At job start (needs the tokenizer): a format that doesn't
   match the checkpoint's template, an unsupported language code.
3. **Old run configs.** `inf`/`merge`/resume of runs saved before this change need `prompt_format:
   chat` (Design 2). Resume cancelled runs before this lands, or add the line by hand.
4. **The template check is a safety net, not a proof** (a string match + a placeholder render). A
   future model whose template has no registered marker passes under `chat` and would get the plain
   chat prompt. Reading its template with `inspect_model.py` (section 7) when adding it is the real
   check.
5. **fp16 overflow.** Gemma 3 overflows in fp16; `gres=gpu:a100` always gives bf16. Don't point
   TranslateGemma runs at V100s.
6. **One shared tokenizer.** The callback flips `padding_side="left"` while decoding and restores it;
   TRL's collator pads right by itself and the datasets are tokenized at init, so this is safe.
7. **Left truncation at decode** (`decode.max_length` 1024) can cut an over-long input's BOS/header.
   Existing behavior, unchanged; the cut-prompt count is already logged.
8. **`NON_LANGUAGE_MODULES` is maintained by hand.** A future multimodal model whose vision/audio
   modules have other names is trained and reported as today (inert adapters, inflated counts) until
   its names are added. The section-1 model dump in `inspect_model.py` and the "Not trained" log line
   in training (what was actually excluded) make such an omission visible.

## Validation (sketch)

1. **Login node, no GPU** (scratch script; placeholder sentences, not data):
   - TranslateGemma tokenizer: the formatter's prompt and prompt+completion render; prompt tokens are
     a prefix of prompt+completion; exactly one BOS and the completion ends with `<end_of_turn>`;
     code `xx` fails clearly; `chat` on it fails naming `translategemma`.
   - A Qwen checkpoint: `chat` renders the same string as before; `translategemma` on it fails.
   - `HFMT_ROOT=$(pwd) python hfmt/test_grid.py` passes.
   - `inspect_model.py google/translategemma-4b-it`: finishes section 7 naming `translategemma`,
     LoRA candidates show `q_proj` 34, lists the vision tower + projector as not trained.
     `inspect_model.py Qwen/Qwen3.5-0.8B`: nothing excluded. `inspect_model.py google/gemma-3-4b-it`:
     section 7 names `chat`, and the vision tower + projector are still listed as not trained
     (exclusion independent of the format).
2. **GPU smoke runs** (`train.max_steps=20 train.eval_steps=10`) for one lora point, one qlora point
   and the full FT (80 GB): no "Mismatch between tokenized prompt" warning; the batch dump shows the
   TranslateGemma prompt with prompt labels at -100; the "Not trained" line lists the vision modules;
   trainable counts (lora/qlora wrap `x34`, full ≈ 3.88B); `generation_config` eos includes 106 and
   dev predictions stop; peak allocated memory logged.
3. **Inference:** `launch.sh inf` on a smoke run reproduces its `test1.pred.trg`; a zero-shot `inf`
   of the experiment yaml gives sensible BLEU (checks our prompt matches the card).
4. **Merge:** `launch.sh merge` on the lora smoke run; merged vision weights equal the original; then
   `inf` from `model_merged/` with `prompt_format: translategemma`.
5. **Qwen regression:** one short Qwen3.5 run; its batch dump matches a pre-change run (Risk 1), and
   its `adapter_config.json` has no `exclude_modules`.
