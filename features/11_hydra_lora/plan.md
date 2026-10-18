# 11_hydra_lora — plan (v4, final)

Status: **v4, final.** Q1–Q9 are decided (developer answers, 2026-10-18). v3 brought back the
LoRA-candidates section and added a training-memory estimate; v4 fixes the estimate's equations
(no calibration) and prints every term with its numbers. `todo.md` is generated from this version.

## Request (summary)

1. **`target_modules` is a list in the config.** It replaces the `lora_target` string that
   `sft_translation.py` maps to module names. The `all-linear` option goes; the user always lists
   the modules, e.g. in `lora_default.yaml`:
   ```yaml
   lora:
     target_modules:
       - q_proj
       - v_proj
   ```
   (The request's sketch writes `- target_modules`, i.e. a list item. Read as the mapping above.)
2. **Shorter LoRA key names:** `peft.lora.lora_r` → `peft.lora.r`, and likewise for the other
   LoRA fields. No backward compatibility.
3. **`tools/inspect_model.py <checkpoint>`** prints:
   - the whole model;
   - module types with their counts, and the total parameter count;
   - other useful statistics (below).

   Its purpose is to show what a checkpoint's modules are called.

## Starting point (`hydra3` @ 5d1b475)

- **Schema:** `LoraConfig` has `lora_r: int`, `lora_alpha: float`, `lora_dropout: float` and
  `lora_target: str` (`qv | all-linear | attention | mlp`).
- **Defaults:** `conf/peft/{lora,qlora}_default.yaml` set `lora_target: qv`.
- **The mapping** (`sft_translation.py:173-188`) turns the string into module names.
  - `attention` maps to `out_proj`, but Qwen's attention output layer is `o_proj`. So `attention`
    silently trains q/k/v only on Qwen, an example of the brittleness the request describes.
  - `mlp` leaves out `gate_proj`.
- **Who reads the LoRA keys:** only training (`LoraConfig(...)` in `sft_translation.py`).
  Inference and merging read the adapter's own `adapter_config.json`, so they don't change.
- **Other references to rename:**
  - `peft.lora.lora_r` in `egs/mmtc/fr-en/sft_lora.sweep.yaml` (a grid axis);
  - `hfmt/test_grid.py` (11 places);
  - README (2 commands), CLAUDE.md (1), and the `launch.sh` comments (1).

  `decisions.md` keeps its historical names.
- **Measured** (meta device, no weights downloaded, 7 s on the login node):
  - **Qwen2.5-1.5B:** 1.54B parameters, 197 `Linear` modules. Each of `q_proj`, `k_proj`, `v_proj`,
    `o_proj`, `gate_proj`, `up_proj` and `down_proj` appears 28 times, plus `lm_head`.
  - **PEFT's built-in default target list:** `['q_proj','v_proj']` for `qwen2`, and **none** for
    `cohere2` (tiny-aya). Leaving the setting empty would make PEFT fail there.
  - **Hydra sweeps over lists work:** `peft.lora.target_modules=[q_proj,v_proj],[q_proj,k_proj,v_proj,o_proj]`
    gives 2 jobs.

## Proposal

### 1. Schema and option files

- **`LoraConfig` (types only, the Rule):**
  ```python
  r: int = MISSING
  alpha: float = MISSING          # PEFT lora_alpha
  dropout: float = MISSING        # PEFT lora_dropout
  target_modules: List[str] = MISSING   # exact module names; no null, no "all-linear" (Q1)
  ```
  All four drop the `lora_` prefix, since the block is already `peft.lora` (Q2). The comments give
  PEFT's names, so the mapping in `sft_translation.py` is obvious. There is no `null`: the user
  always gets a list, from a `conf/peft/` option or an override. PEFT's per-architecture default is
  never used (it doesn't exist for e.g. cohere2).
- **`conf/peft/lora_default.yaml` and `qlora_default.yaml`:** `r: 8`, `alpha: 32`, `dropout: 0.1`,
  `target_modules: [q_proj, v_proj]` (the same modules as today's `qv`). A comment points to
  `tools/inspect_model.py` for finding names.
- **`check_config`:** for lora/qlora, `target_modules` must be a non-empty list. An empty list
  would fail inside PEFT with a less clear message.
- **No check against the model in `grid.py` (Q3, decided).** Checking names on the login node would mean
  building the model for every launch. A wrong name fails at job start with PEFT's own error
  ("Target modules {...} not found in the base model"), and `inspect_model.py` is how to avoid
  that up front.

### 2. Training code

- **`sft_translation.py`:** replace the mapping with
  ```python
  lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                           r=cfg.peft.lora.r, lora_alpha=cfg.peft.lora.alpha,
                           lora_dropout=cfg.peft.lora.dropout,
                           target_modules=list(cfg.peft.lora.target_modules))
  ```
- **Log the matched modules:** after `get_peft_model`, log the module names actually wrapped
  (the distinct last name components, e.g. `q_proj ×28, v_proj ×28`). It's one line, and it makes a
  match that's narrower or broader than intended visible.
- **Matching rule (documented, unchanged):** PEFT matches each list entry against the **end** of
  a module's full name. So `q_proj` matches `model.layers.0.self_attn.q_proj` in every layer.

### 3. Renames everywhere

- **Sweep axis:** `egs/mmtc/fr-en/sft_lora.sweep.yaml` → `peft.lora.r: 8,16`.
- **Tests:** every `lora_r` in `test_grid.py`. New tests:
  - `target_modules` overridden on the command line (`'peft.lora.target_modules=[q_proj,k_proj,v_proj,o_proj]'`);
  - a list-valued grid axis gives 2 jobs, with readable lines in `grid.txt`;
  - an empty list fails;
  - an old key (`peft.lora.lora_r=8`) fails.
- **Docs:** README and CLAUDE.md examples, and the `launch.sh` usage comment.
- **Saved configs:** run folders saved before this feature fail `inf`/`merge` checks (unknown
  `lora_r`), as agreed: no backward compatibility.
- **W&B:** runs logged before this feature have `hfmt.peft.lora.lora_r`, newer ones
  `hfmt.peft.lora.r`. Grouping across the two needs both columns.

### 4. `tools/inspect_model.py`

`python tools/inspect_model.py <checkpoint>`, where the checkpoint is a Hub id or a local folder
(e.g. a `model_merged/`); there are no short names (Q4). It runs on the login node in the `hfmt` env
and has no flags: it always prints everything (Q6). The model is built from its config on the
**meta device**, so no weights are downloaded, nothing runs on a GPU, and it takes seconds.

It prints these sections:
1. **The model** (`print(model)`).
2. **Module types:** count of each type, e.g. `Linear 197`, `Qwen2RMSNorm 57`.
3. **LoRA candidates:** one row per distinct last name component of the `Linear` modules
   (`q_proj`, …, `down_proj`, `lm_head`), with:
   - its count;
   - its `in_features → out_features`;
   - the adapter parameters at r=8 (`8 × (in + out) × count`).

   Below the table, ready-to-paste `target_modules:` lines for attention only and for all linear
   layers except `lm_head`. It also notes PEFT's built-in default for the architecture, or that
   there is none; that's for information only, since `null` isn't allowed (Q1).
4. **Parameters:** the total, plus the embedding, `lm_head` (tied or not) and the rest.
5. **Training memory estimate** (section 4b below).
6. **Config highlights:** `model_type`, architecture, layers, hidden size, intermediate size,
   attention / KV heads, vocabulary size, max positions, dtype, and a quantization config if present.
7. **Tokenizer and chat template (Q5):**
   - pad/bos/eos tokens;
   - whether there is a chat template;
   - how many tokens the template adds around a one-word user message.

   That last number exposes default system prompts, e.g. the ~366-token Aya preamble found on
   2026-10-18. It matters for `decode.max_length`, and for the sequence length in the memory estimate.

#### 4b. Training memory estimate (developer request)

Estimate peak GPU memory for one training step, for our three modes, at a few common settings.
There are no flags, so it prints a fixed table:
- **Rows:** `full`; `lora` and `qlora` with `[q_proj, v_proj]` at r=8 (today's default); and `lora`
  with all linear layers except `lm_head` at r=16.
- **Columns:** (batch, sequence length) ∈ (16, 128), (16, 512), (32, 512), (8, 2048), with gradient
  checkpointing on (our default).

**Transparency over precision (Q9).** The estimate uses fixed, documented equations; nothing is
fitted. The script prints the equation for every term and the numbers plugged in, e.g.
`logits + loss = b·s·V·14 B = 16·128·151936·14 B = 4.36 GB`, then the total. It does this for one
setting (16 × 512, each mode) in full, then shows the compact table. It also prints the
**per-token cost** of each mode (activations + logits per token in a batch), so any other
(batch, length) can be worked out by hand: memory ≈ fixed part + b·s × per-token part.

Notation: P = all parameters, P_lin = parameters in the quantizable Linear layers (all `Linear`
except `lm_head`), T = trainable parameters (P for full; Σ r·(in+out) over the wrapped modules for
LoRA), b = batch, s = sequence length (after padding to the longest in the batch), h = hidden size,
I = MLP intermediate size, L = layers, n_kv·d = key/value width (KV heads × head dim), V = vocabulary.
Values come from the model's config and the meta-device model. Equations (bytes):

| part | equation | why |
|---|---|---|
| weights, full | 4·P | `load_base_model` loads fp32 for full |
| weights, lora | 2·P | bf16 base |
| weights, qlora | 0.516·P_lin + 4·(P − P_lin) | NF4 is 4 bits, plus double-quantized block constants (≈ 0.127 bits/param, QLoRA paper, Dettmers et al. 2023); `prepare_model_for_kbit_training` casts the non-quantized parameters (embeddings, norms, `lm_head`) to fp32 |
| gradients | 4·T | fp32 master weights (full) / fp32 adapters (lora, qlora) |
| AdamW state | 8·T | two fp32 moments per trainable parameter; with the two lines above, the 16 B/param of mixed-precision Adam (ZeRO paper, Rajbhandari et al. 2020) |
| activations, saved | 2·b·s·h·L | gradient checkpointing on: each decoder layer keeps only its bf16 input |
| activations, one layer recomputed | 2·b·s·(6·h + 2·n_kv·d + 4·I) | during backward one layer's activations exist again: attention (norm output, q, k, v, attention output, o_proj input) and SwiGLU MLP (norm output, gate, up, activation, product); fused/SDPA attention, so no b·a·s² score matrix; bf16. Follows the per-layer accounting of Korthikanti et al. 2022, adapted to GQA + SwiGLU, no dropout |
| logits + loss | 14·b·s·V | bf16 logits (2) + fp32 copy for cross-entropy (4) + its gradient (4) + TRL's entropy/token-accuracy metrics on the logits (4); cf. Wijmans et al. 2024 ("Cut your losses…") on logits dominating with large vocabularies |
| CUDA context | 1.5 GB | constant; allocator fragmentation is not included |

Notes:
- **The logits term is large for big vocabularies.** For Aya (V = 262k) at b=16, s=515 (2.2e9 logit values) it is about
  22–30 GB on its own. That is likely why the Aya runs were so much heavier than Qwen's, together
  with the 366-token preamble raising s.
- **It's a guide, not a guarantee** (expect ±30–50%). The output says so, and that it's for
  training: dev decoding and the evaluation pass usually need less. A worked example for Qwen2.5-1.5B
  lora, 16 × 117 (our fr-en runs): weights 3.1 GB + adapters ~0.01 GB + saved activations 0.16 GB +
  one layer 0.17 GB + logits 4.0 GB + context 1.5 GB ≈ 9 GB allocated.
- **`reserved` ≥ `allocated`.** The estimate targets peak **allocated** memory. What `nvidia-smi`
  or `max_memory_reserved` shows can be much larger. Measured, same Aya config: 75 GB reserved on
  an A100 80GB vs 136–138 GB on an H200. The caching allocator expands into free memory when batch
  shapes vary.

#### 4c. Ballpark check (not calibration)

`sft_translation.py` already logs `max_memory_reserved`; it also logs `max_memory_allocated` (one
line, Q7). In phase B a few short runs compare allocated peaks with the estimate, only to confirm
it's in the right ballpark. The equations are not fitted (Q9). Results go in the plan. Runs: Qwen
lora and qlora at b=16 (s ≈ 120), and Aya qlora (s ≈ 515); plus a small full fine-tune
(Qwen2.5-0.5B) if time allows.

Measured so far (reserved only; not a reliable target, see above):

| run | model | mode | batch × first-batch length | GPU | peak reserved |
|---|---|---|---|---|---|
| p09e-0 | Qwen2.5-1.5B | lora r8 qv | 16 × 117 | H200 | 20 GB |
| p09e-1, sft_lora* | Qwen2.5-1.5B | qlora r8/16 qv | 16 × 117 | H200 / L40S | 19 GB |
| longrun | Qwen2.5-1.5B | lora r8 qv | 32 × 118 | A100 80GB | 78 GB |
| aya_tuk_lora | tiny-aya-global | qlora r8 qv | 16 × 515 | A100 80GB | 75 GB |
| aya_tuk_lora2/3 | tiny-aya-global | qlora r8 qv | 16 × 515 | H200 | 136–138 GB |

### 5. Docs

- **README:** the `peft` paragraph explains `target_modules` and shows `inspect_model.py`.
- **CLAUDE.md:** structure (`tools/inspect_model.py`) and the config bullet (`peft.lora.{r,alpha,dropout,target_modules}`).
- **`decisions.md`:** a feature 11 entry (lists, not names mapped from strings; short keys; the
  meta-device inspection).

## Risks

1. **Wrong module names fail only at job start**, after queueing (Q3). Mitigated by
   `inspect_model.py` and PEFT's clear error.
2. **Suffix matching** can match more than intended for generic names. For example, a name that
   also appears in a vision tower of a multimodal model. The logged matched-module summary (2.)
   shows what was actually wrapped.
3. **The memory estimate can be off** by architecture details it doesn't model: sliding-window
   attention, MoE, vision towers, liger/fused kernels, allocator fragmentation. It's labelled as an
   estimate, prints every equation and number, and is ballpark-checked on our own models.
4. **Building on the meta device** may not work for every architecture: remote code, or a config
   with `quantization_config` (pre-quantized checkpoints). The script reports the error and
   says so. Remote-code models and pre-quantized checkpoints are out of scope (no flags).
5. **Every saved run and old W&B column becomes stale** (accepted: no backward compatibility).
6. **List-valued grid axes** make long `grid.txt` labels and run folder index mappings. That's
   cosmetic: `grid.txt` still maps idx → values.

## Phases

- **A — config + training:** schema, both peft options, `check_config`, `sft_translation.py`
  (`LoraConfig` from the list, matched-module log line), the sweep yaml, tests, the `launch.sh`
  comment. GPU check: a 20-step run with `'peft.lora.target_modules=[q_proj,k_proj,v_proj,o_proj]'`.
  - **Training log:** the matched-module line shows the 4 modules × 28 layers.
  - **Adapter:** `adapter_config.json` lists the same modules, and the trainable parameter count
    equals r × (in + out) summed over the wrapped modules (computed from the printed model).
  - **Inference:** `launch.sh inf` reproduces the test output.
- **B — `tools/inspect_model.py` + memory estimate:**
  - run it on the login node for Qwen2.5-1.5B-Instruct, tiny-aya-global, Qwen2.5-0.5B-Instruct, and a
    local `model_merged/` (if one exists);
  - check its parameter totals and LoRA adapter counts against what training reports, and that the
    Aya chat-template count shows the long preamble;
  - add the `max_memory_allocated` log line to training;
  - run the short ballpark runs from 4c (20 steps each, dev decoding off), and record estimate vs
    measured in the plan (no fitting).
- **C — docs:** README, CLAUDE.md, `decisions.md`; line counts; tests.
- **D — developer validation:** inspect a new base model and train with the module list it suggests.

## Decided

- **Q1** `target_modules: List[str] = MISSING`, with no `null`: the user sets it through a `conf/peft/`
  option or an override. `check_config` rejects an empty list.
- **Q2** All four keys are renamed: `peft.lora.r`, `alpha`, `dropout`, `target_modules`.
- **Q3** No module-name check at launch; PEFT's error at job start and `inspect_model.py` cover it.
- **Q4** `inspect_model.py` takes a Hub model id or a local folder, with no short names.
- **Q5** The tokenizer/chat-template section is included.
- **Q6** No flags: the script always prints everything. (v3: the LoRA-candidates section is back,
  at the developer's request.)
- **Q7** Training also logs `max_memory_allocated`.
- **Q9** No calibration: fixed equations, every term printed with its numbers; a few short runs
  only to check the ballpark.
- **Q8** The fixed table is as proposed. Rows: full; lora qv r=8; qlora qv r=8; lora all-linear
  (except `lm_head`) r=16. Columns: 16×128, 16×512, 32×512, 8×2048; gradient checkpointing on.
