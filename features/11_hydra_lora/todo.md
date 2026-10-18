# 11_hydra_lora — todo

Generated from `plan.md` (v4, final). Groups follow the plan's phases; items within a group are
ordered by dependency. Commit once per phase, with subject `11_hydra_lora-<phase>: <summary>`.
Claude presents the message and the developer commits. Each phase ends with its "Done when" checks
before the next phase starts.

Ground rules for every phase:
- **The Rule:** `hydra_config.py` holds types only. Value fields are `MISSING`. Every default lives
  in exactly one yaml.
- **No backward compatibility:** older saved run configs (with `lora_r`, …) are expected to fail.
- **No dataset content to W&B** (CLAUDE.md).
- **Don't touch the untracked leftovers.** Stage files by name.
- **Tests:** `HFMT_ROOT=$(pwd) python hfmt/test_grid.py`. GPU test runs go to `egs/mmtc/fr-en/outputs/`
  as `p11<phase>…`, and are deleted once confirmed.

## A. Config + training (requests 1, 2)

### A1. Schema and options
- [x] `LoraConfig`: `r: int`, `alpha: float`, `dropout: float`, `target_modules: List[str]`, all
      `MISSING`. Comments give PEFT's names (`lora_alpha`, `lora_dropout`) and say exact module
      names, matched against the end of each module's name.
- [x] `conf/peft/lora_default.yaml` and `conf/peft/qlora_default.yaml`: `r: 8`, `alpha: 32`,
      `dropout: 0.1`, `target_modules: [q_proj, v_proj]`, with a comment pointing to `tools/inspect_model.py`.
- [x] `check_config`: for lora/qlora, `target_modules` must be non-empty.

### A2. Training code
- [x] `sft_translation.py`: delete the `lora_target` mapping. Build `LoraConfig(r=…, lora_alpha=…,
      lora_dropout=…, target_modules=list(cfg.peft.lora.target_modules))`.
- [x] After `get_peft_model`, log the wrapped modules by their last name component with counts
      (e.g. `LoRA wraps: q_proj x28, v_proj x28`).

### A3. Renames
- [x] `egs/mmtc/fr-en/sft_lora.sweep.yaml`: axis `peft.lora.r: 8,16`.
- [x] `egs/launch.sh` usage comment: `peft.lora.r=8,16`.
- [x] `hfmt/test_grid.py`: every `lora_r` → `r`.

### A4. Tests
- [x] A `target_modules` list override is saved as a list.
- [x] A list-valued grid axis gives 2 jobs with readable `grid.txt` lines.
- [x] `target_modules=[]` fails; an old key (`peft.lora.lora_r=8`) fails.

### Done when
- [x] Tests pass.
- [x] GPU: 20-step run with `'peft.lora.target_modules=[q_proj,k_proj,v_proj,o_proj]'` (dev decoding off).
  - The log shows the 4 modules × 28.
  - `adapter_config.json` lists them.
  - Trainable parameters = r × Σ(in + out) over the wrapped modules.
  - `launch.sh inf` reproduces `test1.pred.trg`.
- [x] `todo.md` A ticked; commit message presented.

## B. `tools/inspect_model.py` + memory estimate (request 3)

### B1. Script
- [ ] `tools/inspect_model.py <checkpoint>` (Hub id or local folder, no flags). Builds the model on
      the meta device from its config (no weights) and loads the tokenizer.
- [ ] Sections, in order:
  1. the model;
  2. module types with counts;
  3. LoRA candidates (name, count, in→out, adapter params at r=8), ready-to-paste `target_modules:`
     lines (attention only; all linear except `lm_head`), and PEFT's default for the architecture
     or "none";
  4. parameters (total, embedding, `lm_head` tied or not, rest);
  5. training memory estimate;
  6. config highlights;
  7. tokenizer + chat-template overhead (tokens around a one-word user message).
- [ ] Memory estimate (plan 4b equations, constants as documented, nothing fitted):
  - each term printed with its equation and the numbers plugged in, for 16 × 512 per mode;
  - per-token cost per mode;
  - the table: rows full, lora qv r8, qlora qv r8, lora all-linear r16; columns 16×128, 16×512,
    32×512, 8×2048; gradient checkpointing on;
  - a header line: estimate of peak *allocated* training memory, ±30–50%.
- [ ] Clear failure message if the meta-device build fails (remote code, pre-quantized checkpoints
      are out of scope).

### B2. Training log
- [ ] `sft_translation.py`: log `torch.cuda.max_memory_allocated()` next to the reserved line.

### Done when
- [ ] Runs on the login node for Qwen2.5-1.5B-Instruct, tiny-aya-global, Qwen2.5-0.5B-Instruct, and
      a local `model_merged/` if one exists.
- [ ] Parameter totals and LoRA adapter counts match what training reports. The Aya template
      overhead shows the long preamble.
- [ ] Ballpark runs (20 steps, dev decoding off): Qwen lora and qlora b=16, Aya qlora b=16, and
      Qwen2.5-0.5B full if time allows. Estimate vs allocated recorded in the plan (no fitting).
- [ ] `todo.md` B ticked; commit message presented.

## C. Docs

- [ ] README: the `peft` paragraph (`peft.lora.{r,alpha,dropout,target_modules}`, how to find module
      names with `tools/inspect_model.py`), and examples renamed.
- [ ] CLAUDE.md: structure (`tools/inspect_model.py`), the config bullet, examples renamed.
- [ ] `decisions.md`: feature 11 entry (module lists instead of mapped strings, short keys, the
      meta-device inspection and memory estimate).
- [ ] Line counts since the feature's start commit. Tests pass.
- [ ] `todo.md` C ticked; commit message presented.

## D. Developer validation

- [ ] Inspect a new base model and train with the module list it suggests.
