# 14_tiny_aya — plan (v7, draft)

Status: draft, 2026-11-11. Decided: no beam decodes; one extra seed per pair (Q4). Open: Q1–Q3, Q5–Q7.
`todo.md` is generated once they are decided.

## Request

Add `CohereLabs/tiny-aya-global`, and from the fr-en and zh-en results in `_results/` define its
experiments plus the few Qwen3.5 and Gemma 3 experiments still worth running: one full fine-tune, a few
LoRAs, nothing more.

## What the results say

Test BLEU-lc, greedy, one seed each; noise band about 0.5.

| Finding | fr-en | zh-en | For Tiny Aya |
|---|---|---|---|
| Size is the main lever (best run per model) | 0.8B 61.0, 2B 65.6, 4B 68.3, 9B 69.0 | 0.8B 29.6, 2B 33.1, 4B 34.8, 9B 35.6 | Compare with Qwen3.5-2B and 4B |
| All-linear LoRA beats q,v; QLoRA loses; rank 8 = 16 | +1.6 to +2.6 | +0.4 to +0.8 | All-linear LoRA only |
| The all-linear lr axis 3e-5..2e-4 is flat | spread 0.5–1.2 | spread 0.2–0.7 | lr 1e-4, plus 2e-4 |
| Full fine-tune ties LoRA, or trails by under 1 | 2B 65.5 vs 65.6; 4B 67.2 vs 67.9 | 2B 33.1 vs 32.8; 4B 34.8 vs 34.5 | One full fine-tune per pair |
| The annealed schedule gains nothing | 68.3 vs 68.2 | 34.5 vs 34.5 | `shortrun`, no anneal runs |
| Gemma 3 zero-shot is chatty | 6.0, TER 463 | 6.8, TER 242 | Read zero-shot scores with output lengths |

Housekeeping: `_results/zh-en/qwen-3.5-9b/full` and `zh-en/gemma3-4b/full` are all-linear LoRA runs,
misnamed. The 4B and TranslateGemma short full fine-tunes are done on both pairs, so both `notes.md` have
stale "Next" items.

## The model

- `Cohere2ForCausalLM`, 3.35B parameters, 36 layers, 262k vocabulary with tied `lm_head`, 8k context.
  Instruction-tuned, 70+ languages. CC-BY-NC 4.0, gated on the Hub.
- LoRA targets: the standard seven names, the same list as `conf/peft/lora_gemma3_all_linear.yaml`.
- The chat template always adds a fixed system preamble: **prompts are 382–386 tokens** (Qwen3.5 adds 17,
  Gemma 3 14). It is out of the loss but in every forward pass (Q1).
- Measured on a 50-step LoRA test (fr-en, A100 40 GB, batch 8 x 8, eval batch 8): ~9 s per step, 358 s
  per dev decode, no OOM; dev at step 25 already 61.25 BLEU-lc, TER 31.19.

## Proposal

### 1. Code and config

- `conf/peft/lora_tiny_aya_all_linear.yaml` (name: Q5): r 8, alpha 32, dropout 0.1, the seven names.
- No prompt-path change (Q1).
- Run-on counter (Q7): the "Testing - ..." log line also counts hypotheses that reached
  `decode.max_new_tokens`. Diagnostics only.
- `hfmt/test_grid.py` `test_each_peft_option_composes`: add the per-model peft options.
- Docs: README option list; a `decisions.md` entry.

### 2. Tiny Aya experiments (both pairs)

| File | What | Jobs | GPU |
|---|---|---|---|
| `sft_lora_tiny-aya-global_all-linear.sweep.yaml` | all-linear LoRA, `shortrun`, BLEU-lc selection, lr 1e-4,2e-4 in the file (Q3) | 2 | 40 GB A100 or L40S |
| `sft_tiny-aya-global.yaml` with `launch.sh inf ... peft=lora_tiny_aya_all_linear slurm.constraint=null` | zero-shot baseline, bf16 (`inf` refuses a sweep file) | 1 | any A100 |
| `sft_tiny-aya-global.yaml` | full fine-tune, `full_finetune_short`, lr 1e-5 (Q2) | 1 | 80 GB A100 |

- LoRA: batch 8 x 8 (effective 64), eval batch 8.
- Full: batch 4 x 16, eval batch 8, `slurm.constraint: 80gb`, `slurm.mem_gb: 48`; ~57 GiB estimated, ~95 GB
  disk per run.
- `slurm.time: 8:00:00` for zh-en and the full fine-tunes.

Expected: between Qwen3.5-2B and 4B after SFT (fr-en 65.5–68, zh-en 33–34.5).

### 3. Qwen3.5 and Gemma 3

| Experiment | Jobs | Why |
|---|---|---|
| Seed 1 for Qwen3.5-4B all-linear LoRA, lr 1e-4: `sft_lora_qwen-3.5-4b_all-linear_seed1.yaml` | 1 per pair | One seed-to-seed difference, to check the ~0.5 noise band |
| Gemma 3 zero-shot with a stricter instruction, `inf_name=zeroshot_strict` (Q6) | 1 per pair | Format or ability behind the 6-point zero-shot |

Not planned: Qwen3.5-0.8B LoRA, a full fine-tune lr bracket, anneal/q,v/QLoRA/rank runs, beam search
(developer), more seeds (developer), a 9B full fine-tune (does not fit one GPU), other Tiny Aya variants.

### 4. Launch order (fr-en first)

0. Done: license and download; a 50-step LoRA test (`run_name=stoptest_tiny-aya`). To do: read its step-50
   dev scores, then delete its run folder.
1. Done (2026-11-11): peft option, the Tiny Aya and seed-1 yamls for both pairs (each sets the BLEU-lc
   selection and learning rate itself; `conf/train` is unchanged), test and README. Every existing
   experiment composes unchanged. Still to do: the run-on counter (Q7).
2. fr-en zero-shot, then LoRA (both lrs), then full.
3. zh-en zero-shot, LoRA, full.
4. The seed-1 runs.
5. Pull into `_results/<pair>/tiny-aya-global/`, rename the two misnamed folders, rebuild `index.html`,
   update both `notes.md`.

### 5. Risks

- **Zero-shot output**: sometimes chatty (a wrapper around the translation), curly apostrophes, possible
  refusals of offensive tweets. Read the score with output lengths.
- **End tokens**: the generation config stops on `<EOS_TOKEN>` (3), the template ends turns with
  `<|END_OF_TURN_TOKEN|>` (6). No run-ons seen zero-shot or in the LoRA test; the counter (§1) watches.
- **Compute**: the preamble triples tokens per step; covered by the batch and time settings.
- **Full fine-tune**: memory, disk and the tied-weight reload are untested; `model_dev_check` confirms the
  reload, and an OOM resumes on an 80 GB A100.
- **TRL prompt mismatch warning**: not expected (the template keeps the generation prompt a prefix);
  `tools/run_status.py` flags it.

## Open questions (recommendation first)

- **Q1 — Preamble: keep it.** No template switch removes it; dropping it means a per-model template patch.
- **Q2 — Full fine-tune lr: 1e-5**, or 5e-6. Rerun at 5e-6 if dev peaks at step 25 and falls.
- **Q3 — LoRA lrs: 1e-4 and 2e-4**, or 1e-4 alone.
- **Q4 — decided: seed 1 per pair.**
- **Q5 — Option name: `lora_tiny_aya_all_linear.yaml`**, or `lora_cohere2_all_linear.yaml`.
- **Q6 — Gemma 3 zero-shot check: run it** (two decode jobs).
- **Q7 — End tokens: the run-on counter only**, or also pass `generate` the tokenizer's eos. The second
  changes Qwen3.5's token stream, so only if a run-on shows up.

## Done when

- Tests pass and every new yaml composes (done); the step-50 test result is read and its folder deleted.
- Tiny Aya zero-shot, LoRA (both lrs) and full results on both pairs, with `model_dev_check: ok` and no
  hypotheses at `max_new_tokens`.
- Seed-1 runs in `_results/`, `index.html` rebuilt, `notes.md` updated, `decisions.md` entry written.
