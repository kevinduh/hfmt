"""Quick test for plan.md R1 (16_cpt): does TRL's padding-free packing leak context between examples?

For each model, the same made-up sentences are scored two ways, and each example's mean next-token loss is compared:
  alone   one example per forward pass (what decoding sees);
  packed  all examples in one padding-free row from TRL's own collator, as CPT trains:
          DataCollatorForLanguageModeling(padding_free=True) -> input_ids [1, total], position_ids
          restarting at 0 for each example, no attention_mask.
If packing keeps examples apart, packed == alone for every example. The first example of the row has
nothing before it, so its difference is the numeric noise floor; a leak shows as later examples
differing far more than that.

Suspected (from reading transformers 5.8.0): plain attention layers are kept apart (masking_utils builds
a per-example mask from position_ids), but Qwen3.5's Gated DeltaNet layers get no sequence boundaries
(seq_idx=None, no cu_seqlens), so Qwen3.5 leaks and the plain-attention controls don't.

Runs hfmt's own loading (mt_model.load_base_model, lora mode = bf16 weights), then casts to float32 by
default so numeric noise is small, with use_cache=False as in training (job 1252483 ran with the default
cache on, which makes transformers skip the per-example mask, so every model, controls too, leaked). Inputs are made-up sentences, never dataset content; nothing goes to W&B.

    python features/16_cpt/test_packing_leak.py [--bf16] [model ...]
    sbatch features/16_cpt/test_packing_leak.sh            # on a GPU node
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

MODELS = ["Qwen/Qwen3.5-2B",              # hybrid: Gated DeltaNet + attention (suspected to leak)
          "CohereLabs/tiny-aya-global",   # control: attention only (the synlrl CPT model)
          "Qwen/Qwen3-0.6B"]              # control: attention only
SENTENCES = [
    "The small cat sat on the warm windowsill all afternoon.",
    "Rain is expected over the northern hills tomorrow morning.",
    "She bought three green apples and a loaf of bread.",
    "The old bridge was closed for repairs last spring.",
    "Our team finished the project two days ahead of schedule.",
    "A quiet river runs behind the village school.",
    "He forgot his umbrella at the train station again.",
    "The museum opens at nine and closes at five.",
]


def per_example_losses(logits, input_ids, lengths):
    """Mean next-token cross-entropy of each example in a row of concatenated examples (lengths in order)."""
    import torch.nn.functional as F

    losses, start = [], 0
    for n in lengths:
        end = start + n
        pred = logits[0, start:end - 1].float()           # positions predicting tokens start+1 .. end-1
        gold = input_ids[0, start + 1:end]
        losses.append(F.cross_entropy(pred, gold).item())
        start = end
    return losses


def run(checkpoint, fp32, device):
    import torch
    from omegaconf import OmegaConf
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    from hfmt.mt_model import load_base_model, load_tokenizer

    tokenizer = load_tokenizer(checkpoint)
    cfg = OmegaConf.create({"checkpoint": checkpoint, "peft": {"mode": "lora"}})
    model, _ = load_base_model(cfg, device)
    if fp32:
        model = model.float()
    model.eval()
    # As training runs it (train_common.build_model sets this): with a cache, transformers skips the
    # per-example mask it builds from position_ids (masking_utils: only when past_key_values is None).
    model.config.use_cache = False
    # As TRL tokenizes a `text` example: the tokenizer's own special tokens, then its eos_token appended.
    examples = [tokenizer(s + tokenizer.eos_token)["input_ids"] for s in SENTENCES]
    lengths = [len(ids) for ids in examples]

    with torch.no_grad():
        alone = []
        for ids in examples:
            x = torch.tensor([ids], device=device)
            alone += per_example_losses(model(input_ids=x, use_cache=False).logits, x, [len(ids)])
        collator = DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id or 0, padding_free=True)
        batch = collator([{"input_ids": ids} for ids in examples])
        assert "attention_mask" not in batch and batch["input_ids"].shape == (1, sum(lengths))
        x, pos = batch["input_ids"].to(device), batch["position_ids"].to(device)
        packed = per_example_losses(model(input_ids=x, position_ids=pos, use_cache=False).logits, x, lengths)

    diffs = [abs(a - p) for a, p in zip(alone, packed)]
    impl = getattr(model.config, "_attn_implementation", "?")
    layer_types = getattr(getattr(model.config, "text_config", model.config), "layer_types", None)
    kinds = sorted(set(layer_types)) if layer_types else ["(no layer_types)"]
    print(f"\n== {checkpoint}  dtype={next(model.parameters()).dtype}  attn={impl}  layers={kinds}")
    print(f"{'ex':>3} {'tokens':>6} {'alone':>9} {'packed':>9} {'|diff|':>10}")
    for i, (n, a, p, d) in enumerate(zip(lengths, alone, packed, diffs)):
        print(f"{i:>3} {n:>6} {a:>9.5f} {p:>9.5f} {d:>10.2e}" + ("   (first: noise floor)" if i == 0 else ""))
    later = max(diffs[1:])
    print(f"noise floor (ex 0) {diffs[0]:.2e}; max later {later:.2e}; ratio {later / max(diffs[0], 1e-12):.1f}x")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return diffs


def main():
    import torch

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("models", nargs="*", default=MODELS)
    p.add_argument("--bf16", action="store_true", help="keep bf16 weights (default: cast to float32)")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  torch={torch.__version__}")
    for checkpoint in args.models:
        run(checkpoint, not args.bf16, device)


if __name__ == "__main__":
    main()
