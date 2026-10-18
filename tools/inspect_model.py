"""Inspect a causal-LM checkpoint before fine-tuning it (feature 11).

    python tools/inspect_model.py <checkpoint>      # a Hugging Face model id or a local folder

Builds the model from its config on the meta device: no weights are downloaded and no GPU is
used, so it runs on a login node in seconds (in the hfmt env: source install/path.sh). Prints:
  1. the model                       5. a training memory estimate (equations + numbers)
  2. module types with counts        6. config highlights
  3. LoRA candidates (Linear names)  7. tokenizer and chat-template overhead
  4. parameter counts
Remote-code models and pre-quantized checkpoints are not supported.
"""

import sys
from collections import Counter, defaultdict

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

GIB = 1024 ** 3
SETTINGS = [(16, 128), (16, 512), (32, 512), (8, 2048)]    # (batch, sequence length) table columns
DETAIL = (16, 512)                                         # setting shown term by term
CONTEXT_BYTES = 1.5 * GIB                                   # CUDA context: outside PyTorch's allocator
NF4_BYTES = 4.127 / 8                                       # NF4 + double-quantized constants (QLoRA paper)


def header(title):
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


def gib(n_bytes):
    return f"{n_bytes / GIB:.2f} GiB"


def linear_groups(model, lm_head_name):
    """Linear modules grouped by the last component of their name (what LoRA target_modules match)."""
    groups = defaultdict(list)    # name -> [(in, out, parent type), ...]
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            parent = type(model.get_submodule(name.rsplit(".", 1)[0])).__name__ if "." in name else ""
            groups[name.rsplit(".", 1)[-1]].append((module.in_features, module.out_features, parent))
    return groups


def adapter_params(groups, names, r):
    """LoRA parameters for wrapping every module named in ``names``: r·(in + out) each."""
    return sum(r * (i + o) for n in names for i, o, _ in groups.get(n, []))


def main(checkpoint):
    # Build on the meta device: real shapes, no memory, no download of weights.
    try:
        config = AutoConfig.from_pretrained(checkpoint)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config)
    except Exception as e:
        sys.exit(f"inspect_model: can't build {checkpoint} on the meta device: {e}")
    tc = config.get_text_config()
    out_emb, in_emb = model.get_output_embeddings(), model.get_input_embeddings()
    lm_head_name = next(n for n, m in model.named_modules() if m is out_emb)
    groups = linear_groups(model, lm_head_name)
    lm_head_short = lm_head_name.rsplit(".", 1)[-1]

    # ------------------------------------------------------------------------------------------
    header(f"1. Model: {checkpoint}")
    print(model)

    # ------------------------------------------------------------------------------------------
    header("2. Module types (count)")
    for name, n in Counter(type(m).__name__ for m in model.modules()).most_common():
        print(f"  {name:40s} {n:6d}")

    # ------------------------------------------------------------------------------------------
    header("3. LoRA candidates: Linear modules by name (what peft.lora.target_modules matches)")
    r = 8
    print(f"  {'name':16s} {'count':>5s}  {'in -> out':18s} {'parent':24s} {f'adapter params at r={r}':>24s}")
    for name, items in groups.items():
        shapes = sorted({(i, o) for i, o, _ in items})
        shape = f"{shapes[0][0]} -> {shapes[0][1]}" if len(shapes) == 1 else "varies"
        parents = ",".join(sorted({p for _, _, p in items}))
        print(f"  {name:16s} {len(items):5d}  {shape:18s} {parents:24s} {adapter_params(groups, [name], r):>24,d}")
    attention = [n for n, items in groups.items() if any("Attention" in p for _, _, p in items)]
    all_linear = [n for n in groups if n != lm_head_short]
    print("\n  Ready to paste into conf/peft/<option>.yaml (lora: block) or as an override:")
    print(f"    attention only:            target_modules: [{', '.join(attention)}]")
    print(f"    all linear except {lm_head_short}: target_modules: [{', '.join(all_linear)}]")
    try:
        from peft.utils.constants import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING as peft_defaults
        default = peft_defaults.get(config.model_type)
        print(f"  PEFT's built-in default for model_type={config.model_type}: {default or 'none'} "
              "(for information; hfmt always uses the listed target_modules)")
    except ImportError:
        pass

    # ------------------------------------------------------------------------------------------
    header("4. Parameters")
    P = sum(p.numel() for p in model.parameters())                  # shared (tied) weights once
    tied = out_emb is not None and out_emb.weight is in_emb.weight
    P_emb = in_emb.weight.numel()
    P_head = 0 if tied else out_emb.weight.numel()
    P_lin = sum(m.weight.numel() for n, m in model.named_modules()   # what bitsandbytes quantizes
                if isinstance(m, torch.nn.Linear) and n != lm_head_name)
    print(f"  total                         {P:>16,d}")
    print(f"  input embedding               {P_emb:>16,d}")
    print(f"  {lm_head_name:30s}{P_head:>16,d}  ({'tied to the input embedding' if tied else 'separate'})")
    print(f"  rest (layers, norms)          {P - P_emb - P_head:>16,d}")
    print(f"  of which Linear weights (P_lin, quantized by qlora): {P_lin:,d}")

    # ------------------------------------------------------------------------------------------
    header("5. Training memory estimate (peak *allocated* GPU memory for one training step)")
    h, L, I, V = tc.hidden_size, tc.num_hidden_layers, tc.intermediate_size, tc.vocab_size
    head_dim = getattr(tc, "head_dim", None) or h // tc.num_attention_heads
    kv = tc.num_key_value_heads * head_dim
    rows = [("full", None, None),
            ("lora   [q_proj, v_proj] r=8", ["q_proj", "v_proj"], 8),
            ("qlora  [q_proj, v_proj] r=8", ["q_proj", "v_proj"], 8),
            ("lora   all linear r=16", all_linear, 16)]
    print("  A guide, not a guarantee (expect +-30%); gradient checkpointing on; AdamW; bf16 activations.")
    print(f"  The GPU needs more: + ~{CONTEXT_BYTES / GIB:.1f} GiB CUDA context, + what the caching allocator keeps")
    print("  (nvidia-smi / max_memory_reserved; with varying batch shapes it can grow to fill the GPU).")
    print("  Checked on A100s (2026-10-18; Qwen2.5-0.5B/1.5B, tiny-aya; 20 steps): measured / estimate = 0.86-1.04.")
    print(f"  Values: P={P:,}  P_lin={P_lin:,}  h={h}  I={I}  L={L}  n_kv*head_dim={kv}  V={V:,}")

    def fixed_terms(mode, names, rank):
        """(label, equation with numbers, bytes) for the parts that don't depend on batch/length."""
        T = P if mode == "full" else adapter_params(groups, names, rank)
        if mode == "full":
            weights = ("weights (fp32)", f"4*P = 4*{P:,}", 4 * P)
        elif mode == "lora":
            weights = ("weights (bf16) + adapters (fp32)", f"2*P + 4*T = 2*{P:,} + 4*{T:,}", 2 * P + 4 * T)
        else:
            weights = ("weights (NF4 Linear, fp32 rest) + adapters",
                       f"{NF4_BYTES:.3f}*P_lin + 4*(P-P_lin) + 4*T = {NF4_BYTES:.3f}*{P_lin:,} + 4*{P - P_lin:,} + 4*{T:,}",
                       NF4_BYTES * P_lin + 4 * (P - P_lin) + 4 * T)
        return T, [weights,
                   ("gradients (fp32)", f"4*T = 4*{T:,}", 4 * T),
                   ("AdamW state (m, v fp32)", f"8*T = 8*{T:,}", 8 * T)]

    def token_terms(b, s):
        """(label, equation with numbers, bytes) for the parts that grow with batch * length."""
        return [("activations saved (layer inputs)", f"2*b*s*h*L = 2*{b}*{s}*{h}*{L}", 2 * b * s * h * L),
                ("one layer recomputed in backward", f"2*b*s*(6h + 2*kv + 4I) = 2*{b}*{s}*(6*{h} + 2*{kv} + 4*{I})",
                 2 * b * s * (6 * h + 2 * kv + 4 * I)),
                ("logits + loss", f"14*b*s*V = 14*{b}*{s}*{V:,}", 14 * b * s * V)]

    b, s = DETAIL
    for label, names, rank in rows:
        mode = label.split()[0]
        T, fixed = fixed_terms(mode, names, rank)
        terms = fixed + token_terms(b, s)
        note = "" if mode == "full" or T else "   (none of these names is in this model)"
        print(f"\n  {label} at batch {b} x length {s}  (T = trainable = {T:,}){note}")
        for name, eq, n_bytes in terms:
            print(f"    {name:42s} {eq:96s} = {gib(n_bytes):>11s}")
        print(f"    {'total':42s} {'':96s} = {gib(sum(t[2] for t in terms)):>11s}")

    per_token = sum(t[2] for t in token_terms(1, 1))
    print(f"\n  Per token in a batch (all modes): 2*h*L + 2*(6h + 2*kv + 4I) + 14*V = {per_token:,} B "
          f"= {per_token / 1024 ** 2:.2f} MiB")
    print("  So: allocated ~ fixed part (weights, gradients, AdamW) + batch * length * per-token part.")

    print(f"\n  {'GiB allocated':34s}" + "".join(f"{f'{b}x{s}':>12s}" for b, s in SETTINGS))
    for label, names, rank in rows:
        _, fixed = fixed_terms(label.split()[0], names, rank)
        totals = [sum(t[2] for t in fixed) + sum(t[2] for t in token_terms(b, s)) for b, s in SETTINGS]
        print(f"  {label:34s}" + "".join(f"{t / GIB:12.1f}" for t in totals))
    print("  (columns: batch x sequence length after padding to the longest in the batch)")

    # ------------------------------------------------------------------------------------------
    header("6. Config highlights")
    quant = getattr(config, "quantization_config", None)
    for key, value in [("model_type", config.model_type), ("architectures", getattr(config, "architectures", None)),
                       ("layers", L), ("hidden size", h), ("intermediate size", I),
                       ("attention heads", tc.num_attention_heads), ("key/value heads", tc.num_key_value_heads),
                       ("head dim", head_dim), ("vocabulary", V),
                       ("max positions", getattr(tc, "max_position_embeddings", None)),
                       ("sliding window", getattr(tc, "sliding_window", None)),
                       ("dtype", getattr(tc, "dtype", None)), ("tie_word_embeddings", tied),
                       ("quantization_config", quant if quant else "none")]:
        print(f"  {key:22s} {value}")

    # ------------------------------------------------------------------------------------------
    header("7. Tokenizer and chat template")
    try:
        tok = AutoTokenizer.from_pretrained(checkpoint)
    except Exception as e:
        print(f"  can't load the tokenizer: {e}")
        return
    for key in ("pad", "bos", "eos"):
        print(f"  {key}_token  {getattr(tok, f'{key}_token')!r:30s} id {getattr(tok, f'{key}_token_id')}")
    print(f"  vocabulary (tokenizer)  {len(tok):,}")
    if not tok.chat_template:
        print("  no chat template")
        return
    word = "Hello"
    templated = tok.apply_chat_template([{"role": "user", "content": word}], add_generation_prompt=True,
                                        tokenize=True, return_dict=True)["input_ids"]
    plain = tok(word, add_special_tokens=False)["input_ids"]
    print(f"  chat template: yes; it adds {len(templated) - len(plain)} tokens around a one-word user message")
    print("  (includes any default system prompt; counts toward decode.max_length and the sequence length above)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python tools/inspect_model.py <checkpoint>   (Hugging Face model id or local folder)")
    main(sys.argv[1])
