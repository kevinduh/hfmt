"""Model loading, prompting, decoding and scoring shared by SFT and inference (feature 09).

Training (``sft_translation.py``) and inference (``inf_translation.py``) load the base model the
same way for each ``peft.mode``, so a trained adapter is always decoded on the base it was
trained on (e.g. 4-bit only for ``qlora``).

Heavy imports (torch, transformers, ...) live inside functions so config-only use stays fast.
"""

import time
import logging


# Extra arguments for every ``apply_chat_template`` call that renders a prompt (training via TRL's
# per-example ``chat_template_kwargs`` column, dev/test decoding, inference), so prompts render the same
# everywhere. enable_thinking=False: Qwen3.5-4B's template thinks by default (generation prompt
# `<think>\n`, so the model writes its reasoning as the translation); 0.8B/2B already default to off.
# Off, the generation prompt ends with the empty `<think>\n\n</think>\n\n` that the training targets
# start with. Templates without the variable (Gemma) ignore it, so their prompts are unchanged.
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}


def format_input_prompt(instruction_prefix, content):
    """The user turn of the chat prompt for one source sentence: the instruction, a newline, the source."""
    return [{"content": instruction_prefix + "\n" + content, "role": "user"}]


def load_tokenizer(checkpoint):
    """``checkpoint``'s tokenizer, with the eos token as pad token when it has none. Every hfmt path
    loads its tokenizer here."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(cfg, device):
    """Load ``cfg.checkpoint`` for ``cfg.peft.mode``: fp32 for ``full``, bf16 for ``lora``, 4-bit
    with bf16 compute for ``qlora``. bf16 includes PyTorch's emulation, so a V100 (no native bf16)
    runs the same precision as an A100/L40S; fp16 is used only on a card with no bf16 at all.
    Returns ``(model, use_bf16)``."""
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    finetune_mode = cfg.peft.mode  # one of: full | lora | qlora
    # Emulated bf16 counts: same numerics on every card, at a speed cost on V100s (fp16 breaks
    # Gemma 3). decisions.md, features/15_running_on_v100.
    use_bf16 = torch.cuda.is_bf16_supported(including_emulation=True)
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    logging.info(f"finetune_mode: {finetune_mode}, use_bf16: {use_bf16}, compute_dtype: {compute_dtype}")
    if finetune_mode in ("full", "lora"):
        model_dtype = torch.float32 if finetune_mode == "full" else compute_dtype
        model = AutoModelForCausalLM.from_pretrained(
            cfg.checkpoint,
            dtype=model_dtype,
        ).to(device)
    elif finetune_mode == "qlora":
        model_dtype = compute_dtype
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=cfg.peft.quant.load_in_4bit,
            bnb_4bit_use_double_quant=cfg.peft.quant.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=cfg.peft.quant.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.checkpoint,
            dtype=model_dtype,
            quantization_config=bnb_config,
            device_map = {"": torch.cuda.current_device()}
        )
    else:
        raise ValueError(f"Unknown mode: {finetune_mode}")
    return model, use_bf16


def load_checkpoint_weights(model, checkpoint_dir):
    """Copy the full weights saved in ``checkpoint_dir`` (a Trainer ``checkpoint-N/``) into ``model``
    in place, through ``from_pretrained``.

    The Trainer reloads a full model's checkpoint (the best one at the end of training, the last one
    on resume) with a plain non-strict ``load_state_dict``, but ``save_pretrained`` writes the
    checkpoint's original (Hub) weight names, which ``from_pretrained`` converts to the module names
    on load: for Qwen3.5, ``model.language_model.*`` in the files vs ``model.*`` in the model. So
    nothing matches, nothing loads, and the Trainer only warns ("There were missing keys in the
    checkpoint model loaded"), leaving the weights as they were. Loading the checkpoint as a model
    applies the conversion; copying its weights with ``strict=True`` makes any mismatch fail the run.
    Needs CPU RAM for one copy of the weights, as the Trainer's own reload does."""
    dtype = next(model.parameters()).dtype
    logging.info(f"Loading the weights of {checkpoint_dir} into the model (dtype {dtype})")
    saved = type(model).from_pretrained(checkpoint_dir, dtype=dtype)   # on the CPU
    model.load_state_dict(saved.state_dict(), strict=True)
    del saved


def load_trained_model(cfg, model_dir, device):
    """Load what ``sft_translation.py`` saved in ``model_dir``: full weights for ``full``, or an
    adapter on the base from ``load_base_model`` for ``lora``/``qlora``. A ``qlora`` base gets the
    same preparation as in training (non-quantized parameters cast to fp32), so decoding matches
    the test predictions written at the end of training."""
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel, prepare_model_for_kbit_training

    if cfg.peft.mode == "full":
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).to(device)
    else:
        base, _ = load_base_model(cfg, device)
        if cfg.peft.mode == "qlora":
            base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
        model = PeftModel.from_pretrained(base, model_dir)
    model.eval()
    return model


QLORA_MERGE_WARNING = ("qlora merge: the adapter was merged into the 4-bit base dequantized to bf16 (the "
                       "weights it was trained against), so the merged weights carry the 4-bit rounding of "
                       "the original checkpoint")


def merge_trained_model(cfg, model_dir, out_dir, device):
    """Fold the ``lora``/``qlora`` adapter in ``model_dir`` into its base and save a full bf16
    checkpoint (weights + tokenizer) in ``out_dir``, usable as ``checkpoint`` for another round.
    ``lora``: merged into the bf16 base. ``qlora``: the 4-bit base is dequantized to bf16 first, so
    the merged model is the one that was trained and evaluated; logs ``QLORA_MERGE_WARNING``.
    Returns the warning text, or None."""
    import torch
    from peft import PeftModel

    if cfg.peft.mode not in ("lora", "qlora"):
        raise ValueError(f"peft.mode={cfg.peft.mode}: only lora/qlora adapters are merged "
                         "(a full fine-tune's model/ already holds full weights)")
    base, _ = load_base_model(cfg, device)   # bf16 for lora, 4-bit for qlora
    warning = None
    if cfg.peft.mode == "qlora":
        base = base.dequantize(dtype=torch.bfloat16)
        warning = QLORA_MERGE_WARNING
        logging.warning(warning)
    model = PeftModel.from_pretrained(base, model_dir).merge_and_unload()
    model.save_pretrained(out_dir)
    load_tokenizer(cfg.checkpoint).save_pretrained(out_dir)
    return warning


def inference_on_eval_data(tokenizer, model, eval_data, predictions_file, device, instruction_prefix, decode, text_field='text'):
    """Decode ``eval_data[text_field]`` in batches of ``decode.batch_size``, write one stripped hypothesis per
    line to ``predictions_file`` and return them. Prompts are ``format_input_prompt(instruction_prefix, ...)``
    rendered with ``CHAT_TEMPLATE_KWARGS``, exactly as training rendered them."""
    from torch.utils.data import DataLoader
    logging.info(f"inference_on_eval_data: Tokenizer setting originally {tokenizer.padding_side = } {tokenizer.truncation_side = }")
    original_padding_side = tokenizer.padding_side
    original_truncation_side = tokenizer.truncation_side
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    logging.info(f"inference_on_eval_data: Tokenizer setting in inference {tokenizer.padding_side = } {tokenizer.truncation_side = }")

    logging.info(f"inference_on_eval_data: {eval_data}")
    eval_dataloader = DataLoader(eval_data, batch_size=decode.batch_size, shuffle=False)
    all_testout = []
    n_cut, longest = 0, 0
    start_time = time.time()
    with open(predictions_file, "w") as O:
        for i, eval_batch in enumerate(eval_dataloader):

            prompts = [format_input_prompt(instruction_prefix, s) for s in eval_batch[text_field]]
            test_inputs = tokenizer.apply_chat_template(prompts, tokenize=True, add_generation_prompt=True,
                                                        max_length=decode.max_length, truncation=True, padding=True,
                                                        return_tensors="pt", return_dict=True,
                                                        **CHAT_TEMPLATE_KWARGS).to(device)

            boundary = test_inputs["input_ids"].shape[1]
            test_outputs = model.generate(**test_inputs, max_new_tokens=decode.max_new_tokens, num_beams=decode.num_beams, do_sample=decode.do_sample)[:, boundary:]
            test_outputs_detok = tokenizer.batch_decode(test_outputs, skip_special_tokens=True)
            test_outputs_detok_clean = [sent.strip().replace('\n', ' ') for sent in test_outputs_detok]
            for testout in test_outputs_detok_clean:
                O.write(f"{testout}\n")
                all_testout.append(testout)

            # check if prompts get cut off by max_length
            lengths = [len(ids) for ids in tokenizer.apply_chat_template(prompts, tokenize=True, add_generation_prompt=True, return_dict=True,
                                                                         **CHAT_TEMPLATE_KWARGS)["input_ids"]]
            n_cut += sum(n > decode.max_length for n in lengths)
            longest = max(longest, *lengths)

    end_time = time.time()
    logging.info(f"Testing - Elapsed time for {len(eval_data)} sentences in {i+1} batches: {end_time-start_time:.1f}s")
    logging.info(f"Testing - {n_cut} prompts were cut off by max_length={decode.max_length}, longest prompt length={longest}")
    tokenizer.padding_side = original_padding_side
    tokenizer.truncation_side = original_truncation_side
    return all_testout


def decode_test_sets(cfg, tokenizer, model, outdir, device):
    """Decode each ``cfg.data.test.src`` file on its own into ``<outdir>/test<i>.pred.trg`` (i from 1)
    and score it when it has a reference. Writes ``<outdir>/scores.json``
    (``{"test<i>": {"src", "trg", "bleu", "bleu_13a_lowercase", "chrf", "ter"}}``; no scores for a
    set without reference)
    and returns it. Used by both training and inference, so their outputs match."""
    import json
    import os
    from datasets import load_dataset

    test = cfg.data.test
    trgs = list(test.trg) if test.trg is not None else [None] * len(test.src)
    results = {}
    for i, (src, trg) in enumerate(zip(test.src, trgs), start=1):
        name = f"test{i}"
        logging.info(f"Decoding {name}: {src}")
        data = load_dataset("text", data_files=[src], streaming=False, split="train")
        preds = inference_on_eval_data(tokenizer, model, data, os.path.join(outdir, f"{name}.pred.trg"), device,
                                       cfg.instruction, cfg.decode)
        results[name] = {"src": src, "trg": trg}
        if trg is not None:
            refs = load_dataset("text", data_files=[trg], streaming=False, split="train")["text"]
            results[name].update(score(preds, refs))
            logging.info(f"{name} scores: {results[name]}")
    with open(os.path.join(outdir, "scores.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


def score(preds, refs):
    """Corpus BLEU (flores200 tokenization), lowercased BLEU (13a tokenization, exp smoothing:
    sacreBLEU 1.x signature ``BLEU+case.lc+numrefs.1+smooth.exp+tok.13a``), chrF and TER,
    rounded to 2 decimals. Aggregate numbers only: safe to log anywhere (CLAUDE.md)."""
    from sacrebleu.metrics import BLEU, TER, CHRF
    refs = [[r.strip() for r in refs]]
    return {"bleu": round(BLEU(smooth_method="none", max_ngram_order=4, tokenize='flores200').corpus_score(preds, refs).score, 2),
            "bleu_13a_lowercase": round(BLEU(lowercase=True, tokenize='13a', smooth_method="exp").corpus_score(preds, refs).score, 2),
            "chrf": round(CHRF().corpus_score(preds, refs).score, 2),
            "ter": round(TER().corpus_score(preds, refs).score, 2)}
