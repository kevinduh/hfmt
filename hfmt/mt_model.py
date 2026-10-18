"""Model loading, prompting, decoding and scoring shared by SFT and inference (feature 09).

Training (``sft_translation.py``) and inference (``inf_translation.py``) load the base model the
same way for each ``peft.mode``, so a trained adapter is always decoded on the base it was
trained on (e.g. 4-bit only for ``qlora``).

Heavy imports (torch, transformers, ...) live inside functions so config-only use stays fast.
"""

import time
import logging


def format_input_prompt(instruction_prefix, content):
    return [{"content":instruction_prefix + "\n" + content, "role":"user"}]


def load_tokenizer(checkpoint):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(cfg, device):
    """Load ``cfg.checkpoint`` for ``cfg.peft.mode``: fp32 for ``full``, bf16/fp16 for ``lora``,
    4-bit for ``qlora``. Returns ``(model, use_bf16)``."""
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    finetune_mode = cfg.peft.mode  # one of: full | lora | qlora
    use_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
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


def inference_on_eval_data(tokenizer, model, eval_data, predictions_file, device, instruction_prefix, decode, text_field='text'):
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
                                                        return_tensors="pt", return_dict=True).to(device)

            boundary = test_inputs["input_ids"].shape[1]
            test_outputs = model.generate(**test_inputs, max_new_tokens=decode.max_new_tokens, num_beams=decode.num_beams, do_sample=decode.do_sample)[:, boundary:]
            test_outputs_detok = tokenizer.batch_decode(test_outputs, skip_special_tokens=True)
            test_outputs_detok_clean = [sent.strip().replace('\n', ' ') for sent in test_outputs_detok]
            for testout in test_outputs_detok_clean:
                O.write(f"{testout}\n")
                all_testout.append(testout)

            # check if prompts get cut off by max_length
            lengths = [len(ids) for ids in tokenizer.apply_chat_template(prompts, tokenize=True, add_generation_prompt=True, return_dict=True)["input_ids"]]
            n_cut += sum(n > decode.max_length for n in lengths)
            longest = max(longest, *lengths)

    end_time = time.time()
    logging.info(f"Testing - Elapsed time for {len(eval_data)} sentences in {i+1} batches: {end_time-start_time:.1f}s")
    logging.info(f"Testing - {n_cut} prompts were cut off by max_length={decode.max_length}, longest prompt length={longest}")
    tokenizer.padding_side = original_padding_side
    tokenizer.truncation_side = original_truncation_side
    return all_testout


def score(preds, refs):
    """Corpus BLEU (flores200 tokenization), chrF and TER, rounded to 2 decimals.
    Aggregate numbers only: safe to log anywhere (CLAUDE.md)."""
    from sacrebleu.metrics import BLEU, TER, CHRF
    refs = [[r.strip() for r in refs]]
    return {"bleu": round(BLEU(smooth_method="none", max_ngram_order=4, tokenize='flores200').corpus_score(preds, refs).score, 2),
            "chrf": round(CHRF().corpus_score(preds, refs).score, 2),
            "ter": round(TER().corpus_score(preds, refs).score, 2)}
