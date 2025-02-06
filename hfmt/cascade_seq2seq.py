import time
import os, pdb, re, json
import torch
from torch import OutOfMemoryError
from tqdm import tqdm
import argparse
import logging
import sys
#import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, GenerationConfig
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM

#os.environ["WANDB_PROJECT"]="hfmt"
experiment_id=""

def split_sentences(paragraph, language):
# Regular expression patterns for different languages
    if language == 'arabic':
        sentence_endings = r'(?<=[.؟!])\s+'
    elif language in ['mandarin', 'chinese']:
        sentence_endings = r'(?<=[。？！])\s+'
    elif language == 'tamil':
        sentence_endings = r'(?<=[.؟?ஃ!])\s+'
    else:
        sentence_endings = r'(?<=[.?!])\s+(?=[A-Za-z])' # changed to allow lower-case FIXME
    # Split the paragraph into sentences using the pattern
    sentences = re.split(sentence_endings, paragraph)
    
    return sentences

def split_up_tokens(toks, dim=512):
    # Function to avoid token strings longer than 512
    length = toks['input_ids'].shape[1]
    for idx in range(((length - 1) // dim) + 1):
        new_toks = toks.copy()
        for key in toks:
            new_toks[key] = toks[key][:, idx * dim: (idx + 1) * dim]
        yield new_toks

def flores_eval(checkpoint, flores_code):
    # (1) Set up model
    AutoMod = AutoModelForSeq2SeqLM
    torch_dtype = torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint) 
    model = AutoMod.from_pretrained(
        checkpoint,
        torch_dtype=torch_dtype,
        pad_token_id=tokenizer.eos_token_id
    ).to(device)
    generation_config = GenerationConfig.from_pretrained(checkpoint)
    # (2) Set up data
    eval_data = load_dataset(
		"facebook/flores", 
		flores_code, 
		split="devtest",
		trust_remote_code=True
	)
    input_sents = [eval_datum["sentence"].strip() for eval_datum in eval_data]
    output_sents = [] 
    # (3) Inference
    for orig_sent in tqdm(input_sents): 
        generate_kwargs = {"max_new_tokens": 128, "do_sample": False}
        generate_kwargs["eos_token_id"] = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        input_tokens = tokenizer(orig_sent, return_tensors="pt").to(device)
        test_outputs_raw = model.generate(**input_tokens, **generate_kwargs)
        test_outputs = test_outputs_raw[0]
        test_outputs_detok_raw = tokenizer.decode(test_outputs)
        test_outputs_detok = tokenizer.decode(test_outputs, skip_special_tokens=True)
        output_text = test_outputs_detok.strip()
        output_sents.append(output_text)
    # (4) Get references
    ref_data = load_dataset(
		"facebook/flores", 
		"eng_Latn", 
		split="devtest",
		trust_remote_code=True
	)
    ref_sents = [ref_datum["sentence"].strip() for ref_datum in ref_data]
    return ref_sents, output_sents

def main(
		eval_set, 
		checkpoint, 
		pretrain, 
		summarization, 
		outfile, 
		instruction="", 
		language="english",
		verbose=False
	):

    ###################################

    if eval_set.endswith(".json") or eval_set.endswith(".jsonl"):
        filetype = "json"
    else:
        filetype = "text"
    if outfile.endswith(".json") or outfile.endswith(".jsonl"):
        out_filetype = "json"
    else:
        out_filetype = "text"
    outdir = os.path.split(outfile)[0]

    if verbose:
        logging.basicConfig(filename=os.path.join(outdir, "hfmt.log"), level=logging.INFO, \
        	format='%(asctime)s - %(levelname)s - %(message)s', filemode="w")
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(
			eval_set, 
			checkpoint, 
			pretrain, 
			summarization, 
			outfile, 
			instruction, 
			language
		)

    # wandb.init(name=outdir, project='hfmt', dir=os.path.join(outdir,'wandb'),
    #            config=args)

    if summarization:
        AutoMod = AutoModelForCausalLM
        torch_dtype = torch.bfloat16
    else:
        AutoMod = AutoModelForSeq2SeqLM
        torch_dtype = torch.float32
    
    ###################################
    ## User settings 
    global instruction_prefix
    instruction_prefix = instruction
    if verbose:
        logging.info(f"instruction: '{instruction_prefix}'")

    global experiment_id
    experiment_id = outdir.replace(os.sep,'_').replace('models_','')
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        logging.info(f"Using device: {device}")

    ###################################
    ## Model Configuration
    if verbose:
        logging.info(f"======== Model Configuration ========")
    config = AutoConfig.from_pretrained(checkpoint)
    if pretrain:
        if verbose:
            logging.info("Using a pretrained model")
        model = AutoMod.from_pretrained(
			checkpoint, 
			torch_dtype=torch_dtype,
			pad_token_id=tokenizer.eos_token_id
		).to(device)
    else:
        if verbose:
            logging.info("Training from scratch with pretrained model's config only")
        model = AutoMod.from_config(config).to(device)
	
    generation_config = GenerationConfig.from_pretrained(checkpoint)

    num_param = sum(p.numel() for p in model.parameters())
    if verbose:
        logging.info(f"Number of parameters: {num_param}")
    # for i in model.named_parameters():
    #     logging.info(f"{i[0]} -> {i[1].device}")

    ###################################
    ## Inference on Eval set
    if verbose:
        logging.info(f"======== Testing ========")
    eval_data = load_dataset(filetype, data_files=eval_set, streaming=False, split="train")
    model.eval()
    #eval_data = eval_data.select(range(3))
    start_time = time.time()
    outputs = []
    for i in tqdm(range(len(eval_data))): # komya
        orig_inputs = instruction_prefix + eval_data[i]["text"]
        orig_inputs = orig_inputs.strip()
        generate_kwargs = {"max_new_tokens": 128, "do_sample": False}
        generate_kwargs["eos_token_id"] = [
			tokenizer.eos_token_id,
			tokenizer.convert_tokens_to_ids("<|eot_id|>")
		]
        if summarization:
            generate_kwargs["max_new_tokens"] = 256
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = 0.6
            generate_kwargs["top_p"] = 0.9 
            test_inputs = tokenizer(orig_inputs, return_tensors="pt").to(device)
            try:
                test_outputs_raw = model.generate(**test_inputs, **generate_kwargs)
                test_outputs = test_outputs_raw[0][test_inputs['input_ids'].shape[1]:]
                test_outputs_detok_raw = tokenizer.decode(test_outputs)
                test_outputs_detok = tokenizer.decode(test_outputs, skip_special_tokens=True)
                output_text = test_outputs_detok
            except OutOfMemoryError: # HACK FIXME
                output_text = ""
            outputs.append({"summary": output_text.strip()})
        else:
            input_sents = split_sentences(orig_inputs, language)
            output_text = ""
            for input_sent in input_sents:
                input_tokens = tokenizer(input_sent, return_tensors="pt").to(device)
                for tokens_slice in split_up_tokens(input_tokens):
                    test_outputs_raw = model.generate(**tokens_slice, **generate_kwargs)
                    test_outputs = test_outputs_raw[0]
                    test_outputs_detok_raw = tokenizer.decode(test_outputs)
                    test_outputs_detok = tokenizer.decode(test_outputs, skip_special_tokens=True)
                    output_text += test_outputs_detok.strip() + ' '
            outputs.append({"text": output_text.strip()})
        if i <= 3 and verbose:
            logging.info(f"{i}: {eval_data[i]}")
            logging.info(f"original_inputs: {orig_inputs}")
            logging.info(f"inputs: {test_inputs}")
            logging.info(f"outputs: {test_outputs}")
            logging.info(f"detokenized outputs: {test_outputs_detok_raw}")
    end_time = time.time()
    if out_filetype == "json":
        with open(outfile, "w") as O:
            json.dump(outputs, O, indent=4, ensure_ascii=False)
    elif out_filetype == "text":
        lines_to_write = [output.values()[0] + '\n' for output in outputs]
        with open(outfile, "w") as O:
            O.writelines(lines_to_write)

    if verbose:
        logging.info(f"Testing - Elapsed time for {i} sentences: {end_time-start_time:.1f}s")
    del model

    return

if __name__ == "__main__":
    
    ## Set arguments
    parser = argparse.ArgumentParser(description="Train Machine Translation using HuggingFace")
    parser.add_argument("-e", "--eval", help="Eval source text")
    parser.add_argument("-c", "--checkpoint", required=True, help="Checkpoint")
    parser.add_argument("-p", "--pretrain", action='store_true',
                        help="If specified, use Pretrain; Else, Train From Scratch")
    parser.add_argument("-s", "--summarization", action='store_true',
                        help="If specified, use AutoModelForCausalLM")
    parser.add_argument("-o", "--outfile", required=True, help="Output file")
    parser.add_argument("-i", "--instruction", type=str, default="", help="Instruction prefix")
    parser.add_argument("-l", "--language", type=str, default="english", help="Source text language")

    args = parser.parse_args()

    main(
		eval_set=args.eval, 
		checkpoint=args.checkpoint, 
		pretrain=args.pretrain, 
		summarization=args.summarization, 
		outfile=args.outfile, 
		instruction=args.instruction, 
		language=args.language,
		verbose=True
	)
