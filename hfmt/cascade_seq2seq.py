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

from constants import LANG2FLORES_CODE
from prompts import *

#os.environ["WANDB_PROJECT"]="hfmt"
experiment_id=""

MT_BATCHSIZE = 64
SUMMARIZE_BATCHSIZE = 1 #4

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

def batch_list(l, bs=MT_BATCHSIZE):
    i = 0
    while i < len(l):
        yield l[i: i + bs]
        i += bs

def prep_summarize_tokens1(
		doc_batch, 
		tokenizer, 
		device,
		prefix="",
		max_len=None
	):
    """
	Regular text tokenization like before (bad for instructions)
	"""
    doc_batch = [
	   prefix + sent.strip() for sent in doc_batch
	]
    return tokenizer(
        doc_batch,
        return_tensors="pt",
		padding=True,
		truncation=bool(max_len),
		max_length=max_len
	).to(device)

def prep_summarize_tokens3(
		doc_batch, 
		tokenizer, 
		device,
		prefix="",
		max_len=None
	):
    """
	Adding a suffix (helps a lot, actually)
	"""
    doc_batch = [
	    prefix + sent.strip() for sent in doc_batch
	]
    new_doc_batch = [doc + "\nSummary: " for doc in doc_batch]
    return tokenizer(
        new_doc_batch,
        return_tensors="pt",
                padding=True,
                truncation=bool(max_len),
                max_length=max_len
		).to(device)

def prep_summarize_tokens2(
		doc_batch, 
		tokenizer, 
		device,
		prefix="",
		max_len=None
	):
    """
	Uses a templated input and no attention mask, but otherwise same as 
	prior approach
	"""
    doc_batch = [
	    prefix + sent.strip() for sent in doc_batch
	]
    new_doc_batch = [
        [{'role': 'user', 'content': doc}] for doc in doc_batch
    ]
    input_ids = tokenizer.apply_chat_template(
        new_doc_batch,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
		truncation=bool(max_len),
		max_length=max_len
    ).to(device)
    return {"input_ids": input_ids}

def prep_summarize_tokens4(
        doc_batch,
		tokenizer,
        device,
		prefix=SUMMARIZE_INSTRUCTION,
        max_len=None
    ):
    """
	This is the function for one-shot prompting
    """
    #doc_batch = [
    #        prefix + sent.strip() for sent in doc_batch
    #    ]
    new_doc_batch = [
        [
			{'role': 'system', 'content': prefix.split('\n')[0]},
			{'role': 'system', 'content': "Here is an example."},
			{'role': 'user', 'content': "Passage: " + SHOT1_DOC},
			{'role': 'assistant', 'content': "Summary: " + SHOT1_SUMMARY},
			{'role': 'system', 'content': "Now here is the passage for you to summarize."},
			{'role': 'user', 'content': "Passage: " + doc},
		] for doc in doc_batch
    ]
    input_ids = tokenizer.apply_chat_template(
        new_doc_batch,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
        truncation=bool(max_len),
        max_length=max_len,
		truncation_side='left'
    ).to(device)
    return {"input_ids": input_ids}

PROMPTING_STRATEGIES = [
	prep_summarize_tokens1, 
	prep_summarize_tokens2, 
	prep_summarize_tokens3, 
	prep_summarize_tokens4
]

def run_basic_eval(checkpoint, srcs, language):
    AutoMod = AutoModelForSeq2SeqLM
    torch_dtype = torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint) 
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    if "nllb" in checkpoint.lower():
        tokenizer.src_lang = LANG2FLORES_CODE[language]
        tokenizer.tgt_lang = "eng_Latn"
    model = AutoMod.from_pretrained(
        checkpoint,
        torch_dtype=torch_dtype,
        pad_token_id=tokenizer.eos_token_id
    ).to(device)
    # generation_config = GenerationConfig.from_pretrained(checkpoint)
    output_sents = [] 
    # Inference
    for sent_batch in tqdm(batch_list(srcs)): 
        generate_kwargs = {"max_new_tokens": 128, "do_sample": False}
        generate_kwargs["eos_token_id"] = [
            tokenizer.eos_token_id,
            # tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        if "nllb" in checkpoint.lower():
            generate_kwargs = {
            	"forced_bos_token_id": tokenizer.convert_tokens_to_ids("eng_Latn"),
                "max_length": 128
            }
        input_tokens = tokenizer(
			sent_batch, 
			return_tensors="pt",
			padding=True,
			truncation=True,
			max_length=512,
		).to(device)
        test_outs = model.generate(**input_tokens, **generate_kwargs)
        decoded_sents = [
			tokenizer.decode(
				output,
				skip_special_tokens=True
			).strip() for output in test_outs
		]
        output_sents += decoded_sents
    assert len(output_sents) == len(srcs)
    return output_sents 

def run_flores_eval(checkpoint, flores_code, outs_file):
    # (1) Set up model
    AutoMod = AutoModelForSeq2SeqLM
    torch_dtype = torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint) 
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    if "nllb" in checkpoint.lower():
        tokenizer.src_lang = flores_code
        tokenizer.tgt_lang = "eng_Latn"
    model = AutoMod.from_pretrained(
        checkpoint,
        torch_dtype=torch_dtype,
        pad_token_id=tokenizer.eos_token_id
    ).to(device)
    # generation_config = GenerationConfig.from_pretrained(checkpoint)
    if not os.path.exists(outs_file):
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
        for sent_batch in tqdm(batch_list(input_sents)): 
            generate_kwargs = {"max_new_tokens": 128, "do_sample": False}
            generate_kwargs["eos_token_id"] = [
                tokenizer.eos_token_id,
                # tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            if "nllb" in checkpoint.lower():
                generate_kwargs = {
					"forced_bos_token_id": tokenizer.convert_tokens_to_ids("eng_Latn"),
					"max_length": 128
				}
            input_tokens = tokenizer(
				sent_batch, 
				return_tensors="pt",
				padding=True,
				truncation=True,
				max_length=512,
			).to(device)
            test_outs = model.generate(**input_tokens, **generate_kwargs)
            decoded_sents = [
				tokenizer.decode(
					output,
					skip_special_tokens=True
				).strip() for output in test_outs
			]
            output_sents += decoded_sents
        # (3.5) Write outputs for later 
        with open(outs_file, 'w') as f:
            json.dump(output_sents, f, indent=4, ensure_ascii=False)
    else:
        # (2-3) Just read the inference sentences 
        with open(outs_file, 'r') as f:
            output_sents = json.load(f)
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
		eval_set: str, 
		checkpoint: str, 
		pretrain: bool, 
		summarization: bool, 
		outfile: str, 
		instruction: str="", 
		language: str="english",
		verbose: bool=False,
		mt_model_dim: int=512,
		prompting_strategy=1 # corresponding to prep_summarize_tokens2 
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
        prep_summarize_tokens = PROMPTING_STRATEGIES[prompting_strategy]
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
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.bos_token if summarization else tokenizer.eos_token
    if summarization:
        tokenizer.padding_side = 'left'
    if "nllb" in checkpoint.lower():
        tokenizer.src_lang = LANG2FLORES_CODE[language]
        tokenizer.tgt_lang = "eng_Latn"
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
	
    # generation_config = GenerationConfig.from_pretrained(checkpoint)

    num_param = sum(p.numel() for p in model.parameters())
    if verbose:
        logging.info(f"Number of parameters: {num_param}")

    ###################################
    ## Inference on Eval set
    if verbose:
        logging.info(f"======== Testing ========")
    eval_data = load_dataset(filetype, data_files=eval_set, streaming=False, split="train")
    model.eval()
    #eval_data = eval_data.select(range(3))
    start_time = time.time()
    outputs = []
    valid_input_sizes = [1]
    input_texts = [eval_datum["text"] for eval_datum in eval_data]
    print(input_texts, flush=True)
    for doc_batch in tqdm(batch_list(input_texts, bs=SUMMARIZE_BATCHSIZE)): # komya
        #doc_batch = [ # MOVED THIS TO prep_summarize_tokens
		#	instruction_prefix + sent.strip() for sent in doc_batch
		#]
        generate_kwargs = {"max_new_tokens": 128, "do_sample": False}
        generate_kwargs["eos_token_id"] = [
			tokenizer.eos_token_id,
			# tokenizer.convert_tokens_to_ids("<|eot_id|>")
		]
        if summarization:
            # generate_kwargs["max_new_tokens"] = 256
            # generate_kwargs["do_sample"] = False # True
            # generate_kwargs["temperature"] =  0.6
            # generate_kwargs["top_p"] = 0.9 
            test_inputs = prep_summarize_tokens(
				doc_batch, 
				tokenizer, 
				device, 
				prefix=instruction_prefix
			)
            test_input_size = test_inputs['input_ids'].shape[1]
            try:
                test_outs = model.generate(**test_inputs, **generate_kwargs)
                valid_input_sizes.append(test_input_size)
            except OutOfMemoryError: # Now just reduce size of tensor for inputs 
                print("WARNING: OutOfMemoryError")
                valid_input_size = max(valid_input_sizes)
                test_inputs = prep_summarize_tokens(
					doc_batch, 
					tokenizer, 
					device,
					prefix=instruction_prefix,
					max_len=valid_input_size
				)
                test_outs = model.generate(**test_inputs, **generate_kwargs)
                #for key in test_inputs:
                #    test_inputs[key] = test_inputs[key][:, :valid_input_size]
            # Truncate outs
            decode_outs = test_outs[:, test_input_size:]
            decoded_sents = [
                tokenizer.decode(
        	        output,
    	            skip_special_tokens=True
	            ).strip() for output in decode_outs
	        ]
            #print("~~" * 10)
            #print("INPUT:")
            #print(tokenizer.decode(test_inputs['input_ids']))
            #print("OUTPUT:")
            #print(decoded_sents[0])
            #raise NotImplementedError
            outputs += [
				{
					"summary": output_text.strip()
				} for output_text in decoded_sents
			]
        else:
            if pretrain and "nllb" in checkpoint:
                generate_kwargs = {
					"forced_bos_token_id": tokenizer.convert_tokens_to_ids("eng_Latn"),
					"max_length": 128
				}
            output_translations = []
            for input_doc in doc_batch:
                input_sents = split_sentences(input_doc, language)
                output_text = ""
                for sent_batch in batch_list(input_sents):
                    input_tokens = tokenizer(
						sent_batch,
						return_tensors="pt",
						padding=True,
						truncation=True,
						max_length=mt_model_dim,
					).to(device)
                    test_outs = model.generate(**input_tokens, **generate_kwargs)
                    decoded_sents = [
						tokenizer.decode(
							output,
							skip_special_tokens=True
						).strip() for output in test_outs
					]
                    output_text += ' '.join(decoded_sents) + ' '
                output_translations.append(output_text)
            outputs += [
				{
					"text": output_text.strip()
				} for output_text in output_translations
			]
        if verbose:
            # logging.info(f"{i}: {eval_data[i]}")
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
    parser.add_argument("--mt_model_dim", type=int, default=512)

    args = parser.parse_args()

    main(
		eval_set=args.eval, 
		checkpoint=args.checkpoint, 
		pretrain=args.pretrain, 
		summarization=args.summarization, 
		outfile=args.outfile, 
		instruction=args.instruction, 
		language=args.language,
		verbose=True,
		mt_model_dim=args.mt_model_dim
	)
