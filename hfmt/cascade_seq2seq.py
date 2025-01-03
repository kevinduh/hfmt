import time
import os, pdb
import torch
import argparse
import logging
import sys
#import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, GenerationConfig
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM

#os.environ["WANDB_PROJECT"]="hfmt"
experiment_id=""

def main():

    ###################################
    ## Set arguments
    parser = argparse.ArgumentParser(description="Train Machine Translation using HuggingFace")
    parser.add_argument("-e", "--eval", help="Eval source text")
    parser.add_argument("-c", "--checkpoint", required=True, help="Checkpoint")
    parser.add_argument("-p", "--pretrain", action='store_true', 
                        help="If specified, use Pretrain; Else, Train From Scratch")
    parser.add_argument("-s", "--summarization", action='store_true',
                        help="If specified, use AutoModelForCausalLM")
    parser.add_argument("-o", "--outdir", required=True, help="Output directory")
    parser.add_argument("-i", "--instruction", type=str, default="", help="Instruction prefix")
    
    args = parser.parse_args()

    if args.eval.endswith(".json") or args.eval.endswith(".jsonl"):
        args.filetype = "json"
    else:
        args.filetype = "text"

    logging.basicConfig(filename=os.path.join(args.outdir, "hfmt.log"), level=logging.INFO, \
        format='%(asctime)s - %(levelname)s - %(message)s', filemode="w")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(args)

    # wandb.init(name=args.outdir, project='hfmt', dir=os.path.join(args.outdir,'wandb'),
    #            config=args)

    if args.summarization:
        AutoMod = AutoModelForCausalLM
        torch_dtype = torch.bfloat16
    else:
        AutoMod = AutoModelForSeq2SeqLM
        torch_dtype = torch.float32
    
    ###################################
    ## User settings 
    global instruction_prefix
    instruction_prefix = args.instruction
    logging.info(f"instruction: '{instruction_prefix}'")

    global experiment_id
    experiment_id = args.outdir.replace(os.sep,'_').replace('models_','')
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    ###################################
    ## Model Configuration
    logging.info(f"======== Model Configuration ========")
    config = AutoConfig.from_pretrained(args.checkpoint)
    if args.pretrain == True:
        logging.info("Fine-tuning a pretrained model")
        model = AutoMod.from_pretrained(args.checkpoint, torch_dtype=torch_dtype).to(device)
    else:
        logging.info("Training from scratch with pretrained model's config only")
        model = AutoMod.from_config(config).to(device)
	
    generation_config = GenerationConfig.from_pretrained(args.checkpoint)

    num_param = sum(p.numel() for p in model.parameters())
    logging.info(f"Number of parameters: {num_param}")
    # for i in model.named_parameters():
    #     logging.info(f"{i[0]} -> {i[1].device}")

    ###################################
    ## Inference on Eval set
    logging.info(f"======== Testing ========")
    eval_data = load_dataset(args.filetype, data_files=args.eval, streaming=False, split="train")
    model.eval()
	#eval_data = eval_data.select(range(3))
    start_time = time.time()
    if True:#with open(os.path.join(args.outdir,"eval.pred.trg"), "w") as O:
        for i in range(len(eval_data)):
            orig_inputs = instruction_prefix + " " + eval_data[i]["summary"]
            orig_inputs = orig_inputs.strip()
            generate_kwargs = {"max_new_tokens": 128, "do_sample": False}
            if args.summarization:
                generate_kwargs["max_new_tokens"] = 256
                generate_kwargs["do_sample"] = True
                generate_kwargs["temperature"] = 0.6
                generate_kwargs["top_p"] = 0.9 
                generate_kwargs["eos_token_id"] = [
					tokenizer.eos_token_id,
					tokenizer.convert_tokens_to_ids("<|eot_id|>")
				]
                test_inputs = tokenizer(orig_inputs, return_tensors="pt").to(device)
                test_outputs_raw = model.generate(**test_inputs, **generate_kwargs)
                test_outputs = test_outputs_raw[0]
                test_outputs = test_outputs_raw[0][test_inputs['input_ids'].shape[1]:]
                test_outputs_detok_raw = tokenizer.decode(test_outputs)
                test_outputs_detok = tokenizer.decode(test_outputs, skip_special_tokens=True)
                output_text = test_outputs_detok
                outputs.append({"text": output_text.strip()})
            else:
                input_sents = split_sentences(orig_inputs)
                output_text = ""
                for input_sent in input_sents:
                    test_inputs = tokenizer(input_sent, return_tensors="pt").to(device)
                    test_outputs_raw = model.generate(**test_inputs, **generate_kwargs)
                    test_outputs = test_outputs_raw[0]
                    test_outputs_detok_raw = tokenizer.decode(test_outputs)
                    test_outputs_detok = tokenizer.decode(test_outputs, skip_special_tokens=True)
                    output_text += test_outputs_detok.strip() + ' '
                outputs.append({"summary": output_text.strip()})
            if i <= 3:
                logging.info(f"{i}: {eval_data[i]}")
                logging.info(f"original_inputs: {orig_inputs}")
                logging.info(f"inputs: {test_inputs}")
                logging.info(f"outputs: {test_outputs}")
                logging.info(f"detokenized outputs: {test_outputs_detok_raw}")
    end_time = time.time()
    if args.out_filetype == "json":
        with open(args.outfile, "w") as O:
            json.dump(outputs, O)
    elif args.out_filetype == "text":
        lines_to_write = [output.values()[0] + '\n' for output in outputs]
        with open(args.outfile, "w") as O:
            O.writelines(lines_to_write)

    logging.info(f"Testing - Elapsed time for {i} sentences: {end_time-start_time:.1f}s")

if __name__ == "__main__":
    main()
