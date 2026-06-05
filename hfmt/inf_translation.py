import time
import os
import torch
import numpy as np
import csv
import argparse
import logging
import sys
import wandb
import yaml
from datetime import datetime
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import AutoTokenizer, AutoConfig, DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, Trainer, BitsAndBytesConfig, EarlyStoppingCallback
from torch.utils.data import DataLoader
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
from pprint import pprint, pformat
from sacrebleu.metrics import BLEU, TER, CHRF

os.environ["WANDB_PROJECT"]="hfmt"
experiment_id=""


def format_input_prompt(instruction_prefix, content):
    return [{"content":instruction_prefix + "\n" + content, "role":"user"}]

def inference_on_eval_data(tokenizer, model, eval_data, predictions_file, device, text_field='text'):
    logging.info(f"inference_on_eval_data: Tokenizer setting originally {tokenizer.padding_side = } {tokenizer.truncation_side = }")
    original_padding_side = tokenizer.padding_side
    original_truncation_side = tokenizer.truncation_side
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    logging.info(f"inference_on_eval_data: Tokenizer setting in inference {tokenizer.padding_side = } {tokenizer.truncation_side = }")

    logging.info(f"inference_on_eval_data: {eval_data}")
    eval_dataloader = DataLoader(eval_data, batch_size=16, shuffle=False)
    all_testout = []
    start_time = time.time()
    with open(predictions_file, "w") as O:
        for i, eval_batch in enumerate(eval_dataloader):
            # TODO: extract this into helper function so it can also be called in compute_metrics
            prompts = [format_input_prompt(instruction_prefix, s) for s in eval_batch[text_field]]
            test_inputs = tokenizer.apply_chat_template(prompts, tokenize=True, add_generation_prompt=True, 
                                                        max_length=128, truncation=True, padding=True,
                                                        return_tensors="pt", return_dict=True).to(device)

            boundary = test_inputs["input_ids"].shape[1]
            test_outputs = model.generate(**test_inputs, max_new_tokens=128, do_sample=False)[:, boundary:]
            test_inputs_detok = tokenizer.batch_decode(test_inputs["input_ids"], skip_special_tokens=False)
            test_outputs_detok = tokenizer.batch_decode(test_outputs, skip_special_tokens=True)
            test_outputs_detok_clean = [sent.strip().replace('\n', ' ') for sent in test_outputs_detok]
            for testout in test_outputs_detok_clean:
                O.write(f"{testout}\n")
                all_testout.append(testout)

            # todo: change this to be more configurable
            # if i <= 2:
            #     print_n = 2
            #     torch.set_printoptions(profile="full")
            #     logging.info(f"----- debug eval_batch {i}: (show {print_n} samples) -----")
            #     logging.info(f"{test_inputs['input_ids'].shape = } {test_inputs['input_ids'][:print_n] = }")
            #     logging.info(f"{test_outputs.shape = } {test_outputs[:print_n] = }")
            #     logging.info(f"detokenized input w/ special token: {pformat(test_inputs_detok[:print_n])}")
            #     logging.info(f"detokenized outputs w/ special token: {pformat(tokenizer.batch_decode(test_outputs[:print_n]))}")
            #     torch.set_printoptions(profile="default")

    end_time = time.time()
    logging.info(f"Testing - Elapsed time for {len(eval_data)} sentences in {i+1} batches: {end_time-start_time:.1f}s")
    tokenizer.padding_side = original_padding_side
    tokenizer.truncation_side = original_truncation_side
    return all_testout






def main():



    ###################################
    ## Set arguments
    parser = argparse.ArgumentParser(description="Train Machine Translation using HuggingFace")
    parser.add_argument("-e", "--eval", help="Eval source text")
    parser.add_argument("-c", "--checkpoint", required=True, help="Checkpoint")
    parser.add_argument("-p", "--pretrain", action='store_true', 
                        help="If specified, use Pretrain; Else, Train From Scratch")
    parser.add_argument("-o", "--outdir", required=True, help="Output directory")
    parser.add_argument("-i", "--instruction", type=str, default="", help="Instruction prefix")
    parser.add_argument("--max_steps", type=int, default=10000, help="Max number of train steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--label_smoothing_factor", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()

    logging.basicConfig(filename=os.path.join(args.outdir, "inf.log"), level=logging.INFO, \
        format='%(asctime)s - %(levelname)s - %(message)s', filemode="w")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(args)

    # wandb.init(name=args.outdir, project='hfmt', dir=os.path.join(args.outdir,'wandb'),
    #            config=args)
    
    ###################################
    ## User settings 
    global instruction_prefix
    instruction_prefix = args.instruction
    logging.info(f"instruction: '{instruction_prefix}'")

    global experiment_id
    experiment_id = args.outdir.replace(os.sep,'_').replace('models_','')
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # TODO - check
    import datasets
    datasets.config.HF_DATASETS_OFFLINE = True

    ###################################
    ## Helper functions


    def preprocess_fn(samples):
        prompt = [format_input_prompt(instruction_prefix, s) for s in samples["src"]]
        completion = [[{"content":t, "role":"assistant"}] for t in samples["trg"]]
        return {"prompt": prompt, "completion": completion}
    


    ###################################
    ## Model Configuration
    logging.info(f"======== Model Configuration ========")
    config = AutoConfig.from_pretrained(args.checkpoint)
    
    if args.pretrain == True:
        logging.info("Fine-tuning a pretrained model")
        model = AutoModelForCausalLM.from_pretrained(args.checkpoint).to(device)
    else:
        logging.info("Training from scratch with CausalLM is not supported")
        exit(1)






    ###################################
    ## Inference on Eval set
    logging.info(f"======== Testing ========")
    eval_data = load_dataset("text", data_files=args.eval, streaming=False, split="train")
    #inference_on_eval_data(tokenizer, model, eval_data.select(range(64)), os.path.join(args.outdir,"eval.pred.trg"), device)
    inference_on_eval_data(tokenizer, model, eval_data, os.path.join(args.outdir,"eval.orig_pretrain.trg"), device)


if __name__ == "__main__":
    main()
