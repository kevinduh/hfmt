import time
import os
import torch
import evaluate
import numpy as np
import csv
import argparse
import logging
import sys
#import wandb
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, DataCollatorForSeq2Seq, GenerationConfig
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

#os.environ["WANDB_PROJECT"]="hfmt"
experiment_id=""

def main():



    ###################################
    ## Set arguments
    parser = argparse.ArgumentParser(description="Train Machine Translation using HuggingFace")
    parser.add_argument("-t", "--train", required=True, help="Train bitext")
    parser.add_argument("-d", "--dev", required=True, help="Dev bitext")
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

    logging.basicConfig(filename=os.path.join(args.outdir, "hfmt.log"), level=logging.INFO, \
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
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    ###################################
    ## Helper functions
    def preprocess_fn(samples):
        inputs = [instruction_prefix + " " + s for s in samples["src"]]
        targets = [s for s in samples["trg"]]
        model_inputs = tokenizer(inputs, text_target=targets,
                                max_length=100,truncation=True,padding='max_length')
        return model_inputs


    def get_data(train_path, dev_path, tokenizer):
        custom_splits = {"train": train_path, "dev": dev_path}
        data = load_dataset("csv", delimiter = "\t", column_names=['src','trg'], 
                            data_files = custom_splits, streaming=True,
                            quoting=csv.QUOTE_NONE)
        data = data.map(preprocess_fn, batched=True)
        return data


    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels


    metric1 = evaluate.load("sacrebleu", experiment_id=experiment_id+"_sacrebleu")
    metric2 = evaluate.load("chrf", experiment_id=experiment_id+"_chrf")
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        r1 = metric1.compute(predictions=decoded_preds, references=decoded_labels)
        r2 = metric2.compute(predictions=decoded_preds, references=decoded_labels)
        result = {
            "bleu": round(r1["score"],2), 
            "chrf": round(r2["score"],2),
            "precisions": [round(p,1) for p in r1["precisions"]],
            "bp": round(r1["bp"],2),
            "ref_len": r1["ref_len"], 
            "sys_len": r1["sys_len"]
            }
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["sys_len_avg"] = round(np.mean(prediction_lens),2)
        return result


    ###################################
    ## Load data
    logging.info(f"======== Loading data ========")
    start_time = time.time()
    D = get_data(args.train, args.dev, tokenizer)
    logging.info(D)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=args.checkpoint)
    end_time = time.time()
    logging.info(f"Loading data - Elapsed time: {end_time-start_time:.1f}s")



    ###################################
    ## Model Configuration
    logging.info(f"======== Model Configuration ========")
    config = AutoConfig.from_pretrained(args.checkpoint)
    if args.pretrain == True:
        logging.info("Fine-tuning a pretrained model")
        model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint).to(device)
    else:
        logging.info("Training from scratch with pretrained model's config only")
        model = AutoModelForSeq2SeqLM.from_config(config).to(device)

    generation_config = GenerationConfig.from_pretrained(args.checkpoint)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.outdir,
        eval_strategy="steps",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        save_total_limit=3,
        max_steps=args.max_steps,
        predict_with_generate=True,
        fp16=False,
        push_to_hub=False,
        report_to="none",#"wandb",
        run_name=args.outdir.replace(os.sep,'_').replace('models_',''),
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        generation_config=generation_config,
        overwrite_output_dir=True,
        seed=args.seed,
        label_smoothing_factor=args.label_smoothing_factor,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps
    )
    logging.info(training_args)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=D["train"],
        eval_dataset=D["dev"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    num_param = sum(p.numel() for p in model.parameters())
    logging.info(f"Number of parameters: {num_param}")
    # for i in model.named_parameters():
    #     logging.info(f"{i[0]} -> {i[1].device}")
        

    ###################################
    ## Training
    logging.info(f"======== Training ========")
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    logging.info(f"Training - Elapsed time: {end_time-start_time:.1f}s")

    ###################################
    ## Inference on Eval set
    logging.info(f"======== Testing ========")
    eval_data = load_dataset("text", data_files=args.eval, streaming=False, split="train")
    #eval_data = eval_data.select(range(3))
    start_time = time.time()
    with open(os.path.join(args.outdir,"eval.pred.trg"), "w") as O:
        for i in range(len(eval_data)):
            orig_inputs = instruction_prefix + " " + eval_data[i]["text"]
            test_inputs = tokenizer(orig_inputs, return_tensors="pt").to(device)
            test_outputs = model.generate(**test_inputs, max_new_tokens=128, do_sample=False)
            test_outputs_detok = tokenizer.decode(test_outputs[0])
            test_outputs_detok2 = tokenizer.decode(test_outputs[0], skip_special_tokens=True)
            
            O.write(test_outputs_detok2 + "\n")
            if i <= 3:
                logging.info(f"{i}: {eval_data[i]}")
                logging.info(f"original_inputs: {orig_inputs}")
                logging.info(f"inputs: {test_inputs}")
                logging.info(f"outputs: {test_outputs}")
                logging.info(f"detokenized outputs: {test_outputs_detok}")
    end_time = time.time()
    logging.info(f"Testing - Elapsed time for {i} sentences: {end_time-start_time:.1f}s")

if __name__ == "__main__":
    main()
