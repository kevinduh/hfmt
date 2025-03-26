from transformers import pipeline
from datasets import load_dataset
from tqdm.auto import tqdm
import evaluate
import torch
import sys
import os
import logging
import time
import argparse

parser = argparse.ArgumentParser(description="Summarization Decoding")
parser.add_argument("-e", "--eval", help="Eval JSONL")
parser.add_argument("-c", "--checkpoint", required=True, help="Checkpoint")
parser.add_argument("--prompt_choice", required=True, default='1', help="Choice of prompt (see code)")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("-o", "--outprefix", required=True, help="Output file prefix")
args = parser.parse_args()


char_limit=10000

sysprompts={}
sysprompts['1'] = "You are a helpful summarization agent. Please summarize the article below in less than 20 words"
sysprompts['2'] = "Please summarize the following news article in less than 20 words"
sysprompts['3'] = "You are a newspaper editor. Your job is to provide a sentence heading for the following article. Please use less than 20 words."
sysprompts['4'] = """Please summarize the following news article in less than 20 words. Below is an example.

Example article: The bus overturned and was sent hurtling down the road for about 25 metres. The local hospital says fifteen children were among those killed. The government blamed the Tamil Tiger rebels and called it a barbaric terrorist act. It's the worst attack since a ceasefire was signed in 2002. Escalating violence has left hundreds dead since the beginning of the year. A suicide bombing in Colombo in April led to air strikes. The government says the Tigers are trying to provoke a Sinhalese backlash. Sri Lanka is already in an undeclared war. Now hopes for peace must be even further away. The Tamil Tigers have denied and condemned the attack. The ancient holy city of Anuradhapura suffered an attack allegedly by the Tamil Tigers in May 1985 in which About 250 men, women and children were gunned down at the central bus stand at the Anuradhapura town. The Tigers never denied resposibility for the attack.

Example summary: The attack happened in the early morning when the bus was packed with villagers travelling to work and school. It was hit by a mine explosion.

"""

logging.basicConfig(filename=args.outprefix+".log", level=logging.INFO, \
                    format='%(asctime)s - %(levelname)s - %(message)s', filemode="w")
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

test_data = load_dataset("json", data_files=args.eval)['train']
logging.info(f"{sys.argv}")
logging.info(test_data)

pipe = pipeline(
    "text-generation",
    model=args.checkpoint,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

logging.info(f"Loaded {args.checkpoint}")
logging.info(f"Tokenizer pad token ID: {pipe.tokenizer.pad_token_id}")
logging.info(f"Model max_position_embedding: {pipe.model.config.max_position_embeddings}")

if args.batch_size != 1:
    # Code for batch_size>1 uses own preparation, so set up here; code for batch_size=1 just use default
    pipe.tokenizer.padding_side = 'left'
    pipe.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    pipe.model.resize_token_embeddings(len(pipe.tokenizer))
    pipe.model.generation_config.pad_token_id = pipe.tokenizer.pad_token_id
    input_max_length=4096
    logging.info(f"Fixing input max length for generation at: {input_max_length} and character limit at: {char_limit}")

    # Static compilation might run faster: https://huggingface.co/docs/transformers/llm_optims
    pipe.model.generation_config.cache_implementation = "static" 
    pipe.model.forward = torch.compile(pipe.model.forward, mode="reduce-overhead", fullgraph=True)



def pipedata(mydataset, batch_size):
    num_batch = -1 * (-1* len(mydataset) // batch_size)
    for i in range(num_batch):
        j = min(len(mydataset), i*batch_size+batch_size)
        yield mydataset[i*batch_size:j]

rouge = evaluate.load('rouge', experiment_id=args.outprefix.replace(os.sep,'_'))



start_time = time.time()
with open(args.outprefix+".summary",'w') as OUT:
    results=[]
    for j, samples in enumerate(tqdm(pipedata(test_data, args.batch_size))):

        prompts = []
        for txt in samples['text']:
            messages = [
                {"role": "system", "content": sysprompts[args.prompt_choice]},
                {"role": "user", "content": txt[:char_limit]},
            ]
            prompts.append(messages)

        if args.batch_size == 1:
            outputs = pipe(messages,
                max_new_tokens=256,
                do_sample=False,
                temperature=1.0,
                top_p=1,
            )
            summary_hyp = outputs[0]["generated_text"][-1]['content']
            OUT.write(f"batch:{j}\t{summary_hyp}\n")
            results.append(summary_hyp)

        else:
            inputs = pipe.tokenizer.apply_chat_template(prompts, 
                                                        tokenize=True,
                                                        add_generation_prompt=True,
                                                        padding='max_length', 
                                                        max_length=input_max_length, 
                                                        truncation=True,
                                                        return_dict=True,
                                                        return_tensors='pt').to(pipe.model.device)

            outputs = pipe.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=1.0,
                top_p=1,
                pad_token_id=pipe.tokenizer.pad_token_id
            )

            outputsDec = pipe.tokenizer.batch_decode(outputs[:,len(inputs['input_ids'][0]):],
                                                    skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=True)

            for k in range(outputs.size(0)):
                summary_hyp = outputsDec[k]
                OUT.write(f"batch:{j}\t{summary_hyp}\n")
                results.append(summary_hyp)

    overall_score = rouge.compute(predictions=results, references=test_data['summary'], use_aggregator=True, rouge_types=['rouge1','rouge2','rougeL'])
    logging.info(f"ROUGE: {overall_score}")

end_time = time.time()
logging.info(f"Inference - Elapsed time: {end_time-start_time:.1f}s")

