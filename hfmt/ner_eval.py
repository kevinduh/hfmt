import argparse, json

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

import numpy as np
from tqdm import tqdm

def my_f1(labels: list, preds: list) -> float:
	# F1 = 2TP / (2TP + FP + FN)
	TP = len([word for word in preds if word in labels])
	FP = len([word for word in preds if word not in labels])
	FN = len([word for word in labels if word not in preds])
	return 2 * TP / (TP + TP + FP + FN)

def ner_eval(ref_file: str, src_file: str, hyp_file: str) -> float:

	print(f"Running NER eval with ref_file {ref_file}, src_file {src_file}, and hyp_file {hyp_file}")

	tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
	model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")

	nlp = pipeline("ner", model=model, tokenizer=tokenizer)

	with open(ref_file, 'r') as f:
		ref_data = json.load(f)
		url2doc = {datum['source_url']: datum['text'] for datum in ref_data}

	with open(src_file, 'r') as f:
		src_data = json.load(f)

	with open(hyp_file, 'r') as f:
		hyp_data = json.load(f)
		#summaries = [datum['summary'] for datum in hyp_data]

	assert len(hyp_data) == len(src_data), \
		"hyp_data and src_data must be same len"

	f1_scores = []
	ct_skipped = 0
	for hyp_datum, src_datum in tqdm(zip(hyp_data, src_data)):
		url = src_datum['target_url']
		if url not in url2doc:
			ct_skipped += 1
			continue

		summary = hyp_datum['summary']
		summary_ner_results = nlp(summary)
		summary_nes = [result['word'] for result in summary_ner_results]

		doc = url2doc[url]
		doc_ner_results = nlp(doc)
		doc_nes = [result['word'] for result in doc_ner_results]

		# Now get F1
		datum_f1 = my_f1(labels=doc_nes, preds=summary_nes)
		f1_scores.append(datum_f1)
	
	avg_f1 = np.mean(f1_scores)

	print(f"WARNING: skipped {ct_skipped} instances")

	return avg_f1


if __name__ == "__main__":

	## Set arguments
	parser = argparse.ArgumentParser(description="Train Machine Translation using HuggingFace")
	parser.add_argument("-r", "--ref_file", required=True, help="Reference JSONL file")
	parser.add_argument("-t", "--hyp_file", required=True, help="Hypothesis JSONL file")
	parser.add_argument("-s", "--src_file", required=True, help="Source JSONL file")
	parser.add_argument("-o", "--out_file", default="", help="Output score JSON")
	args = parser.parse_args()
	
	score = ner_eval(
        ref_file=args.ref_file,
		src_file=args.src_file,
        hyp_file=args.hyp_file,
	)

	print(f"NER average F1: {score}")

	""" Example runs:
	- Spanish to English: python3 hfmt/ner_eval.py -r egs/data/CrossSum-test/english-spanish.jsonl -s egs/data/CrossSum-test/spanish-english.jsonl -t egs/models/nllb/es-en_e2e-llama.1/outs/final_outs.jsonl
	- English monolingual: python3 hfmt/ner_eval.py -r egs/data/CrossSum-test/english-english_270.jsonl -s egs/data/CrossSum-test/english-english_270.jsonl -t egs/models/en-en_e2e-llama.1/outs/template_bs1.jsonl
	"""
