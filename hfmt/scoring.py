import os, pdb, json, argparse
import evaluate

import qe

#os.environ["WANDB_PROJECT"]="hfmt"

METRIC2KEY = {
	"rouge": "summary", 
	"comet": "text"
}

def get_texts(data_file, key='summary'):
	if key is None: # In this case assume txt file, not jsonl
		with open(data_file, 'r') as f:
			lines = f.readlines() 
		return [line.strip() for line in lines]
	with open(data_file, 'r') as f:
		data = json.load(f)
	return [datum[key] for datum in data]

def get_multi_refs(srcs, refs):
	src2refs = {}
	for src, ref in zip(srcs, refs):
		if src not in src2refs:
			src2refs[src] = []
		src2refs[src].append(ref)
	return [src2refs[src] for src in srcs]

def main(
		ref_file, 
		hyp_file, 
		flores_ref_file="", 
		flores_hyp_file="", 
		out_file="", 
		mt_out_file=""
	):

	###################################

	if out_file and os.path.exists(out_file):
		with open(out_file, 'r') as f:
			score_dict = json.load(f)
	else:
		score_dict = {}

	if "rouge2" not in score_dict:
		rouge_score = get_score(
			ref_file, 
			hyp_file, 
			metric="rouge", 
			submetric="rougeL"
		)
		score_dict["rougeL"] = rouge_score
		print(f"Calculated rougeL for {hyp_file}: {rouge_score}")
		
		rouge_score = get_score(
                        ref_file,
                        hyp_file,
                        metric="rouge",
                        submetric="rouge2"
                )
		score_dict["rouge2"] = rouge_score
		print(f"Calculated rouge2 for {hyp_file}: {rouge_score}")


	if "bleu" not in score_dict and mt_out_file and flores_ref_file\
			and flores_hyp_file:
		bleu_score = get_score(
			flores_ref_file, 
			flores_hyp_file,
			metric="bleu",
			submetric="bleu"
		)
		score_dict["bleu"] = bleu_score
	
	if "comet_qe" not in score_dict and mt_out_file:
		comet_qe = get_score(
			ref_file,
			mt_out_file,
			metric="comet",
			submetric="qe"
		)
		score_dict["comet_qe"] = comet_qe

	# Write to out file
	if out_file:
		with open(out_file, 'w') as f:
			json.dump(score_dict, f, indent=4)
		print("Check", out_file)
	
	return score_dict

def get_score(ref_, hyp_, src_=None, metric="rouge", submetric="rougeL"):

	key = METRIC2KEY.get(metric, None)

	if type(ref_) == list:
		assert type(hyp_) == list, "Both must be lists or both str's"
		refs = ref_
		hyps = hyp_
		srcs = src_
	elif type(ref_) == str:
		assert type(hyp_) == str, "Both must be lists or both str's"
		refs = get_texts(ref_, key)
		hyps = get_texts(hyp_, key)
		if src_:
			srcs = get_texts(src_, key)
		else:
			srcs = None
	else:
		raise TypeError("ref_ and hyp_ must be lists or str's")

	if submetric == "qe":
		return qe.get_qe_score(refs=refs, hyps=hyps)

	assert type(refs) == list and type(hyps) == list 
	assert type(refs[0]) == str and type(hyps[0]) == str
	#if metric == "bleu":
	#	refs = [refs]

	load_kwargs = {} 
	compute_kwargs = {}
	if metric == "chrf":
		if submetric == "chrf++":
			load_kwargs["word_order"] = 2
	if metric == "bleu":
		assert submetric.startswith("bleu")
		if len(submetric) == 5:
			max_order = int(submetric[4])
			compute_kwargs['max_order'] = max_order
		submetric = "bleu"

	# Compute ROUGE or BLEU

	scorer = evaluate.load(metric, **load_kwargs)
	if srcs and metric == "bleu":
		refs = get_multi_refs(srcs, refs)
	results = scorer.compute(predictions=hyps, references=refs, **compute_kwargs)
	metric_key = "score" if metric == "chrf" else submetric
	divisor = 100. if metric == "chrf" else 1.
	metric_score = results[metric_key] / divisor

	return metric_score

if __name__ == "__main__":

	## Set arguments
	parser = argparse.ArgumentParser(description="Train Machine Translation using HuggingFace")
	parser.add_argument("-r", "--ref_file", required=True, help="Reference JSONL file")
	parser.add_argument("-t", "--hyp_file", required=True, help="Hypothesis JSONL file")
	parser.add_argument("-o", "--out_file", default="", help="Output score JSON")
	parser.add_argument("-m", "--mt_out_file", default="", help="MT out file for BLEU")
	args = parser.parse_args()

	main(
		ref_file=args.ref_file, 
		hyp_file=args.hyp_file, 
		out_file=args.out_file, 
		mt_out_file=args.mt_out_file
	)
