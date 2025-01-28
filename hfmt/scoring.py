import os, pdb, json, argparse
import evaluate

#os.environ["WANDB_PROJECT"]="hfmt"

def get_texts(jsonl_file, key='summary'):
	with open(jsonl_file, 'r') as f:
		data = json.load(f)
	return [datum[key] for datum in data]

def main(ref_file, hyp_file, out_file, mt_out_file=""):

	###################################

	if out_file and os.path.exists(out_file):
		with open(out_file, 'r') as f:
			score_dict = json.load(f)
	else:
		score_dict = {}

	if "rouge" not in score_dict:
		rouge_score = get_score(
			ref_file, 
			hyp_file, 
			metric="rouge", 
			submetric="rougeL"
		)
		score_dict["rouge"] = rouge_score
		print(f"Calculated rouge for {hyp_file}: {rouge_score}")

	if "bleu" not in score_dict and mt_out_file:
		bleu_score = get_score(
			ref_file, 
			mt_out_file,
			metric="bleu",
			submetric="bleu"
		)
		score_dict["bleu"] = bleu_score

	# Write to out file
	if out_file:
		with open(out_file, 'w') as f:
			json.dump(score_dict, f, indent=4)
		print("Check", out_file)
	
	return score_dict

def get_score(ref_file, hyp_file, metric="rouge", submetric="rougeL"):

	key = "text" if metric == "bleu" else "summary"

	refs = get_texts(ref_file, key)
	hyps = get_texts(hyp_file, key)

	assert type(refs) == list and type(hyps) == list 
	assert type(refs[0]) == str and type(hyps[0]) == str
	#if metric == "bleu":
	#	refs = [refs]

	# Compute ROUGE

	scorer = evaluate.load(metric)
	results = scorer.compute(predictions=hyps, references=refs)
	metric_score = results[submetric]

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
