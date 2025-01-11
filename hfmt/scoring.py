import os, pdb, json

#os.environ["WANDB_PROJECT"]="hfmt"

def get_summaries(jsonl_file):
	with open(jsonl_file, 'r') as f:
		data = json.load(f)
	return [datum['summary'] for datum in data]

def main():

	###################################
	## Set arguments
	parser = argparse.ArgumentParser(description="Train Machine Translation using HuggingFace")
	parser.add_argument("-r", "--ref_file", required=True, help="Reference JSONL file")
	parser.add_argument("-h", "--hyp_file", required=True, help="Hypothesis JSONL file")	
	parser.add_argument("-o", "--out_file", default="", help="Output score JSON")
	args = parser.parse_args()

	rouge_score = get_rouge_score(args.ref_file, args.hyp_file)
	print(f"Calculated rouge for {args.hyp_file}: {rouge_score}")

	# Write to out file
	if args.out_file:
		with open(args.out_file, 'w') as f:
			json.dump({"rouge": rouge_score})
		print("Check", args.out_file)

def get_rouge_score(ref_file, hyp_file, rouge_type="rougeL")

	refs = get_summaries(ref_file)
	hyps = get_summaries(hyp_file)

	assert type(refs) == list and type(hyps) == list 
	assert type(refs[0]) == str and type(hyps[0]) == str

	# Compute ROUGE

	rouge = load("rouge")
	results = rouge.compute(predictions=hyps, references=refs)
	rouge_score = results[rouge_type]

	return rouge_score

if __name__ == "__main__":
	main()
