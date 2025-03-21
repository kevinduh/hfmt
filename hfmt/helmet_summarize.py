import os
from pprint import pprint

import cascade_seq2seq as cascade
import scoring
from prompts import *

ENG_XSUM_JSON = "egs/data/Multi-LexSum-test/english-english.jsonl"
HF_SUMMARIZE_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
ENG_OUTDIR = "egs/models/en-en_e2e-llama.1/outs"

def sanity_check(
		outdir=ENG_OUTDIR, 
		run_name="final_outs", 
		ps=0, 
		summarize_instruction=HELMET_INSTRUCTION#SUMMARIZE_INSTRUCTION
	):

	if not os.path.exists(outdir):
		os.makedirs(outdir)
	outfile = os.path.join(outdir, f"{run_name}.jsonl")

	# Summarization step
	cascade.main(
		eval_set=ENG_XSUM_JSON,
		checkpoint=HF_SUMMARIZE_MODEL,
		pretrain=True,
		summarization=True,
		outfile=outfile,
		instruction=summarize_instruction,
		prompting_strategy=ps
	)

	# Scoring
	scores_json = os.path.join(outdir, f"{run_name}_score.json")
	score = scoring.main(
		ref_file=ENG_XSUM_JSON,
		hyp_file=outfile,
		out_file=scores_json,
	)

	print("done with sanity check")
	return score

if __name__ == "__main__":

	score_dict = sanity_check(run_name="helmet", ps=4)
	print("\\" * 10, f"SCORE for helmet")
	pprint(score_dict)
	#for idx, name in enumerate(["text_only", "template_bs1", "with_suffix"]):
	#	score_dict = sanity_check(run_name=name, ps=idx)
	#	print("\\" * 10, f"SCORE for {name}")
	#	pprint(score_dict)
