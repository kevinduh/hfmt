import os
import cascade_seq2seq as cascade
import scoring

ENG_XSUM_JSON = "egs/data/CrossSum-test/english-english.jsonl"
HF_SUMMARIZE_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
ENG_OUTDIR = "egs/models/es-en_1000-marian.1/outs"

SUMMARIZE_INSTRUCTION="Summarize the following passage in one sentence. "\
        "Do not provide any explanations or text apart from the summary.\n"\
		        "Passage: "

def sanity_check(outdir=ENG_OUTDIR):

	if not os.path.exists(outdir):
		os.makedirs(outdir)
	outfile = os.path.join(outdir, "final_outs.jsonl")

	# Summarization step
	cascade.main(
		eval_set=ENG_XSUM_JSON,
		checkpoint=HF_SUMMARIZE_MODEL,
		pretrain=True,
		summarization=True,
		outfile=outfile,
		instruction=SUMMARIZE_INSTRUCTION
	)

	# Scoring
	scores_json = os.path.join(outdir, "score.json")
	score = scoring.main(
		ref_file=ENG_XSUM_JSON,
		hyp_file=outfile,
		out_file=scores_json,
	)

	print("done with sanity check")

if __name__ == "__main__":
	sanity_check()

