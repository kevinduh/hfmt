import os, glob, json
from tqdm import tqdm

import cascade_seq2seq as cascade
import scoring

RUNNING_E2E=True # switch this later

SUMMARIZE_INSTRUCTION="Summarize the following passage in one sentence. Do not provide any explanations or text apart from the summary.\nPassage: "
E2E_INSTRUCTION="Summarize the following passage in one sentence in English. Do not provide any explanations or text apart from the summary.\nPassage: "
PROJECT_DIR="/exp/nrobinson/xling_summarizn/hfmt"
HF_SUMMARIZE_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
LANGS2AMOUNTS = {
	"es": [1000, 10000, 100000, 1000000],
	"sw": [1000, 10000, 100000, 1000000],
	"ar": [1000, 10000, 100000, 1000000],
	"zh": [1000, 10000, 100000, 1000000],
	"ja": [1000, 10000, 100000, 1000000],
	"ru": [1000, 10000, 100000, 1000000],
	"ta": [1000, 10000, 100000, 1000000],
	"pcm": [1000, 10000],
}
ISO2NAME = {
	"es": "spanish",
	"ar": "arabic", 
	"zh": "chinese_simplified",
	"ru": "russian",
	"ja": "japanese", 
	"ta": "tamil",
	"sw": "swahili",
	"pcm": "pidgin"
}

def run_eval(
		crosssum_testset="egs/data/CrossSum-test/spanish-english.toy.jsonl",
		model_checkpoint="egs/models/marian.scratch.1/checkpoint-100000",
		mt_outfile="$outdir/mt_outs.jsonl",
		src_language="spanish",
		hf_summarize_model="meta-llama/Meta-Llama-3-8B-Instruct",
		final_outfile="$outdir/final_outs.jsonl",
		summarize_instruction=SUMMARIZE_INSTRUCTION,
		scores_json="$outdir/scores.jsonl",
		e2e=False,
		score_only=False
	) -> float:

	if not score_only:
		if e2e:
			assert not model_checkpoint 
			assert not mt_outfile
			assert not src_language
			summarize_infile = crosssum_testset
		else:
			# MT step
			cascade.main(
				eval_set=crosssum_testset, 
				checkpoint=model_checkpoint, 
				pretrain=True, 
				summarization=False, 
				outfile=mt_outfile, 
				language=src_language
			)
			summarize_infile = mt_outfile

		# Summarization step
		cascade.main(
			eval_set=summarize_infile, 
			checkpoint=hf_summarize_model, 
			pretrain=True, 
			summarization=True, 
			outfile=final_outfile, 
			instruction=summarize_instruction
		)

	# Scoring
	mt_file_for_bleu = mt_outfile if os.path.exists(str(mt_outfile)) else ""
	score = scoring.main(
		ref_file=crosssum_testset, 
		hyp_file=final_outfile, 
		out_file=scores_json,
		mt_out_file=mt_file_for_bleu
	)

	return score

def main(rootdir=PROJECT_DIR, jobs=LANGS2AMOUNTS):
	
	all_results = {}

	for lang in jobs:

		all_results[lang] = {}

		# define variables independent of train_amount
	
		language = ISO2NAME[lang]

		# crosssum_testset
		crosssum_json = f"egs/data/CrossSum-test/{language}-english.jsonl"
		crosssum_testset = os.path.join(rootdir, crosssum_json)

		for train_amount in tqdm(jobs[lang], desc=lang):

			# model_checkpoint
			chkpt_dir = f"egs/models/{lang}-en_{train_amount}-marian.1"
			full_chkpt_dir = os.path.join(rootdir, chkpt_dir)
			glob_string = os.path.join(full_chkpt_dir, "checkpoint-*")
			possible_checkpoints = glob.glob(glob_string)
			if not len(possible_checkpoints) == 1:
				continue # Skip if 0 or 2 checkpoints (training not complete)
			model_checkpoint = possible_checkpoints[0]

			# out_dir (and derivatives mt_outfile, final_outfile, scores_json
			out_dir = os.path.join(full_chkpt_dir, "outs")
			if not os.path.exists(out_dir):
				os.makedirs(out_dir)
			mt_outfile = os.path.join(out_dir, "mt_outs.jsonl")
			final_outfile = os.path.join(out_dir, "final_outs.jsonl")
			scores_json = os.path.join(out_dir, "scores.jsonl")

			# score_only
			score_only = False

			if os.path.exists(final_outfile):
				print(final_outfile, "already exists")
				score_only = True
				#with open(scores_json, 'r') as f:
				#	score = json.load(f)["rouge"]
				#all_results[lang][train_amount] = score
				#continue

			# src_language already done
			# hf_summarize_model already done

			score = run_eval(
				crosssum_testset=crosssum_testset,
				model_checkpoint=model_checkpoint,
				mt_outfile=mt_outfile,
				src_language=language,
				hf_summarize_model=HF_SUMMARIZE_MODEL,
				final_outfile=final_outfile,
				summarize_instruction=SUMMARIZE_INSTRUCTION,
				scores_json=scores_json,
				score_only = score_only
			)
			
			# collect score 
			all_results[lang][train_amount] = score

		if RUNNING_E2E:
			# Now run end to end

			# define final_outfile, scores_json
			chkpt_dir = f"egs/models/{lang}-en_e2e-llama.1"
			full_chkpt_dir = os.path.join(rootdir, chkpt_dir)
			out_dir = os.path.join(full_chkpt_dir, "outs")
			if not os.path.exists(out_dir):
				os.makedirs(out_dir)
			final_outfile = os.path.join(out_dir, "final_outs.jsonl")
			scores_json = os.path.join(out_dir, "scores.jsonl")

			# score_only
			score_only = False

			if os.path.exists(final_outfile):
				print(final_outfile, "already exists")
				score_only = True
				#with open(scores_json, 'r') as f:
				#	score = json.load(f)["rouge"]
				#all_results[lang]['e2e'] = score
				#continue 
		
			score = run_eval(
				crosssum_testset=crosssum_testset,
				model_checkpoint=None,
				mt_outfile=None,
				src_language=None,
				hf_summarize_model=HF_SUMMARIZE_MODEL,
				final_outfile=final_outfile,
				summarize_instruction=E2E_INSTRUCTION,
				scores_json=scores_json,
				e2e=True,
				score_only=score_only
			)

			all_results[lang]['e2e'] = score
	
	# Create out JSON for all results
	out_json_dir = os.path.join(rootdir, 'egs', 'scores')
	if not os.path.exists(out_json_dir):
		os.makedirs(out_json_dir)
	out_json_path = os.path.join(out_json_dir, "full_results.json")
	with open(out_json_path, 'w') as f:
		json.dump(all_results, f, indent=4)
	print("Written", out_json_path)

	return all_results


if __name__ == "__main__":

	main()

