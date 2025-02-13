import os, glob, json
from tqdm import tqdm

import cascade_seq2seq as cascade 
import train_seq2seq as train
import scoring

from transformers import AutoTokenizer

RUNNING_E2E=True 
RUNNING_HELSINKI=True
SANITY_CHECK=True # switch this later

SUMMARIZE_INSTRUCTION="Summarize the following passage in one sentence. "\
		"Do not provide any explanations or text apart from the summary.\n"\
		"Passage: "
E2E_INSTRUCTION="Summarize the following passage in one sentence in English."\
		" Do not provide any explanations or text apart from the summary.\n"\
		"Passage: "
PROJECT_DIR="/exp/nrobinson/xling_summarizn/hfmt"
HF_SUMMARIZE_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
LANGS2AMOUNTS = {
	"es": [1000, 10000, 100000, 1000000], # has Helsinki
	"sw": [1000, 10000, 100000, 1000000],
	"ar": [1000, 10000, 100000, 1000000], # has Helsinki
	"zh": [1000, 10000, 100000, 1000000], # has Helsinki
	"ja": [1000, 10000, 100000, 1000000], # has Helsinki
	"ru": [1000, 10000, 100000, 1000000], # has Helsinki
	"ta": [1000, 10000, 100000, 1000000],
	"pcm": [1000, 10000],
}
LANGS2HELSINKI_IDS = {
	"es": "Helsinki-NLP/opus-mt-es-en",
	"sw": "Helsinki-NLP/opus-mt-tc-bible-big-mul-mul",
	"ar": "Helsinki-NLP/opus-mt-ar-en",
	"zh": "Helsinki-NLP/opus-mt-zh-en",
	"ja": "Helsinki-NLP/opus-mt-ja-en",
	"ru": "Helsinki-NLP/opus-mt-ru-en",
	"ta": "Helsinki-NLP/opus-mt-dra-en",
	"pcm": "Helsinki-NLP/opus-mt-tc-bible-big-mul-mul"
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
LANG2FLORES_CODE = {
	"spanish": "spa_Latn",
	"arabic": "arb_Arab", 
	"chinese_simplified": "zho_Hans",
	"russian": "rus_Cyrl",
	"japanese": "jpn_Jpan",
	"tamil": "tam_Taml",
	"swahili": "swh_Latn", 
}

def get_args(
		lang, 
		id_, 
		model_id="marian", 
		rootdir=PROJECT_DIR, 
		home_trained=True
	):
	"""
	Get args to run run_eval() 
	"""
	# model_checkpoint
	chkpt_dir = f"egs/models/{lang}-en_{id_}-{model_id}.1"
	full_chkpt_dir = os.path.join(rootdir, chkpt_dir)
	if home_trained:
		glob_string = os.path.join(full_chkpt_dir, "checkpoint-*")
		possible_checkpoints = glob.glob(glob_string)
		if not len(possible_checkpoints) == 1:
			print(f"WARNING: multiple checkpoints for {lang} {id_}")
		model_checkpoint = possible_checkpoints[-1]
	else:
		model_checkpoint = None 

	# out_dir (and derivatives mt_outfile, final_outfile, scores_json
	out_dir = os.path.join(full_chkpt_dir, "outs")
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	mt_outfile = os.path.join(out_dir, "mt_outs.jsonl")
	final_outfile = os.path.join(out_dir, "final_outs.jsonl")
	scores_json = os.path.join(out_dir, "scores.jsonl")
	flores_outfile = os.path.join(out_dir, "flores_hyps.jsonl")

	# score_only
	score_only = False

	if os.path.exists(final_outfile):
		print(final_outfile, "already exists")
		score_only = True
		#with open(scores_json, 'r') as f:
		#	score = json.load(f)["rouge"]
		#all_results[lang][train_amount] = score
		#continue
	
	return model_checkpoint, \
		mt_outfile, \
		final_outfile, \
		scores_json, \
		flores_outfile, \
		score_only

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
		score_only=False,
		flores_eval=True,
		flores_outfile=None,
		sanity_check=SANITY_CHECK
	) -> float:

	if not score_only:
		if e2e:
			assert not model_checkpoint 
			assert not mt_outfile
			assert not src_language
			assert not flores_eval
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
	mt_file_for_eval = mt_outfile if os.path.exists(str(mt_outfile)) else ""
	score = scoring.main(
		ref_file=crosssum_testset, 
		hyp_file=final_outfile, 
		out_file=scores_json,
		mt_out_file=mt_file_for_eval
	)

	# Do FLORES eval
	if flores_eval and not score_only and src_language != "pidgin": # TODO add kreyol-mt FIXME
		flores_code = LANG2FLORES_CODE[src_language]
		refs, hyps = cascade.run_flores_eval(
			model_checkpoint, 
			flores_code, 
			flores_outfile
		)
		flores_bleu = scoring.get_score(
			refs, 
			hyps, 
			metric="bleu", 
			submetric="bleu"
		)
		score['bleu'] = flores_bleu
		score['flores_bleu'] = flores_bleu
	
	# Do a train set eval here 
	if sanity_check and not score_only:
		tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
		D = train.get_data(
			train_data_path, 
			total_n=200, 
			train_ratio=1.,
			tokenizer=tokenizer
		)
		srcs = [datum['src'] for datum in D['train']]
		refs = [datum['trg'] for datum in D['train']]
		hyps = cascade.run_basic_eval(model_checkpoint, srcs)
		trainset_bleu = scoring.get_score(
			refs, 
			hyps, 
			metric="bleu", 
			submetric="bleu"
		)
		score['trainset_bleu'] = trainset_bleu

	with open(scores_json, 'w') as f:
			json.dump(score, f)

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

			model_checkpoint, \
			mt_outfile, \
			final_outfile, \
			scores_json, \
			flores_outfile, \
			score_only = get_args(lang=lang, id_=train_amount)

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
				score_only=score_only,
				flores_outfile=flores_outfile
			)
			
			# collect score 
			all_results[lang][train_amount] = score

		if RUNNING_HELSINKI:

			_, \
			mt_outfile, \
			final_outfile, \
			scores_json, \
			flores_outfile, \
			score_only = get_args(
				lang=lang, 
				id_="pretrain", 
				model_id="helsinki", 
				home_trained=False
			)

			score = run_eval(
				crosssum_testset=crosssum_testset,
                model_checkpoint=LANGS2HELSINKI_IDS[lang],
                mt_outfile=mt_outfile,
                src_language=language,
                hf_summarize_model=HF_SUMMARIZE_MODEL,
                final_outfile=final_outfile,
                summarize_instruction=SUMMARIZE_INSTRUCTION,
                scores_json=scores_json,
                score_only=score_only, 
				flores_outfile=flores_outfile
            )

			# collect score
			all_results[lang]["helsinki"] = score

		if RUNNING_E2E:
			# Now run end to end

			# # define final_outfile, scores_json
			# chkpt_dir = f"egs/models/{lang}-en_e2e-llama.1"
			# full_chkpt_dir = os.path.join(rootdir, chkpt_dir)
			# out_dir = os.path.join(full_chkpt_dir, "outs")
			# if not os.path.exists(out_dir):
			# 	os.makedirs(out_dir)
			# final_outfile = os.path.join(out_dir, "final_outs.jsonl")
			# scores_json = os.path.join(out_dir, "scores.jsonl")

			# # score_only
			# score_only = False

			# if os.path.exists(final_outfile):
			# 	print(final_outfile, "already exists")
			# 	score_only = True
			# 	#with open(scores_json, 'r') as f:
			# 	#	score = json.load(f)["rouge"]
			# 	#all_results[lang]['e2e'] = score
			# 	#continue 

			_, \
			mt_outfile, \
			final_outfile, \
			scores_json, \
			flores_outfile, \
			score_only = get_args(
				lang=lang, id_="e2e", model_id="llama", home_trained=False
			)
		
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
				score_only=score_only,
				flores_eval=False
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

