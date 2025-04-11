import os, glob, json, argparse
from tqdm import tqdm
import torch

import cascade_seq2seq as cascade 
import train_seq2seq as train
import scoring
from constants import LANG2FLORES_CODE

from transformers import AutoTokenizer

CHKPT_DIR="{lang}-en_{id_}-{model_id}.1"
# NLLB_CHKPT_DIR="egs/models/nllb/{lang}-en_{id_}-marian.1"

SUMMARIZE_INSTRUCTION="Summarize the following passage in one sentence. "\
		"Do not provide any explanations or text apart from the summary.\n"\
		"Passage: "
E2E_INSTRUCTION="Summarize the following passage in one sentence in English."\
		" Do not provide any explanations or text apart from the summary.\n"\
		"Passage: "
PROJECT_DIR="/export/fs05/nrobin38/xling_summarizn/hfmt"
LANGS = [
	'es', 
	'sw', 
	'ar', 
	'zh', 
	'ja', 
	'ru', 
	'ta', 
	# 'pcm'
]
LANGS2HELSINKI_IDS = {
	"es": "Helsinki-NLP/opus-mt-es-en",
	"sw": "Helsinki-NLP/opus-mt-tc-bible-big-mul-mul",
	"ar": "Helsinki-NLP/opus-mt-ar-en",
	"zh": "Helsinki-NLP/opus-mt-zh-en",
	"ja": "Helsinki-NLP/opus-mt-ja-en",
	"ru": "Helsinki-NLP/opus-mt-ru-en",
	"ta": "Helsinki-NLP/opus-mt-dra-en",
	"pcm": "Helsinki-NLP/opus-mt-tc-bible-big-mul-mul",
    "am": "Helsinki-NLP/opus-mt-tc-bible-big-mul-mul",
    "ky": "Helsinki-NLP/opus-mt-tc-bible-big-mul-mul",
    "si": "Helsinki-NLP/opus-mt-tc-bible-big-mul-mul",
    "rn": "Helsinki-NLP/opus-mt-tc-bible-big-mul-mul"
}
NLLB_MOD_ID = "facebook/nllb-200-distilled-600M"
NLLB_LANG2PT_ID = {lang_: NLLB_MOD_ID for lang_ in LANGS2HELSINKI_IDS}
KMT_LANG2PT_ID = {lang_: "jhu-clsp/kreyol-mt" for lang_ in LANGS2HELSINKI_IDS}
MODEL2LANG2PT_ID = {
	"helsinki": LANGS2HELSINKI_IDS,
	"nllb": NLLB_LANG2PT_ID,
	"kreyol-mt": KMT_LANG2PT_ID
}
ISO2NAME = {
	"es": "spanish",
	"ar": "arabic", 
	"zh": "chinese_simplified",
	"ru": "russian",
	"ja": "japanese", 
	"ta": "tamil",
	"sw": "swahili",
	"pcm": "pidgin",
	"am": "amharic",
	"ky": "kyrgyz",
	"si": "sinhala",
	"rn": "kirundi"
}
NAME2ISO = {ISO2NAME[iso]: iso for iso in ISO2NAME}

def get_args(
		lang, 
		id_, 
		model_id="marian", 
		rootdir=os.path.join(PROJECT_DIR, "egs/models/nllb"), 
		home_trained=True,
		allow_score_only=True
	):
	"""
	Get args to run run_eval() 
	"""
	if home_trained and model_id != "marian":
		print("WARNING: need marian for home trained for now", flush=True)
		model_id = "marian"
	# model_checkpoint
	chkpt_dir = CHKPT_DIR.format(lang=lang, id_=id_, model_id=model_id)
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

	if os.path.exists(final_outfile) and allow_score_only:
		print(final_outfile, "already exists")
		score_only = True
		#with open(scores_json, 'r') as f:
		#	score = json.load(f)["rouge"]
		#all_results[lang][run_type] = score
		#continue
	
	return model_checkpoint, \
		mt_outfile, \
		final_outfile, \
		scores_json, \
		flores_outfile, \
		score_only
	
def tset_eval(
		model_checkpoint, 
		src_language, 
		mt_outfile, 
		split='train',
		data_temp="egs/data/CCMatrix-{split}/{lang}-en.{split}.bitext",
		use_iso=True,
		total_n=None,
		keyword=None
	):
	if 'kreyol' in model_checkpoint:
		tokenizer = AutoTokenizer.from_pretrained(
			model_checkpoint, 
			do_lower_case=False, 
			use_fast=False, 
			keep_accents=True
		) 
	else:
		tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
		
	# Define train_data_path
	lang = NAME2ISO[src_language] if use_iso else src_language
	train_data_path = data_temp.format(split=split, lang=lang)

	D = train.get_data(
		train_data_path, 
		total_n=total_n, 
		train_ratio=1.,
		tokenizer=tokenizer,
		checkpoint_name=model_checkpoint
	)
	srcs = [datum['src'] for datum in D['train']]
	refs = [datum['trg'] for datum in D['train']]
	hyps = cascade.run_basic_eval(
		model_checkpoint, 
		srcs, 
		language=src_language
	) 
	
	# Create outfile for this 
	outdir = os.path.split(mt_outfile)[0]
	outdata = [
		{"hyp": hyp, "ref": ref, "src": src} for hyp, ref, src in zip(
			hyps, refs, srcs
		)
	]
	if not keyword:
		keyword = split
	outline_file = os.path.join(outdir, f"{keyword}set_hyp_ref_srcs.jsonl")
	with open(outline_file, 'w') as f:
		json.dump(outdata, f, indent=4, ensure_ascii=False)
	
	score = {}
	for submetric in ["chrf++", "bleu2", "bleu4"]:
		metric = submetric[:4] 
		tset_score = scoring.get_score(
			refs, 
			hyps,
			srcs,
			metric=metric, 
			submetric=submetric
		)
		score[f'{keyword}_{submetric}'] = tset_score
	
	return score 

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
		train_test=True,
		flores_outfile=None,
		skip_mt=False
	) -> float:

	if not score_only:
		if e2e:
			assert not model_checkpoint 
			assert not mt_outfile
			assert not src_language
			assert not flores_eval
			summarize_infile = crosssum_testset
		else:
			if not skip_mt:
				# MT step
				print("STEP: Running MT inference", flush=True)
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
		print("STEP: Running LLM inference", flush=True)
		cascade.main(
			eval_set=summarize_infile, 
			checkpoint=hf_summarize_model, 
			pretrain=True, 
			summarization=True, 
			outfile=final_outfile, 
			instruction=summarize_instruction
		)
		print(f"(*) Completed summarization step for {src_language}!", flush=True)

	# Scoring
	print("STEP: Running main scoring", flush=True)
	mt_file_for_eval = mt_outfile if os.path.exists(str(mt_outfile)) else ""
	score = scoring.main(
		ref_file=crosssum_testset, 
		hyp_file=final_outfile, 
		out_file=scores_json,
		mt_out_file=mt_file_for_eval
	)
	print(f"STEP: Result of main scoring = {score}", flush=True)

	# EXTRA SCORING

	# Do FLORES eval
	if flores_eval and src_language != "pidgin": # TODO add kreyol-mt FIXME
		print("STEP: Running FLORES eval", flush=True)
		flores_code = LANG2FLORES_CODE[src_language]
		refs, hyps = cascade.run_flores_eval(
			model_checkpoint, 
			flores_code, 
			flores_outfile
		)
		for submetric in ["chrf++", "bleu4"]:
			metric = submetric[:4]
			flores_score = scoring.get_score(
				refs, 
				hyps, 
				metric=metric, 
				submetric=submetric
			)
			score[f'flores_{submetric}'] = flores_score
	
	if not e2e:
		# Test on CCMatrix test set
		if train_test:
			print("STEP: Running CCMatrix test eval", flush=True)
			testset_score = tset_eval(
				model_checkpoint, 
				src_language, 
				mt_outfile, 
				split='test',
				total_n=1000
			)
			score.update(testset_score)

		# Test on in-domain CrossSum test set 
		print("STEP: Running CrossSum devtest eval", flush=True)
		xsum_score = tset_eval(
			model_checkpoint,
			src_language,
			mt_outfile,
			split='devtest',
			data_temp="egs/data/CrossSum-MT-{split}/{lang}-english.txt",
			use_iso=False,
			keyword='xsum_dev'
		)
		score.update(xsum_score)

	print(f"STEP: Final score dict: {score}")
	with open(scores_json, 'w') as f:
			json.dump(score, f, indent=4)

	return score

def main(
		rootdir=PROJECT_DIR, 
		langs=LANGS,
		run_type='100000', # can be 'pretrain' or 'e2e'
		pt_mod='marian',
		home_trained=True,
		allow_score_only=True,
		skip_mt=False,
		hf_summarize_model="meta-llama/Meta-Llama-3-8B-Instruct",
		crosssum_path="egs/data/CrossSum-test/{language}-english.jsonl",
		model_dir=os.path.join(PROJECT_DIR, "egs/models/nllb"),
		instruction_type='cascade'
	):

	if run_type == 'e2e':
		assert pt_mod == 'llama'
	
	all_results = {}

	out_json_dir = os.path.join(rootdir, 'egs', 'scores')
	if not os.path.exists(out_json_dir):
		os.makedirs(out_json_dir)
	out_json_path = os.path.join(out_json_dir, "some_results-0.json")
	while os.path.exists(out_json_path):
		out_path_idx = int(os.path.split(
				out_json_path
		)[-1].split('-')[-1].split('.')[0])
		next_idx = out_path_idx + 1
		out_json_path = out_json_path.replace(
				f"-{out_path_idx}.json",
				f"-{next_idx}.json"
		)

	for lang in langs:

		if lang not in all_results:
			all_results[lang] = {}

		# define variables independent of run_type
	
		language = ISO2NAME[lang]

		# crosssum_testset
		crosssum_json = crosssum_path.format(language=language)
		crosssum_testset = os.path.join(rootdir, crosssum_json)

		model_checkpoint, \
		mt_outfile, \
		final_outfile, \
		scores_json, \
		flores_outfile, \
		score_only = get_args(
			lang=lang, 
			id_=run_type,
			model_id=pt_mod,
			rootdir=model_dir,
			home_trained=home_trained,
			allow_score_only=allow_score_only
		)

		if not home_trained and run_type != 'e2e':
			model_checkpoint = MODEL2LANG2PT_ID[pt_mod][lang]

		if run_type == 'e2e':
			model_checkpoint = None
			mt_outfile = None
			language = None 

		e2e = run_type == 'e2e'
		summarize_instruction = E2E_INSTRUCTION if instruction_type == 'e2e' else SUMMARIZE_INSTRUCTION

		# src_language already done
		# hf_summarize_model already done

		score = run_eval(
			crosssum_testset=crosssum_testset,
			model_checkpoint=model_checkpoint,
			mt_outfile=mt_outfile,
			src_language=language,
			hf_summarize_model=hf_summarize_model,
			final_outfile=final_outfile,
			summarize_instruction=summarize_instruction,
			scores_json=scores_json,
			e2e=e2e,
			score_only=score_only,
			flores_eval=False,
			train_test=False,
			skip_mt=skip_mt
		)
		print(f"(*) Completed eval for {lang}!", flush=True)
		
		# collect score 
		label = run_type if home_trained else pt_mod
		all_results[lang][label] = score

		torch.cuda.empty_cache()
	
	# Create out JSON for all results
	with open(out_json_path, 'w') as f:
		json.dump(all_results, f, indent=4)
	print("Written", out_json_path)

	with open(out_json_path, 'r') as f:
		read_json = json.load(f)
	with open(out_json_path, 'w') as f:
		json.dump(read_json, f, indent=4, ensure_ascii=False)

	return all_results


if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument(
            "--langs",
            nargs='+',
            type=str, 
            default=['all']
    )
	parser.add_argument("--pt_mod", type=str, default='marian')
	parser.add_argument("--home_trained", action="store_true")
	parser.add_argument("--allow_score_only", action="store_true")
	parser.add_argument("--skip_mt", action="store_true")
	parser.add_argument(
		"--run_type", 
		type=str, 
		default='100000', 
		choices=["100000", "10000", "pretrain", "e2e"]
	)
	parser.add_argument(
		"--hf_summarize_model", 
		type=str, 
		default="meta-llama/Meta-Llama-3-8B-Instruct"
	)
	parser.add_argument(
		"--crosssum_path", 
		type=str, 
		default="egs/data/CrossSum-test/{language}-english.jsonl"
	)
	parser.add_argument(
		"--model_dir", 
		type=str, 
		default="/export/fs05/nrobin38/xling_summarizn/hfmt/egs/models/nllb"
	)
	parser.add_argument(
		"--instruction_type",
		type=str,
		default="cascade",
		choices=["cascade", "e2e"]
	)

	args = parser.parse_args()

	main(
		rootdir=PROJECT_DIR, 
		langs=LANGS if 'all' in args.langs else args.langs,
		run_type=args.run_type, # can be '100000' or 'pretrain' or 'e2e'
		pt_mod=args.pt_mod,
		home_trained=args.home_trained,
		allow_score_only=args.allow_score_only,
		skip_mt=args.skip_mt,
		hf_summarize_model=args.hf_summarize_model,
		crosssum_path=args.crosssum_path,
		model_dir=args.model_dir,
		instruction_type=args.instruction_type
	)

