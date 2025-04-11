from ner_eval import ner_eval
from scoring import get_score, get_texts

import fasttext
from huggingface_hub import hf_hub_download
import numpy as np
import json, argparse, os, pdb

# Formerly called ner_all

"""
python3 hfmt/ner_eval.py -r egs/data/CrossSum-test/english-spanish.jsonl -s egs/data/CrossSum-test/spanish-english.jsonl -t egs/models/nllb/es-en_e2e-llama.1/outs/final_outs.jsonl

ner_eval(
        doc_file=args.doc_file,
        src_file=args.src_file,
        hyp_file=args.hyp_file,
    )

egs/models/nllb/ja-en_100000-marian.1  egs/models/nllb/ja-en_pretrain-helsinki.1
egs/models/nllb/ja-en_e2e-llama.1      egs/models/nllb/ja-en_pretrain-nllb.1
"""

ALL_METRICS = ["ner", "bertscore", "fasttext"]
MOD2DIR = {
    "100000": "100000-marian.1",
	"10000": "10000-marian.1",
    "nllb": "pretrain-nllb.1",
    "helsinki": "pretrain-helsinki.1",
	"kreyol-mt": "pretrain-kreyol-mt.1",
    "e2e": "e2e-llama.1"
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
standard_mods = ["100000", "nllb", "helsinki", "e2e"]
ISO2MODS = {iso_: standard_mods[:] for iso_ in ISO2NAME}
ISO2MODS["pcm"] = ["10000", "kreyol-mt", "e2e"]
FULL_SCORE_FILE="egs/scores/full_results.json"
SINGLE_SCORE_FILE="egs/scores/single_scores.json"

FT_MOD_PATH = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")

def main(
        full_score_file=FULL_SCORE_FILE,
		single_score_file=SINGLE_SCORE_FILE,
        metrics=ALL_METRICS
    ):
	
    if "fasttext" in metrics:
        ft_model = fasttext.load_model(FT_MOD_PATH)

    with open(full_score_file, 'r') as f:
        full_scores = json.load(f)

    backup_score_file = full_score_file.replace(".json", "_backup.json")
    with open(backup_score_file, 'w') as f:
        json.dump(full_scores, f, indent=4)
	
    single_scores = {iso: {} for iso in ISO2NAME}

    for iso in ISO2NAME:
        language = ISO2NAME[iso]
        for model in ISO2MODS[iso]:
            moddir = MOD2DIR[model]

            doc_file = f"egs/data/CrossSum-test/english-{language}.jsonl"
            src_file = f"egs/data/CrossSum-test/{language}-english.jsonl"
            hyp_file_dir = f"egs/models/nllb/{iso}-en_{moddir}/outs"
            hyp_file = os.path.join(hyp_file_dir, "final_outs.jsonl")

            if model not in full_scores[iso]:
                full_scores[iso][model] = {}
            if model not in single_scores[iso]:
                single_scores[iso][model] = {}

            # First NER F1
            if "ner" in metrics:
                ner_f1_score = ner_eval(
                    ref_file=doc_file,
                    src_file=src_file,
                    hyp_file=hyp_file
                )

                full_scores[iso][model]['ner_f1'] = ner_f1_score

            # Now BERTScore

            if "bertscore" in metrics:
                ref_urls = get_texts(src_file, key="target_url")
                bert_f1s = get_score(
                    ref_=src_file, 
                    hyp_=hyp_file, 
                    metric="bertscore", 
                    submetric="f1"
                )
                bert_f1_score = np.mean(bert_f1s)
                full_scores[iso][model]['bert_f1'] = bert_f1_score
                assert len(bert_f1s) == len(ref_urls)
                single_scores[iso][model]['bert_f1'] = dict(zip(ref_urls, bert_f1s))

                #bert_score_outfile = os.path.join(hyp_file_dir, "bert_scores.jsonl")
                #with open(bert_score_outfile, 'w') as f:
                #    json.dump(bert_f1s, f)
                #print("Written", bert_score_outfile)

                '''if "rouge" in metrics:
				ref_urls = get_texts(ref_file, key="source_url")
				rouge_score = get_score(
					doc_file,
					hyp_file,
		            metric="rouge",
		            submetric="rougeL"
				)
				pass'''

            # Now FastText LID
            if "fasttext" in metrics:
                ref_urls = get_texts(src_file, key="target_url")
                hyps = get_texts(hyp_file, key="summary")
                lid_vals = []
                for hyp_text in hyps:
                    hyp_text_clean = hyp_text.strip().replace('\n', ' ')
                    lid_pred = ft_model.predict(hyp_text_clean)[0][0]
                    lid_val = int(lid_pred == '__label__eng_Latn')
                    lid_vals.append(lid_val)
                lid_score = np.mean(lid_vals)
                full_scores[iso][model]['lid%'] = lid_score
                assert len(lid_vals) == len(ref_urls)
                single_scores[iso][model]['lid%'] = dict(zip(ref_urls, lid_vals))
				
                #lid_score_outfile = os.path.join(hyp_file_dir, "lid_vals.jsonl")
                #with open(lid_score_outfile, 'w') as f:
                #    json.dump(lid_vals, f)
                #print("Written", lid_score_outfile)


    with open(full_score_file, 'w') as f:
        json.dump(full_scores, f, indent=4)
    with open(single_score_file, 'w') as f:
        json.dump(single_scores, f, indent=4)

    print("Rewritten", full_score_file)
    print("Written", single_score_file)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
            "--metrics",
            nargs='+',
            type=str, 
            default=['all']
    )

    args = parser.parse_args()

    main(metrics=ALL_METRICS if "all" in args.metrics else args.metrics)
