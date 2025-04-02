from ner_eval import ner_eval
from scoring import get_score

import numpy as np

import json

# Formerly called ner_all

"""
python3 hfmt/ner_eval.py -r egs/data/CrossSum-test/english-spanish.jsonl -s egs/data/CrossSum-test/spanish-english.jsonl -t egs/models/nllb/es-en_e2e-llama.1/outs/final_outs.jsonl

ner_eval(
        ref_file=args.ref_file,
        src_file=args.src_file,
        hyp_file=args.hyp_file,
    )

egs/models/nllb/ja-en_100000-marian.1  egs/models/nllb/ja-en_pretrain-helsinki.1
egs/models/nllb/ja-en_e2e-llama.1      egs/models/nllb/ja-en_pretrain-nllb.1
"""

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

with open(FULL_SCORE_FILE, 'r') as f:
    full_scores = json.load(f)

backup_score_file = FULL_SCORE_FILE.replace(".json", "_backup.json")
with open(backup_score_file, 'w') as f:
    json.dump(full_scores, f, indent=4)

for iso in ISO2NAME:
    language = ISO2NAME[iso]
    for model in ISO2MODS:
        moddir = MOD2DIR[model]

        ref_file = f"egs/data/CrossSum-test/english-{language}.jsonl"
        src_file = f"egs/data/CrossSum-test/{language}-english.jsonl"
        hyp_file = f"egs/models/nllb/{iso}-en_{moddir}/outs/final_outs.jsonl"

		# First NER F1

        ner_f1_score = ner_eval(
            ref_file=ref_file,
            src_file=src_file,
            hyp_file=hyp_file
        )

        full_scores[iso][model]['ner_f1'] = ner_f1_score

		# Now BERTScore

        bert_f1s = get_score(
			ref_=ref_file, 
			hyp_=hyp_file, 
			metric="bertscore", 
			submetric="f1"
		)
        bert_f1_score = np.mean(bert_f1s)
        full_scores[iso][model]['bert_f1'] = bert_f1_score

with open(FULL_SCORE_FILE, 'w') as f:
    json.dump(full_scores, f, indent=4)

print("Rewritten", FULL_SCORE_FILE)
