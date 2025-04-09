import os, json

OUTDIR = "eng_counterparts"

langs = [
	"spanish",
	"arabic",
	"japanese",
	"russian",
	"swahili",
	"tamil",
	"chinese_simplified",
	"pidgin"
]

for lang in langs:
	print("...", "Processing", lang, "...")

	bitext_json = f"{lang}-english.jsonl"
	english_json = "english-english_full.jsonl"

	with open(bitext_json, 'r') as f:
		bitext_data = json.load(f)
	with open(english_json, 'r') as f:
		english_data = json.load(f)
	
	url2datum = {datum['target_url']: datum for datum in english_data}
	english_counterpart_data = [
		url2datum[datum['target_url']] for datum in bitext_data
	]

	out_json = os.path.join(OUTDIR, f"english-counterpart_{lang}.jsonl")
	with open(out_json, 'w') as f:
		json.dump(english_counterpart_data, f, indent=4, ensure_ascii=False)
	
	print("Written", out_json)

print('done')

