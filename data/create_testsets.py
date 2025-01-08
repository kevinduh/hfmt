from datasets import load_dataset
import os, json

# Loaded dataset maps "test" -> list{item -> "summary", "text"}
# languages are: ["es", "ar", "ja", "ru", "sw", "zh", "pcm", "ta"]

langs = [
	"spanish",
	"arabic",
	"japanese",
	"russian",
	"swahili",
	"chinese_simplified",
	"pidgin",
	"tamil",
]

outdir = "CrossSum-test"
if not os.path.exists(outdir):
	os.makedirs(outdir)
	print("Created", outdir, flush=True)

for lang in langs:
	lang_pair = f"{lang}-english"
	print(f"\t... Loading {lang_pair} ...", flush=True)
	ds = load_dataset("csebuetnlp/CrossSum", lang_pair)
	# Loaded dataset maps "test" -> list{item -> "summary", "text"}
	test_ds = ds['test']
	test_list = [item for item in test_ds]
	
	out_json = os.path.join(outdir, f"{lang_pair}.jsonl")
	with open(out_json, 'w') as f:
		json.dump(test_list, f, indent=4, ensure_ascii=False)
	print('Written', out_json, flush=True)

print('done')
# 
