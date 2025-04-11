from datasets import load_dataset
import os, json, pdb

# Loaded dataset maps "test" -> list{item -> "summary", "text"}
# languages are: ["es", "ar", "ja", "ru", "sw", "zh", "pcm", "ta"]

langs = [
	#"spanish",
	#"arabic",
	#"japanese",
	#"russian",
	#"swahili",
	#"chinese_simplified",
	#"pidgin",
	#"tamil",
	"amharic",
	"kyrgyz",
	"kirundi",
	"sinhala",
	#"english"
]

# Create output directories

outdir = "CrossSum-test"
mtdevdir = "CrossSum-MT-devtest"
if not os.path.exists(outdir):
	os.makedirs(outdir)
	print("Created", outdir, flush=True)
if not os.path.exists(mtdevdir):
	os.makedirs(mtdevdir)
	print("Created", mtdevdir, flush=True)

# Get english sset
#eng_ds = load_dataset(f"csebuetnlp/CrossSum", "english-english")
#eng_url2idx = {}
#for i, sample in enumerate(eng_ds['test']):
#	eng_url2idx[sample['target_url']] = i

# Retrieve test sets 

for lang in langs:
	# define language pair and load set
	lang_pair = f"{lang}-english"
	bwd_lang_pair = f"english-{lang}"
	print(f"\t... Loading {lang_pair} ...", flush=True)
	ds = load_dataset("csebuetnlp/CrossSum", lang_pair)
	bwd_ds = load_dataset("csebuetnlp/CrossSum", bwd_lang_pair)
	
	# Prep dataset for processing
	# Loaded dataset maps "test" -> list{item -> "summary", "text"}
	test_ds = ds['test']
	test_list = [item for item in test_ds]
	bwd_test_ds = bwd_ds['test']
	bwd_test_list = [item for item in bwd_test_ds]

	# get MT segments for dev
	bwd_url2idx = {}
	for i, sample in enumerate(bwd_ds['test']):
		bwd_url2idx[sample['source_url']] = i
	mt_lines = []
	for j, sample in enumerate(test_list):
		if sample['target_url'] in bwd_url2idx:
			i = bwd_url2idx[sample['target_url']]
			eng = sample['summary']
			src = bwd_ds['test'][i]['summary']
			mt_lines.append(src.strip() + '\t' + eng.strip() + '\n')
	mtdev_path = os.path.join(mtdevdir, f"{lang_pair}.txt")
	with open(mtdev_path, 'w') as f:
		f.writelines(mt_lines)
	print(f'(Written {mtdev_path})')

	# Write output test set
	out_json = os.path.join(outdir, f"{lang_pair}.jsonl")
	with open(out_json, 'w') as f:
		json.dump(test_list, f, indent=4, ensure_ascii=False)
	print('Written', out_json, flush=True)

	# And reverse
	bwd_out_json = os.path.join(outdir, f"{bwd_lang_pair}.jsonl")
	with open(bwd_out_json, 'w') as f:
		json.dump(bwd_test_list, f, indent=4, ensure_ascii=False)
	print('Written', bwd_out_json, flush=True)


print('done')
# 
