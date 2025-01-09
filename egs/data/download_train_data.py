from datasets import load_dataset, config
import sys, os

config.HF_DATASETS_TIMEOUT = 300

# languages are: ["es", "ar", "ja", "ru", "sw", "zh", "pcm", "ta"]

"""
Data point:
{'id': 0,
 'score': 1.2499677,
  'translation': {'en': 'They come from all parts of Egypt, just like they will at the day of His coming.',
    'nl': 'Zij kwamen uit alle delen van Egypte, evenals zij op de dag van Zijn komst zullen doen.'}}
"""

"""
For pcm:

and each datum maps ("src_text" | "tgt_text") -> text
"""

try:
	LIMIT = sys.argv[1] 
except IndexError:
	LIMIT = 1000000
OUTDIR = "CCMatrix-train"
if not os.path.exists(OUTDIR):
	os.makedirs(OUTDIR)
cc_langs = ["es", "ar", "ja", "ru", "sw", "zh", "ta"]

for src in cc_langs:
	
	dataset = load_dataset(
		"yhavinga/ccmatrix", 
		f"en-{src}", 
		streaming=True, 
		split='train'
	)

	my_data = [] 
	for batch in dataset.iter(batch_size=100):
		my_data += batch['translation']
		if len(my_data) >= LIMIT:
			break
	
	# Each datum maps 'translation' -> ('en' | src) -> text
	lines2write = [] 
	for datum in my_data:
		src_text = datum[src].replace('\t', ' ').strip()
		tgt_text = datum['en'].replace('\t', ' ').strip()
		line = src_text + '\t' + tgt_text + '\n'
		lines2write.append(line)
	
	outfile = os.path.join(OUTDIR, f"{src}-en.train.bitext")
	with open(outfile, 'w') as f:
		f.writelines(lines2write)
	print("Written", outfile, f"with {len(lines2write)} lines")
	
# Now for pcm 

pcm_dataset = load_dataset("jhu-clsp/kreyol-mt", "pcm-eng", split="train")
my_pcm_data = pcm_dataset[:LIMIT]['translation']

lines2write_pcm = [] 
for datum in my_pcm_data:
	src_text = datum['src_text'].replace('\t', ' ').strip()
	tgt_text = datum['tgt_text'].replace('\t', ' ').strip()
	line = src_text + '\t' + tgt_text + '\n'
	lines2write_pcm.append(line)

outfile = os.path.join(OUTDIR, "pcm-en.train.bitext")
with open(outfile, 'w') as f:
	f.writelines(lines2write_pcm)
print("Written", outfile, f"with {len(lines2write_pcm)} lines")
