from datasets import load_dataset, config
import sys, os, random

random.seed(42)
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
	TRAIN_LIMIT = sys.argv[1] 
except IndexError:
	TRAIN_LIMIT = 1000000
TEST_LIMIT = 1000
LIMIT = TRAIN_LIMIT + (5 * TEST_LIMIT)
TRAINDIR = "CCMatrix-train"
TESTDIR = "CCMatrix-test"
if not os.path.exists(TRAINDIR):
	os.makedirs(TRAINDIR)
if not os.path.exists(TESTDIR):
	os.makedirs(TESTDIR)
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
	
	outfile = os.path.join(TRAINDIR, f"{src}-en.train.bitext")
	train_lines = lines2write[:TRAIN_LIMIT]
	with open(outfile, 'w') as f:
		f.writelines(train_lines)
	print("Written", outfile, f"with {len(train_lines)} lines")

	testfile = os.path.join(TESTDIR, f"{src}-en.test.bitext")
	# Create test lines now 
	print("\t... making test set ...")
	possible_test_lines = lines2write[TRAIN_LIMIT:]
	train_set = set(train_lines)
	test_lines = [] 
	for test_line in possible_test_lines:
		if test_line not in train_lines:
			test_lines.append(test_line)
		if len(test_lines) >= TEST_LIMIT:
			break
	with open(testfile, 'w') as f:
		f.writelines(test_lines)
	print("Written", testfile, f"with {len(test_lines)} lines")

	print('=')

# Now for pcm 

pcm_dataset = load_dataset("jhu-clsp/kreyol-mt", "pcm-eng", split="train")
my_pcm_data = pcm_dataset[:TRAIN_LIMIT]['translation']

lines2write_pcm = [] 
for datum in my_pcm_data:
	src_text = datum['src_text'].replace('\t', ' ').strip()
	tgt_text = datum['tgt_text'].replace('\t', ' ').strip()
	line = src_text + '\t' + tgt_text + '\n'
	lines2write_pcm.append(line)

outfile = os.path.join(TRAINDIR, "pcm-en.train.bitext")
with open(outfile, 'w') as f:
	f.writelines(lines2write_pcm)
print("Written", outfile, f"with {len(lines2write_pcm)} lines")
