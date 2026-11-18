# MMTC experiments

SFT experiments on the **Multilingual Microblog Translation Corpus (MMTC)**: manually translated
Twitter posts, 13 source languages into English.

* Paper: McNamee and Duh, [The Multilingual Microblog Translation Corpus: Improving and Evaluating
  Translation of User-Generated Text](https://www.cs.jhu.edu/~kevinduh/papers/mcnamee22mttt.pdf), LREC 2022.
* Data: <https://pmcnamee.net/research/mmtc/mmtc.html>. The data is not in this repo, and its content
  must never be committed or sent to W&B (`CLAUDE.md`, "Data confidentiality").

## Language pairs

Each pair has its own folder, using the corpus's `train` / `valid` / `test` partitions (sizes from the
paper, Table 4):

| Folder | Pair | train | valid | test |
|---|---|---|---|---|
| `fr-en/` | French-English | 2,485 | 2,000 | 2,000 |
| `zh-en/` | Chinese-English | 1,580 | 2,000 | 2,000 |
| `ur-en/` | Urdu-English | 1,118 | 2,000 | 2,000 |

## Setup

See `fr-en/` for example usage; the other pairs work the same way.

1. Put the pair's data in one folder, as sentence-aligned text files named
   `{train,valid,test}.<src>-en.<src>` and `{train,valid,test}.<src>-en.en` (e.g. `train.fr-en.fr`,
   `train.fr-en.en`). `dataset.yaml` lists these files.
2. Set `fr-en/local_setting.sh` to point to that folder:

   ```bash
   cp egs/mmtc/fr-en/local_setting.sh.example egs/mmtc/fr-en/local_setting.sh
   # then edit it:  export HFMT_DATA_ROOT=/path/to/data/fr-en
   ```

   `local_setting.sh` is untracked (one per machine, one per pair).
3. Launch an experiment from the repo root, on a login node:

   ```bash
   bash egs/launch.sh sft egs/mmtc/fr-en/sft_lora.yaml          # one LoRA run
   bash egs/launch.sh sft egs/mmtc/fr-en/sft_lora.sweep.yaml    # a 4-job LoRA sweep
   DRYRUN=1 bash egs/launch.sh sft egs/mmtc/fr-en/sft_lora.yaml # print the sbatch command only
   ```

   Outputs go to `<pair>/outputs/<run_name>/`. `python tools/run_status.py egs/mmtc/fr-en/outputs`
   shows each run's state and scores.

Each pair also has full fine-tunes of Qwen3.5-4B (`sft_qwen-3.5-4b.yaml`), Tiny Aya
(`sft_tiny-aya-global.yaml`) and Gemma 3 4B (`sft_gemma-3-4b-it.yaml`); these need an 80 GB A100.
Overrides, resume, inference and merging: the top-level `README.md`, "Running SFT experiments".

## Metrics

Each test set is scored into `scores.json`, and the dev set during training (`eval_*` keys):

| Key | Metric |
|---|---|
| `bleu_13a` | **The MMTC metric.** Case-insensitive BLEU as in the paper: sacreBLEU, 13a tokenization, exp smoothing (`BLEU+case.lc+numrefs.1+smooth.exp+tok.13a`). Report this one to compare with the paper. |
| `sp_bleu` | Cased BLEU with sacreBLEU's `flores200` SentencePiece tokenization. |
| `chrf` | chrF |
| `ter` | TER |

To select checkpoints on the MMTC metric, set `train.metric_for_best_model: eval_bleu_13a` (with
`train.greater_is_better: true`); the `full_finetune_short` schedule used by the full fine-tunes
already does. To score a prediction file by
hand: `python analysis/run_sacrebleu.py --ref <ref> --hyp <hyp>`.
