# synlrl experiments

Low-resource X-English translation with Tiny Aya global: SFT on a small seed bitext, and continued
pre-training (CPT) on synthetic monolingual text plus the seed bitext before SFT.

## Language pairs

| Folder | Pair |
|---|---|
| `tuk-eng/` | Turkmen-English |
| `shn-eng/` | Shan-English |
| `ewe-eng/` | Ewe-English |

All pairs share `dataset.yaml` (`- ../dataset`); each experiment sets `data.src` / `data.trg`.

## Setup

1. Put the data in one folder. For a pair `<src>-eng` (e.g. `tuk-eng`), `dataset.yaml` reads:

   | File | Used for |
   |---|---|
   | `seed1.<src>-eng.<src>`, `seed1.<src>-eng.eng` | training bitext (SFT; CPT when `cpt.use_bitext`) |
   | `synth1.<src>` | monolingual text (CPT when `cpt.use_monotext`; SFT ignores it) |
   | `flores200.smalldev-256.<src>`, `flores200.smalldev-256.eng` | dev |
   | `flores200.dev.<src>`, `flores200.dev.eng` | test (SFT; CPT never decodes) |

2. Point each pair's `local_setting.sh` at that folder:

   ```bash
   cp egs/synlrl/tuk-eng/local_setting.sh.example egs/synlrl/tuk-eng/local_setting.sh
   # then edit it:  export HFMT_DATA_ROOT=/path/to/synlrl/data
   ```

   `local_setting.sh` is untracked (one per machine, one per pair).

## Experiments

Each pair folder has three experiments, all LoRA (`lora_tiny_aya_all_linear`):

| File | What it runs |
|---|---|
| `sft_lora.yaml` | SFT from Tiny Aya global (the baseline) |
| `cpt_lora.yaml` | CPT from Tiny Aya global on `synth1` + `seed1` (`conf/cpt/default.yaml`) |
| `sft_lora_after_cpt.yaml` | SFT from `cpt_lora`'s merged model (`outputs/cpt_lora/model_merged`) |

From the repo root, on a login node:

```bash
bash egs/launch.sh cpt   egs/synlrl/tuk-eng/cpt_lora.yaml                       # CPT -> outputs/cpt_lora/
bash egs/launch.sh merge egs/synlrl/tuk-eng/outputs/cpt_lora/run_config.yaml    # LoRA -> model_merged/
bash egs/launch.sh sft   egs/synlrl/tuk-eng/sft_lora_after_cpt.yaml             # SFT after CPT
bash egs/launch.sh sft   egs/synlrl/tuk-eng/sft_lora.yaml                       # baseline: SFT without CPT
```

`sft_lora_after_cpt.yaml` is refused until `model_merged/` exists. CPT options are overridden like any
key, e.g. `cpt.monotext_ratio=0.3` or `cpt.bitext_format=concat`. `python tools/run_status.py
egs/synlrl` shows each run's state and the experiments not launched yet. CPT, its options and the forced
train keys: the top-level `README.md`.
