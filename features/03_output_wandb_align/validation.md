# Validation — 03_output_wandb_align

Off-cluster checks run during development, plus the on-cluster smoke test the **user** runs.
No GPU / training deps in the sandbox, so training itself is never executed here.

Env (off-cluster): a throwaway `uv` venv with `hydra-core 1.3.7`, `omegaconf 2.3.1`, `pyyaml`,
`pytest` — the **submitit launcher plugin is not installed**, so the in-process *basic* sweeper
stands in for fan-out. The resolver, label builder, config composition, and identity derivation
are exercised against the **real** `conf/` and `hfmt/sft_translation.py`; training is stubbed.

---

## Group A — label builder + resolver

Unit tests: `hfmt/test_run_label.py` (pure functions, no GPU/torch). **20/20 pass**
(13 for `build_run_label`/`_strict`, 7 for `derive_run_identity`).

- **A1 — descriptive label.** `build_run_label("model.lora_r=16,model.lora_target=all-linear,
  train.learning_rate=0.0002,train.seed=37")` → `lora_r-16__lora_target-all-linear__
  learning_rate-0.0002__seed-37`: group prefix stripped, leaf keys verbatim (`lora` kept),
  pairs joined by `__`.
- **A2 — shell-safe.** Output is strictly `[A-Za-z0-9._-]` even for a hostile value
  (`train.instruction=a b;c|d` collapses to dashes). Verified unquoted `cat`/`ls`/glob on a path
  containing a label needs no escaping.
- **A3 — confidentiality (CLAUDE.md).** `data.train_yaml=/proprietary/corpus.yaml`, a bare
  `data=mmtc_fr-en`, and a `/`-bearing `model.checkpoint=/path/...` all leave **no trace** in the
  label; `experiment`/`sweep`/`hydra*` selectors dropped; `+`/`~` prefixes stripped.
- **A4 — fallback.** Empty/all-excluded override_dirname → `job<N>`.
- **A5 — resolver wired.** `hfmt_runlabel` registers on import (`replace=True`); through
  OmegaConf, `${hfmt_runlabel:${od},${num}}` with `od="model.lora_r=8,train.seed=37"` →
  `lora_r-8__seed-37`, and empty `od` → `job4`.

## Group B — nested output layout (`conf/config.yaml`)

Ran the **real** `conf/config.yaml` through the in-process basic sweeper (no-op app importing
`sft_translation`, pointed at `conf/` via `--config-dir`; no torch):

```
--multirun +experiment=mmtc_fr-en_sft1 model.lora_r=8,16 model.lora_target=qv,all-linear train.seed=37
→ outputs/mmtc_fr-en_sft1/sweep-2026-10-08_15-47-06/lora_r-8__lora_target-qv__seed-37
  outputs/mmtc_fr-en_sft1/sweep-2026-10-08_15-47-06/lora_r-8__lora_target-all-linear__seed-37
  outputs/mmtc_fr-en_sft1/sweep-2026-10-08_15-47-06/lora_r-16__lora_target-qv__seed-37
  outputs/mmtc_fr-en_sft1/sweep-2026-10-08_15-47-06/lora_r-16__lora_target-all-linear__seed-37
```

- **B1 — nested + shared sweep dir.** All 4 jobs share one `sweep-<ts>/` parent (R3), each named
  by its swept values (B).
- **B2 — single run.** `+experiment=mmtc_fr-en_sft1 model.lora_r=8` (no `-m`) →
  `outputs/mmtc_fr-en_sft1/<ts>/`.
- **B3 — exclude_keys composed.** `--cfg hydra` shows
  `hydra.job.config.override_dirname.exclude_keys: [experiment, sweep, hydra/launcher]`, and the
  raw `override_dirname` for `+experiment=... model.lora_r=8` is just `model.lora_r=8`
  (experiment excluded).
- *Caveat:* `--cfg hydra --resolve` can't be used to print resolved dirs here — it resolves the
  hydra subtree in isolation, so sibling `${output_root}`/`${experiment}` are out of scope
  (unrelated to this change; the full runtime resolves them fine, as the run above shows).

## Group C — W&B identity read from the path

End-to-end against the real config (same probe, now also calling `derive_run_identity` with the
resolved `output_dir`):

```
DIR    mmtc_fr-en_sft1/sweep-2026-10-08_15-51-11/lora_r-8__seed-37
  name   = lora_r-8__seed-37                                   # == dir basename
  group  = mmtc_fr-en_sft1-sweep-2026-10-08_15-51-11
  run_id = mmtc_fr-en_sft1_sweep-2026-10-08_15-51-11_lora_r-8__seed-37
  tags   = ['sft', 'fr-en', 'sweep', 'lora_r-8', 'seed-37']
# job 2 (lora_r-16) shares the group, distinct name/run_id
SINGLE mmtc_fr-en_sft1/<ts> → name mmtc_fr-en_sft1-<ts>, group mmtc_fr-en_sft1, no sweep tag
```

- **C1 — run name == dir basename** (single source of truth).
- **C2 — group from the `sweep-<ts>` parent**; both sweep jobs share it; single runs use the
  base group.
- **C3 — run_id** embeds the sweep dir → unique across sweeps, identical on a re-derive from the
  same path (requeue-stable).
- **C4 — tags** split from the label, data-free; no `/` or data content in any of
  name/group/id/tags (unit-tested).

## Group D — adapter dir

- **D1 —** `model.save_pretrained(os.path.join(outdir, "model"))`; the adapter nests at
  `<run_dir>/model/`. Grep confirms the only remaining `_b"` references are a code comment and
  the unrelated legacy `egs/mmtc/fr-en/inf.sh`; `inf_translation.py` takes the adapter via
  `-p/--peft`, so load with `-p <run_dir>/model`.

---

## On-cluster smoke test (USER runs — needs a GPU + the conda env)

Not runnable in the sandbox. From a login node with `HFMT_ROOT=$(pwd)`:

```bash
bash egs/run.sh mmtc_fr-en_sft1 train.seed=37,42 train.max_steps=20
```

Confirm:

1. **Two Slurm jobs** submitted via submitit; both land under **one** shared
   `outputs/mmtc_fr-en_sft1/sweep-<ts>/` — i.e. `${now}` in `hydra.sweep.dir` is resolved once on
   the login node and shared across jobs (**R1/R3 — the one thing the in-process probe can't
   confirm for submitit**).
2. Run dirs `sweep-<ts>/seed-37/` and `sweep-<ts>/seed-42/`, each containing `model/` (adapter),
   `eval.pred.trg`, `dev.step_*.pred`, and `.hydra/`.
3. In W&B: two runs named `seed-37` / `seed-42` in group `mmtc_fr-en_sft1-sweep-<ts>`, tagged
   `sweep`, `seed-37`/`seed-42`; metrics only, **no dataset content** in config/names/tags.
4. Score a run: `python analysis/run_sacrebleu.py --ref <ref.trg>
   --hyp outputs/mmtc_fr-en_sft1/sweep-<ts>/seed-37/eval.pred.trg`.
5. (Optional) Requeue/resume: a requeued job reattaches to the same W&B run (same `run_id`, since
   its path is unchanged).
