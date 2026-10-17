# TODO — 07_hydra_sweep_refactor

Generated from `plan.md` (all open questions resolved). Phase letters match the
`<feature_dir>-<phase>` commit convention. Phases are ordered so the tree stays runnable after
each one: A removes every user of the `sweep:` block first (the probe still works and just prints
nothing), B removes the probe, C removes the schema it relied on, D updates docs, and E validates.

Out of scope / do not touch: untracked `egs/experiment/`, `data/`, `egs/mmtc/fr-en/sft1_lora.yaml.old`,
`conf/hydra/launcher/slurm_h100.yaml`, `slurm_v100.yaml`. The uncommitted edits in
`conf/hydra/launcher/slurm.yaml` must be preserved (comment lines only).

---

## A. Experiments (`egs/mmtc/fr-en/`)

- [x] Before deleting anything, note `slurm_fullft.yaml`'s resource values (`mem_gb`, `timeout_min`,
      `gres`, etc.) and the `sft1_fullft.yaml` settings (`finetune_mode: full`, `learning_rate: 2e-5`,
      `basemodel: qwen2.5-0.5b`) for the README snippet in D3.
      *(Recorded: `slurm_fullft` = `slurm` + `timeout_min: 720`, `mem_gb: 64`; fullft experiment =
      `finetune_mode: full`, `learning_rate: 2e-5`, `basemodel: qwen2.5-0.5b`, `seed: 37`.)*
- [x] Create `sft_lora.yaml` from `sft1_lora.yaml`:
  - [x] `experiment: mmtc_fr-en_sft_lora`, `wandb.group: mmtc_fr-en_sft_lora`.
  - [x] Header comment: a single run; `bash egs/run.sh mmtc/fr-en/sft_lora.yaml`; no
        feature-04 `sweep` wording.
- [x] Create `sft_lora.sweep.yaml` (self-contained, not inheriting from `sft_lora.yaml`):
  - [x] Same `defaults`, `data`, `train.seed`, `wandb.tags` as `sft_lora.yaml`.
  - [x] `experiment: mmtc_fr-en_sft_lora_sweep`, `wandb.group: mmtc_fr-en_sft_lora_sweep`.
  - [x] `hydra.sweeper.params`: `train.learning_rate: 1e-4,2e-4`, `lora.lora_r: 8,16` (4 jobs).
  - [x] Header comment: always runs the full grid; the job count; CONFIDENTIALITY note (axes are
        hyperparameters only).
- [x] `git rm` the old experiments: `sft1.yaml`, `sft1_lora.yaml`, `sft1_fullft.yaml`,
      `sft1_fullft_modelsweep.yaml`, `sft1_lora_modelsweep.yaml`, `sft1_lora_model_lr_sweep.yaml`,
      `sft1_refine1.yaml`.
- [x] `git rm conf/hydra/launcher/slurm_fullft.yaml`.
- [x] Check: `ls egs/mmtc/fr-en/` shows only `dataset.yaml`, `inf.sh`, `sft_lora.yaml`,
      `sft_lora.sweep.yaml` (plus the ignored untracked `.old` file and `outputs/`).
- [x] Extra check: under the not-yet-modified code, `sweep_probe` prints nothing for both files (exit 0);
      `--cfg hydra -p hydra.sweeper` on the sweep file shows the 2 axes under `BasicSweeper`;
      `--cfg job` on `sft_lora` composes (`experiment: mmtc_fr-en_sft_lora`, lr 2e-4, r 8, seed 37).
- [x] Check: `grep -rn "sweep:" egs/` (excluding `outputs/` and untracked dirs) returns nothing.

## B. Launcher path (`egs/run.sh`, `hfmt/sweep_probe.py`)

- [x] `egs/run.sh`: delete the probe block (`probe_overrides` loop + `sweep_axes=$(…)`, ≈`:76-95`)
      and the `${sweep_axes}` arg in the final `python` call.
- [x] `egs/run.sh`: add the dataset fast-fail after `dataset_dir` is set:
      `[ -f "${dataset_dir}/dataset.yaml" ] || { echo "egs/run.sh: dataset template not found: …" >&2; exit 1; }`
      with a one-line comment that only the default name is checked and `main()` validates fully.
- [x] `egs/run.sh`: rewrite the header comment:
  - [x] Two kinds of experiment (single run vs `hydra.sweeper.params` grid), with example invocations
        of `sft_lora.yaml` / `sft_lora.sweep.yaml`.
  - [x] Remove the `sweep.enabled=true` examples; keep the ad-hoc `lora.lora_r=8,16,32` example.
  - [x] Note that the launcher logs `#idx : overrides` for every job just before submitting.
  - [x] Update the in-process example to `+mmtc/fr-en=sft_lora`.
- [x] `egs/run.sh`: fix the launcher comment (drop "or a preset like slurm_fullft").
- [x] `git rm hfmt/sweep_probe.py`.
- [x] Check: `sh -n egs/run.sh` (syntax) passes; `grep -n "probe\|sweep_axes" egs/run.sh` is empty.
      *(run.sh 105 -> 92 lines. Only remaining `sweep_probe` mention is the `SweepConfig` docstring,
      removed in C.)*
- [x] Extra checks: a temp experiment in `egs/mmtc/zh-en/` (no `dataset.yaml`) exits 1 with the new
      fast-fail message before Python (temp file removed; covers E2); a missing experiment exits 1
      with the `sft_lora.yaml` hint. Composing run.sh's exact args (selector + `dataset_dir` +
      `output_root`) with `--cfg` works for both files. Note: `--cfg` can't be combined with
      `--multirun` (Hydra: "Only one of --run, --multirun, --cfg ..."), so E1 must call python
      directly, not via run.sh.

## C. Schema, label code, config, tests

- [x] `hfmt/hydra_config.py`: delete the `SweepConfig` dataclass and the `sweep` field on `Config`;
      drop the now-unused `Dict` import if nothing else uses it.
- [x] `hfmt/hydra_config.py`: refresh the module docstring (no inline `sweep` block; experiments
      live in `egs/<dataset>/<pair>/`, not `egs/experiment/`).
- [x] `hfmt/sft_translation.py` `_label_pairs`: change the drop rule to
      `if key == "experiment" or key.startswith("hydra"):` (remove `"sweep"` / `"sweep."`).
- [x] `hfmt/sft_translation.py` `build_run_label` docstring + `main()` comments: remove
      `sweep`/`sweep.*`/`sweep.enabled` mentions; describe the plain-run vs sweep-file cases.
- [x] `conf/config.yaml`: remove `sweep` and `sweep.enabled` from
      `hydra.job.config.override_dirname.exclude_keys`.
- [x] `conf/config.yaml`: refresh comments (header `+experiment=` usage → `+<dataset>/<pair>=<name>`;
      searchpath comment; launcher comment without `slurm_fullft`; "plain run" wording).
- [x] `hfmt/test_run_label.py`: update the structural-drop test (≈`:86-87`) to use only
      `experiment=…` and `hydra.launcher=…` tokens; remove the `sweep=…` / `sweep.enabled=…` tokens and
      their comment. Leave the other fixtures unless they reference the removed mechanism.
- [x] Check: `pytest hfmt/test_run_label.py hfmt/test_dataset_inflation.py` passes.
      *(pytest is not installed in the `hfmt` env; ran both standalone instead:
      `python hfmt/test_run_label.py` 25 passed, `python hfmt/test_dataset_inflation.py` 11 passed.)*
- [x] Check: `python -c "from hfmt.hydra_config import Config, register_configs; register_configs()"` OK.
- [x] Extra checks: both experiments compose with no `sweep` key; `sweep.enabled=true` now fails
      with `Key 'sweep' not in 'Config'` (direct python + `--cfg job`, so nothing could submit;
      the run.sh form is E3); `build_run_label` gives `lora_r-8__learning_rate-1e-4` for a
      sweeper-style `override_dirname` (with or without the `+mmtc/fr-en=…` selector) and
      falls back to `mmtc_fr-en_sft_lora` for a single run; `git grep` for
      `sweep_probe|sweep.enabled|SweepConfig|sweep.params|slurm_fullft` over `hfmt conf egs/run.sh
      egs/mmtc` is empty.

## D. Docs and comments

- [x] D1 `README.md` "Sweeping parameters" (≈`:91-126`): rewrite around the two kinds of file:
  - [x] A single-run experiment → 1 job; a sweep experiment with a `hydra.sweeper.params` example.
  - [x] Ad-hoc CLI sweeps still work on any experiment; a single-value CLI override pins a swept axis.
  - [x] Explain why: `run.sh` is always `--multirun` (submitit), and Hydra can't switch off a grid
        stored in config, so a grid is its own file.
  - [x] Keep the cross-product / `array_parallelism` and output/W&B naming bullets; update the
        example names (`mmtc_fr-en_sft_lora_sweep`, `learning_rate-…__lora_r-…`).
  - [x] Mention that the launcher prints `#idx : overrides` per job before submission.
- [x] D2 `README.md` experiment bullet (≈`:47`) + every `sft1*` example (≈10 refs) →
      `sft_lora` / `sft_lora.sweep`; drop the `sweep` block from the bullet.
- [x] D3 `README.md`: add snippets that replace the deleted examples:
  - [x] Full fine-tune: `train.finetune_mode=full train.learning_rate=2e-5 basemodel=qwen2.5-0.5b`
        plus the `hydra.launcher.*` resource overrides noted in A.
  - [x] A config-group sweep axis: `basemodel: qwen2.5-0.5b,qwen2.5-1.5b` under `hydra.sweeper.params`.
- [x] D4 `CLAUDE.md`: rewrite the "Parameter sweeps" bullet; fix the "Configuration" bullet
      (`egs/experiment/<name>.yaml` → `egs/<dataset>/<pair>/<name>.yaml`, the `run.sh` usage, and
      inline `sweep` mention); fix the tree comment for `conf/` (no `sweep`/`experiment`/`data` group).
- [x] D5 `decisions.md`: add a feature-07 entry: why the toggle and probe were dropped (fewer
      moving parts; two kinds of file), that it supersedes "Sweep toggle via a compose probe", the
      CLI-rejection correction (only comma-lists on `hydra.*` are rejected), and that `slurm_fullft`
      was removed in favour of CLI resource overrides.
- [x] D6 `tools/run_on_free_gpu.sh`: usage examples (`:8,9,14`) →
      `mmtc/fr-en/sft_lora.sweep.yaml` / `mmtc/fr-en/sft_lora.yaml train.seed=37,42 …`.
- [x] D7a `conf/hydra/launcher/local.yaml:2` usage comment → `bash egs/run.sh mmtc/fr-en/sft_lora.yaml hydra/launcher=local`.
- [ ] D7b `conf/hydra/launcher/slurm.yaml:2-4` usage comment: **left for the developer**, because the file
      holds uncommitted WIP edits (`:5`, `:26`) and a plain `git add` would mix them. Suggested:
      line 2 → `#   bash egs/run.sh mmtc/fr-en/sft_lora.yaml   (selects this launcher via its defaults)`;
      line 4 `egs/mmtc/fr-en/sft1.sh` no longer exists (only `egs/mmtc/zh-en/sft1.sh`).
- [x] Check: `git ls-files | xargs grep -n "sweep_probe\|sweep\.enabled\|SweepConfig\|sweep\.params\|slurm_fullft"`
      matches only `features/0[0-7]/` and the historical text in `decisions.md`.
      *(Also matches README's one-line "former `slurm_fullft`" note, which is intentional.)*
- [x] Extra (found during D): README `:43`/`:83` and CLAUDE.md still said `lora.enabled=false` selects a
      full fine-tune, but the code dispatches on `train.finetune_mode` since `ea6cfa0` and never reads
      `lora.enabled`. Fixed in both while adding the full-FT snippet. (`LoraConfig.enabled` is still in
      the schema, unused; left as is, out of scope.)
- [x] Extra: marked the feature-04 "Sweep toggle via a compose probe" bullet in `decisions.md` as superseded.

## E. Validation

- [ ] E1 Config-only compose (login node):
  - [ ] `python -m hfmt.sft_translation +mmtc/fr-en=sft_lora data.dataset_dir=$HFMT_ROOT/egs/mmtc/fr-en --cfg job`
        composes cleanly; no `sweep` key.
  - [ ] Same with `+mmtc/fr-en=sft_lora.sweep … --cfg hydra -p hydra.sweeper.params` shows
        the two axes.
- [ ] E2 Fast-fail: running `run.sh` against a copy of the experiment in a scratch `egs/<x>/<y>/` with no
      `dataset.yaml` exits before Python with the new message (remove the scratch dir afterwards).
- [ ] E3 Schema guard: `bash egs/run.sh mmtc/fr-en/sft_lora.yaml sweep.enabled=true` fails fast
      with a "Key 'sweep' not in 'Config'"-style error and submits nothing.
- [ ] E4 Smoke (GPU node, `hydra/launcher=local`, `train.max_steps=2`-ish, `wandb.enabled=false` or a
      scratch project):
  - [ ] `sft_lora.yaml` → 1 job; run dir leaf `mmtc_fr-en_sft_lora`; `is_sweep` false (no `sweep` tag,
        base group).
  - [ ] `sft_lora.sweep.yaml` → 4 job dirs named `learning_rate-…__lora_r-…` under one
        `sweep-<ts>/`; W&B names `mmtc_fr-en_sft_lora_sweep-<label>`, group `…-sweep-<ts>` (R1).
- [ ] E5 Real Slurm (login node): `bash egs/run.sh mmtc/fr-en/sft_lora.sweep.yaml` prints the sweep dir
      and 4 `#idx : …` lines and submits a 4-task job array (cancel after confirming if not needed).
- [ ] E6 Confidentiality: inspect a smoke run's W&B config/name/tags; only hyperparameters, paths,
      and the instruction string; no data content.
- [ ] E7 Report `git diff --stat` line delta (expect a net reduction) and mark this todo complete.
