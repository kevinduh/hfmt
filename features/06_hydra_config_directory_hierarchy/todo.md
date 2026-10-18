# TODO — 06_hydra_config_directory_hierarchy

Generated from `plan.md` (all four design decisions settled; F3 addressing validated by the
2026-10-16 compose spike). Phase letters match the `<feature_dir>-<phase>` commit convention.
Dependencies flow top-to-bottom; the tree is temporarily inconsistent between phases B–F and
coherent again after F.

Group **I** was added on 2026-10-18 from plan §H (naming schema; decision 5 settled as
bare-stem groups). It reverses F's "kept each `experiment:` value as-is".

---

## A. Confidentiality guard (do first — must precede any run)

- [x] Add `egs/**/outputs/` to `.gitignore` (outputs move inside the tracked `egs/` tree and
      hold data-derived `*.pred` / `eval.pred.trg` dumps — R4).
- [x] Confirm nothing else globs `egs/**` into packaging/sdist (`MANIFEST.in`, `pyproject`,
      `setup.cfg`) that would sweep run outputs in. *(None exist in the repo.)*
- [x] `git check-ignore egs/mmtc/fr-en/outputs/x.pred` returns a match (sanity-check the rule).
      *(Matches at single-run + sweep depth; `dataset.yaml` correctly NOT ignored.)*

## B. Schema foundation — repurpose `DataConfig` (`hfmt/hydra_config.py`)

- [x] Add a `DataSplit` dataclass: `src: List[str] = field(default_factory=list)`,
      `trg: List[str] = field(default_factory=list)` (post-inflation absolute paths).
- [x] Repurpose `DataConfig`:
  - [x] Rename `src_lang`→`src`, `trg_lang`→`trg`.
  - [x] **Remove** `train_yaml` and `evalset`.
  - [x] Add `data_root: str = MISSING` (abs dir where data FILES live — inflation input).
  - [x] Add `dataset_file: str = "dataset.yaml"` (template basename, incl. `.yaml`).
  - [x] Add `dataset_dir: str = MISSING` (experiment's own dir; injected by `run.sh`).
  - [x] Keep `instruction: str = ""`.
  - [x] Add `train: DataSplit` (required), `dev: DataSplit` (required),
        `test: Optional[DataSplit] = None` (optional).
- [x] Keep the `data: DataConfig` field on `Config` (schema stays; only the `conf/data/` group
      goes) and keep `register_configs()` wiring intact.
- [x] `conf/config.yaml`: remove the `- data: mmtc_fr-en` line from `defaults`. Leave
      `hydra.searchpath`, `hydra.run.dir`/`sweep.dir`, and `override_dirname.exclude_keys`
      unchanged (the stale `experiment` entry is a harmless no-op).
  - *Verified:* `OmegaConf.structured(Config)` + `compose(config_name="config")` (no
    experiment) succeed; `data.src` is `MISSING`, `dataset_file`=`dataset.yaml`, `test`=`None`.

## C. Data inflation, validation & loading (`hfmt/sft_translation.py`)

- [x] Add a template-inflation helper (`inflate_dataset_template`): read
      `join(data.dataset_dir, data.dataset_file)` as **text**, substitute `$DATA_ROOT`/`$SRC`/
      `$TRG` via `string.Template`, then `yaml.safe_load`. Split values stay **lists** (R8).
- [x] **Template-existence validation (R10)** (`resolve_dataset_template`): assert the resolved
      path exists; clear single-line errors for `dataset_file` without `.yaml`, a path/absolute
      `dataset_file`, and unset `dataset_dir`. *(All 5 error cases verified, incl. missing
      `data_root`/`src`/`trg`.)*
- [x] Early in `main()` (`populate_data_paths`, before `get_data` and the resolved-config log):
      inflate and write `train`/`dev` into `cfg.data`. For `test`: populate `cfg.data.test`
      only if the `test` block exists **and** its `src` file(s) exist on disk; else `None`.
  - [x] Struct-mode writeback (R2b): assign **`DataSplit` instances**, not dicts (plain dict is
        rejected — confirmed); `test=None` for the skip case.
- [x] Rework `get_data()` to read `cfg.data.train` / `cfg.data.dev` directly (dropped the
      `train_yaml` file read).
- [x] Final inference pass: read source from `cfg.data.test.src`, **guarded on `cfg.data.test
      is not None`** — skip with an info log when there is no test set.
- [x] Retire `evalset` terminology: final-inference local is now `test_data`; the shared
      `inference_on_eval_data` helper keeps its generic name (used by dev callback + test).
      Grep confirms no `evalset`/`train_yaml`/`src_lang`/`trg_lang` references remain (fixed a
      stale `build_run_label` docstring example too).

## D. Run-label guard (`hfmt/sft_translation.py`)

- [x] In `_label_pairs`, extend the drop condition to `... or "/" in val or "/" in key:` so the
      slash-keyed group selector (`+mmtc/fr-en=sft1`) is filtered out of run-dir/W&B labels and
      the `is_sweep` check. *(Verified: plain run → `[]`/`is_sweep=False`; sweep axes + group
      choices kept; drops at any nesting depth; legacy flat `experiment=` still handled.)*
- [x] Update the `build_run_label`/`_label_pairs` docstring to mention the nested group
      selector and why slash-keyed keys are dropped (also retired the obsolete `data=<name>`
      group-choice narrative).

## E. Launcher addressing (`egs/run.sh` + `hfmt/sweep_probe.py`)

- [x] `egs/run.sh`: take one `egs`-relative arg (the `.yaml` file). Derive
      `group=$(dirname …)`, `name=$(basename …)` after stripping a trailing `.yaml` (tolerate a
      missing `.yaml`); reject a bare name with no `<dataset>/<lang>/` dir; check the file exists.
  - [x] Emit `+${group}=${name}` to both the `sweep_probe` call (as `<group>=<name>`) and the
        training launch.
  - [x] Inject `data.dataset_dir=${HFMT_ROOT}/egs/${group}` and
        `output_root=egs/${group}/outputs`.
  - [x] Update the usage string / header comments and the inline `python -m …` example to the
        new single-arg form; note a raw `python -m` run must pass `data.dataset_dir=…`.
  - [x] Keep the `sweep.enabled=*` passthrough probing logic working with the new selector.
  - *Verified:* path derivation across `.yaml`/no-ext/other-langpair inputs + bare-name
    rejection; `sh -n` clean.
- [x] `hfmt/sweep_probe.py`: compose with `+${selector}` (`<group>=<name>`) instead of
      `+experiment=<name>`; selector is argv[0], `data.dataset_dir`/`sweep.*` are extra overrides.
  - [x] Add the **login-node fast-fail** template-existence check here (cheap existence subset;
        full validation stays authoritative in `sft_translation.py`); surfaced as rc≠0 so
        `run.sh`'s `$(...)`+`set -e` aborts the submit.
  - [x] Keep it dependency-light (no `transformers`); reuse the stub `hfmt_runlabel` resolver.
  - *Verified:* disabled→empty/rc0, enabled→axes/rc0, missing template→clear stderr/rc2.

## F. Templates, experiment migration & conf/data removal

- [x] Create `egs/mmtc/fr-en/dataset.yaml` from the plan's template (train/dev/test × src/trg).
      **Fixed the request's typo:** `train.trg` uses `$TRG` (not `$SRC`) — R6.
- [x] Migrate the 7 experiments from `egs/experiment/` into `egs/mmtc/fr-en/`, dropping the
      `mmtc_fr-en_` filename prefix (dir now carries it) — used `git mv`:
  - [x] `mmtc_fr-en_sft1.yaml` → `sft1.yaml`
  - [x] `mmtc_fr-en_sft1_lora.yaml` → `sft1_lora.yaml`
  - [x] `mmtc_fr-en_sft1_lora_modelsweep.yaml` → `sft1_lora_modelsweep.yaml`
  - [x] `mmtc_fr-en_sft1_lora_model_lr_sweep.yaml` → `sft1_lora_model_lr_sweep.yaml`
  - [x] `mmtc_fr-en_sft1_fullft.yaml` → `sft1_fullft.yaml`
  - [x] `mmtc_fr-en_sft1_fullft_modelsweep.yaml` → `sft1_fullft_modelsweep.yaml`
  - [x] `mmtc_fr-en_sft1_refine1.yaml` → `sft1_refine1.yaml`
- [x] In each migrated experiment:
  - [x] Remove `- override /data: ...` from the `defaults` list.
  - [x] Add the inline `data:` block — `data_root: ${oc.env:HFMT_ROOT}/data/fr-en`, `src: fr`,
        `trg: en`, `instruction` (`dataset_file` omitted → schema default `dataset.yaml`).
  - [x] Kept each `experiment:` value as-is (W&B group / output-path stability).
  - [ ] **Developer to confirm:** the `{train,dev,test}.fr-en.{fr,en}` files exist under
        `data_root` (`$HFMT_ROOT/data/fr-en`) — the old conf used this path for `evalset`, but
        the Hydra `train_yaml` manifest pointed elsewhere, so verify the layout before a run.
- [x] Delete `conf/data/mmtc_fr-en.yaml` (and the now-empty `conf/data/` dir). `egs/experiment/`
      removed once empty.
- [ ] (Optional / R9) scaffold `egs/mmtc/de-en/`, `egs/mmtc/zh-en/` `dataset.yaml` — deferred;
      only `mmtc/fr-en` populated now.
- *Found + fixed during verification:* `string.Template` tripped on a literal `${...}` in a
  template comment; inflation now parses YAML first and substitutes only in string values
  (robust to comment `$`). All 7 experiments compose + inflate (train/dev paths correct,
  test→None here since data files are absent in this sandbox).

## G. Validation & testing

- [x] Compose check (via the `compose()` API — Py3.14/Hydra-1.3.7 CLI is broken on this box):
      all 7 experiments resolve with inflated absolute `train`/`dev`/`test` paths; **no**
      `conf/data`/`train_yaml`/`evalset` refs remain in the migrated SFT workflow (phase F).
- [x] Added `hfmt/test_dataset_inflation.py` (11 tests): inflated absolute paths, `src≠trg`
      (typo regression), `test=None` when files **or** the `test` block are absent, `test`
      populated when files exist, comment-`$`/`${}` safety.
- [x] Added `_label_pairs` tests to `hfmt/test_run_label.py`: `+mmtc/fr-en=sft1`→`[]`,
      selector+axes→axes only, dropped at any nesting depth, plain-run fallback to experiment.
- [x] Added validation tests (R10): unknown placeholder, missing `dataset_dir`/`data_root`,
      `dataset_file` without `.yaml`, path-bearing `dataset_file`, nonexistent template.
  - *All 36 tests pass (`pytest hfmt/test_dataset_inflation.py hfmt/test_run_label.py`).*
- [ ] **Dry run via `bash egs/run.sh mmtc/fr-en/sft1.yaml`** (single + 2-axis sweep): outputs
      under `egs/mmtc/fr-en/outputs/...`, W&B name == run-dir basename. *(DEFERRED — needs the
      real GPU/cluster env: this sandbox lacks torch/datasets, the submitit launcher plugin,
      and a Hydra-compatible Python. Run on a login node.)*
- [x] `git status` shows nothing under `egs/**/outputs/` staged; repo-wide grep for
      `train_yaml`/`evalset`/`src_lang`/`trg_lang`/`conf/data` returns only feature-06 comments
      and the **unmigrated non-SFT entry points** (seq2seq/inference/summarization + their shell
      recipes), which are out of scope per CLAUDE.md.
- [x] **Confidentiality pass (CLAUDE.md):** only `wandb.log` of BLEU/CHRF/TER (aggregate) and
      data-free config (hyperparameters + data *paths* + instruction) reach W&B;
      `WANDB_LOG_MODEL=false` unchanged; `populate_data_paths` writes only paths, never content.

## H. Docs & decisions

- [x] `README.md`: replaced the `conf/data/` bullet with the inline-`data` + `dataset.yaml`
      template description; updated the "Running"/"Adding an experiment"/"Sweeping" sections and
      all invocations to `bash egs/run.sh mmtc/fr-en/<name>.yaml` + the nested selector; outputs
      path now `egs/<dataset>/<src>-<trg>/outputs/`.
- [x] `overview.md`: §2.3 notes the SFT workflow's feature-06 `dataset.yaml` template inflation
      (other entry points keep the plain manifest).
- [x] `decisions.md`: added the feature-06 entry (hierarchy move, nested-group addressing with
      the spike finding, code-side YAML-first inflation + comment-`$` fix, optional test, fast-
      fail validation, outputs/confidentiality, scope).
- [x] Updated this `todo.md` (items checked off) as phases completed.

## I. Naming schema — stem-only names, W&B context via tags/config (plan §H)

Target for `egs/mmtc/fr-en/sft1_refine1.yaml`: the dir is
`egs/mmtc/fr-en/outputs/sft1_refine1/sweep-<ts>/learning_rate-0.0003__seed-37/`, the W&B name is
`sft1_refine1-learning_rate-0.0003__seed-37`, the group is `sft1_refine1-sweep-<ts>`, and the
tags include `mmtc` + `fr-en`. No name or group contains `mmtc` or `fr-en`.

### I1. Schema (`hfmt/hydra_config.py`)
- [x] Add `dataset_name: Optional[str] = None` to `DataConfig` (comment: dataset identifier,
      filled in code from `dataset_dir` when unset; identifier only, never data content).
- [x] Update the `DataConfig` docstring to mention `dataset_name` and its derivation.

### I2. Context derivation + lang-pair guard (`hfmt/sft_translation.py`, `populate_data_paths`)
*Depends on I1.*
- [x] After `resolve_dataset_template` succeeds (so `dataset_dir` is known to be set), fill in
      `cfg.data.dataset_name = basename(dirname(normpath(dataset_dir)))` **only if unset**
      (`mmtc` for `.../egs/mmtc/fr-en`).
- [x] **Lang-pair consistency check (fail fast):** require
      `basename(normpath(dataset_dir)) == f"{data.src}-{data.trg}"`; otherwise raise a clear
      single-line `ValueError` naming both values (catches a fr-en yaml copied into `de-en/`
      without editing `src`/`trg`).
- [x] Update the `populate_data_paths` docstring (now also fills in `dataset_name` + runs the
      lang-pair check).
- [x] *(Optional)* Mirror the lang-pair check in `hfmt/sweep_probe.py`'s login-node fast-fail,
      next to the template-existence check (cheap; aborts before the Slurm submit).

### I3. W&B identity (`hfmt/sft_translation.py`, `derive_run_identity`)
*Depends on I2 (`populate_data_paths` runs before `derive_run_identity` in `main()`; confirm the
order is kept).*
- [x] **Auto-tags:** build `tags` as `[data.dataset_name, f"{data.src}-{data.trg}"]` +
      `wandb.tags`, deduplicated with order kept; then append `sweep` + param tags as today
      (sweep branch only). Run each through `_sanitize_label`.
- [x] **Run-id prefix (R11):** prepend `f"{dataset_name}_{src}-{trg}_"` to both run_id forms
      (`{experiment}_{sweep_dir}_{label}` and `{experiment}_{label}`) before the existing
      `[^A-Za-z0-9_.-]` → `-` cleanup.
- [x] Run **name** and **group** logic stays unchanged (both come from `cfg.experiment`, now the
      stem). Confirm by reading that no other code path adds dataset/lang context to them.
- [x] Update the `derive_run_identity` docstring: names/groups are stem-only, dataset + lang
      pair go to the tags and the hidden id, and why (R11).
- [x] Fix the `build_run_label` docstring example `mmtc_fr-en_sft1_fullft` → `sft1_fullft`.
- [x] **Confidentiality check (CLAUDE.md):** the new tags and the `dataset_name` config field are
      identifiers only (dataset name + lang codes), not data content. Nothing new or
      data-bearing reaches W&B.

### I4. Launcher (`egs/run.sh`)
- [x] Inject `experiment="${exp_name}"` into the training launch, next to `data.dataset_dir`
      and `output_root`. Confirm it doesn't affect the label: `experiment` is in
      `override_dirname.exclude_keys` and dropped by `_label_pairs`, so `is_sweep` stays false
      for a plain run.
- [x] Update the header comments: the experiment name is the file stem (injected), and the raw
      `python -m hfmt.sft_translation` example adds `experiment=sft1` next to `data.dataset_dir=…`.

### I5. Experiment yamls (`egs/mmtc/fr-en/*.yaml`, 7 files)
*Do with I4: once the `experiment:` line is removed, only run.sh sets the name.*
- [x] Delete the `experiment: mmtc_fr-en_*` line from all 7.
- [x] Delete `wandb.group: mmtc_fr-en_*` from all 7 (the group falls back to `cfg.experiment`).
- [x] Drop the hand-written `fr-en` from each `wandb.tags` list (now added automatically); keep
      the descriptive tags (`sft`, `lora`, `fullft`, `refine1`).
- [x] Update comments that cite the old names (`sft1_lora.yaml` "identical to mmtc_fr-en_sft1",
      `sft1_fullft_modelsweep.yaml` "see mmtc_fr-en_sft1_fullft") to the stem names.

### I6. Analysis tooling (`analysis/pull_wandb.py`)
- [x] Add a repeatable `--tag` arg (`action="append"`). Combine tags with **AND**
      (`{"$and": [{"tags": {"$in": [t]}} for t in tags]}`) together with the existing
      `group`/`state` filters, so `--group sft1 --tag mmtc --tag fr-en` pins one lang pair (R12).
- [x] Update the usage docstring example (`--group mmtc_fr-en_sft1` →
      `--group sft1 --tag mmtc --tag fr-en`).

### I7. Tests
*Depends on I2–I3.*
- [x] `hfmt/test_run_label.py`: move the fixtures to stem names (`_cfg(experiment="sft1", …)`,
      `SWEEP_OUT`/`SINGLE_OUT`/`PLAIN_OUT` under `.../egs/mmtc/fr-en/outputs/sft1…`). Give
      `_cfg` the `data.dataset_name`/`src`/`trg` fields and update the expected names, groups and ids.
- [x] Test: sweep job → name `sft1-<label>`, group `sft1-sweep-<ts>`, neither contains `mmtc`
      or `fr-en`; tags start with `["mmtc", "fr-en", …]`, with no duplicate `fr-en` when
      `wandb.tags` already has it.
- [x] Test: plain run → name == group == `sft1_lora`; no `sweep` tag; auto-tags present.
- [x] Test (R11): same `experiment`, `<ts>` and label, but `fr-en` vs `zh-en` → **different**
      run_ids; for mmtc/fr-en sweep jobs the id equals today's form.
- [x] `hfmt/test_dataset_inflation.py`: name the fixture `dataset_dir` `<tmp>/<dataset>/<src>-<trg>/`
      so the lang-pair check passes. Add tests for `dataset_name` derivation, for explicit
      `dataset_name` winning, and for the lang-pair mismatch error.
- [x] Run `pytest hfmt/test_dataset_inflation.py hfmt/test_run_label.py`; all pass.
  - *62 passed* across those two files plus `analysis/test_pull_wandb.py`, which gained three `--tag`
    tests (AND filter, no `$and` without tags, repeatable arg). There are also extra tests for a
    trailing slash on `dataset_dir` and for a missing `dataset_name`, where the lang pair is still
    tagged.

### I8. Verification
- [x] Compose check (the `compose()` API, as in G) for all 7 experiments with
      `experiment=<stem>` injected: `cfg.experiment` == file stem,
      `data.dataset_name == "mmtc"`, and `derive_run_identity` returns stem-only name and group
      plus `mmtc`/`fr-en` tags.
- [x] Grep: no `mmtc_fr-en_` remains in `egs/`, `hfmt/`, `conf/`, `tools/`, `README.md`, or
      `CLAUDE.md`, except the historical `analysis/` and earlier `features/` records. The
      `CLAUDE.md` analysis-dir example `analysis/2026-10-08_mmtc_fr-en_sft1/` is also fine:
      `analysis/` is flat, so its dir names need the context.
  - *Verified:* all 7 compose OK. `sft1_refine1` gives dir
    `egs/mmtc/fr-en/outputs/sft1_refine1/sweep-<ts>/…`, name
    `sft1_refine1-learning_rate-0.0003__seed-37`, group `sft1_refine1-sweep-<ts>`, tags
    `[mmtc, fr-en, sft, refine1, sweep, …]`, id `mmtc_fr-en_sft1_refine1_sweep-<ts>_…`.
    Run.sh's real override string (`+mmtc/fr-en=…,experiment=…,data.dataset_dir=…,output_root=…`)
    gives `_label_pairs` → `[]`, so a plain run is still not a sweep. `sweep_probe` still emits
    the refine1 axes (rc 0) and rejects `data.src=de` with a one-line error (rc 2).
    `sh -n egs/run.sh` passes.
  - Remaining `mmtc_fr-en_` hits are intentional: "never `mmtc_fr-en_…`" examples, the old file
    names of historical `conf/sweep/` presets in yaml comments, run-id expectations in tests, and
    a legacy `data=` group-choice label test.
- [ ] **Cluster dry run (deferred, like G):** `bash egs/run.sh mmtc/fr-en/sft1_refine1.yaml
      train.max_steps=20` → dirs under `egs/mmtc/fr-en/outputs/sft1_refine1/sweep-<ts>/…`; in
      W&B the names/group are stem-only and tags `mmtc` + `fr-en` filter them; a plain
      `sft1_lora` run is named/grouped `sft1_lora`.

### I9. Docs & decisions
- [x] `README.md`: replace the stale run-name example (`mmtc_fr-en_sft1-lora_r-16__…` →
      `sft1-lora_r-16__…`) and the `outputs/<experiment>/…` paths with
      `egs/<dataset>/<src>-<trg>/outputs/<stem>/…`. Say that the experiment name is the file stem
      (injected by run.sh, no `experiment:`/`wandb.group:` in the yaml). Add a short "Filtering
      runs in W&B" note (tags `mmtc`/`fr-en`, config `hfmt.data.dataset_name`, and
      `pull_wandb.py --tag`), including the R12 caveat that plain runs from different lang pairs
      can share a group.
- [x] `decisions.md`: record the naming rule ("a name holds only what its location doesn't
      already say"), decision 5a (bare-stem groups, rejected alternative), the hidden-id prefix
      (R11), and the one-time break with old `mmtc_fr-en_*` dirs, runs and groups (R13).
- [x] Fix stale launch examples: `CLAUDE.md` (`bash egs/run.sh mmtc_fr-en_sft1 …` →
      `mmtc/fr-en/sft1.yaml`), `conf/config.yaml:5`, `conf/hydra/launcher/{local,slurm}.yaml:2`
      (`+experiment=…` → `+mmtc/fr-en=sft1 experiment=sft1 data.dataset_dir=…`), and
      `tools/run_on_free_gpu.sh` usage comments.
- [x] Check off group I items here as they complete.
