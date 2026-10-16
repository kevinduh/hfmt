# Plan — 06_hydra_config_directory_hierarchy

## Goal

Reorganize the Hydra SFT workflow so an experiment carries its dataset + language-pair
context in the directory hierarchy. Concretely:

1. Move experiments from the flat `egs/experiment/<name>.yaml` into
   `egs/<dataset>/<src>-<trg>/<name>.yaml` (e.g. `egs/mmtc/fr-en/sft1_lora.yaml`).
2. Add a per-lang-pair `dataset.yaml` **template** next to the experiments, with
   `$DATA_ROOT` / `$SRC` / `$TRG` placeholders for train/dev/test paths.
3. Experiments declare only `data_root`, `src`, `trg` (+ instruction); the template is
   **inflated** into absolute paths at data-load time.
4. Remove the `conf/data/` config group entirely.
5. Write run outputs into `egs/<dataset>/<src>-<trg>/outputs/` instead of repo-root `outputs/`.

## Current state (what we're changing)

- **`conf/data/mmtc_fr-en.yaml`** (the group being removed) holds `src_lang`, `trg_lang`,
  `train_yaml` (a manifest path), `evalset` (a flat test-source path), `instruction`.
- **`DataConfig`** in `hfmt/hydra_config.py` is the typed schema for that group
  (`train_yaml`, `evalset` are `MISSING`).
- **`conf/config.yaml`** defaults `- data: mmtc_fr-en`, sets `output_root: outputs`, and
  composes `hydra.run.dir` / `hydra.sweep.dir` as `${output_root}/${experiment}/...`. It
  also sets `hydra.searchpath: [file://${HFMT_ROOT}/egs]` so `+experiment=<name>` resolves
  `egs/experiment/<name>.yaml`.
- **`hfmt/sft_translation.py`** `get_data()` (≈L398) reads `cfg.data.train_yaml`, loads
  `train`/`dev` src+trg via `load_dataset("text", ...)`. The test pass (≈L603) reads
  `cfg.data.evalset` (test **source** only). Reference (`test.trg`) is not consumed here —
  scoring is external (`analysis/run_sacrebleu.py`).
- **`egs/run.sh`** takes a flat `<experiment>` arg → `+experiment=<name>`, runs
  `--multirun`, and shells out to **`hfmt/sweep_probe.py`** which *also* composes the
  experiment via `+experiment=<name>` + the `egs/` searchpath.
- **7 experiment files** under `egs/experiment/` each do `- override /data: mmtc_fr-en`.
- **`.gitignore`** ignores repo-root `outputs/` only.

## Proposed design

### A. Directory layout
```
egs/
  mmtc/
    fr-en/
      dataset.yaml        # template (placeholders, committed)
      sft1.yaml           # experiment (was mmtc_fr-en_sft1.yaml)
      sft1_lora.yaml
      ...
      outputs/            # run outputs (gitignored)
```
Experiments drop the dataset/lang prefix from their filename (the directory now carries it):
`mmtc_fr-en_sft1.yaml` → `egs/mmtc/fr-en/sft1.yaml`.

### B. `dataset.yaml` template (verbatim placeholders, inflated in code)
```yaml
train:
  src: [$DATA_ROOT/train.$SRC-$TRG.$SRC]
  trg: [$DATA_ROOT/train.$SRC-$TRG.$TRG]
dev:
  src: [$DATA_ROOT/dev.$SRC-$TRG.$SRC]
  trg: [$DATA_ROOT/dev.$SRC-$TRG.$TRG]
test:
  src: [$DATA_ROOT/test.$SRC-$TRG.$SRC]
  trg: [$DATA_ROOT/test.$SRC-$TRG.$TRG]
```
> ⚠️ **Typo in the request's template** (lines 14–16): `train.trg` uses `.$SRC`; it should
> be `.$TRG` (otherwise src and trg point at the same file). Plan assumes the corrected form
> above — **please confirm**.

Inflation: read `dataset.yaml` as text, substitute with `string.Template` /
`str.replace` using `DATA_ROOT`, `SRC`, `TRG` from the resolved config, then `yaml.safe_load`.
`$DATA_ROOT` syntax is shell-style (not Hydra `${...}`), so `dataset.yaml` stays a plain
template, never a Hydra config — consistent with "inflated during dataset creation." The
inflated `train`/`dev`/`test` src+trg path lists are written back into `cfg.data` (Section D),
so the resolved config logs show real paths (paths are W&B-safe; only *content* is forbidden).

**Test split is optional.** `test.src`/`test.trg` may resolve to files that don't exist for a
given dataset. After inflation, if `test` is absent or its `src` file(s) don't exist on disk,
**skip the final test-inference pass** (don't error). `train`/`dev` remain required — `dev`
drives `eval_loss`, early stopping, and best-model selection *during* training; `test` only
feeds the optional end-of-run held-out inference.

### C. Experiment `.yaml` changes
- Remove `- override /data: ...` from `defaults`.
- Add inline data context. Proposed schema block (see D):
  ```yaml
  data:
    data_root: /exp/kduh/data/mt/mmtc    # absolute path (never committed data content)
    src: fr
    trg: en
    instruction: "Translate French to English"
    dataset_file: dataset.yaml           # NAME of the template file (default); must include .yaml
  ```
- **`dataset_file`** names the template *file* (basename **with** the `.yaml` suffix), default
  `dataset.yaml`, overridable per experiment / on the CLI. The file is resolved **relative to
  the experiment's own directory** — the directory (`data.dataset_dir`) is **injected by
  `run.sh`** from `dirname` of the arg (`${HFMT_ROOT}/egs/<group>`), so the experiment never
  repeats its own location. (A raw `python -m …` run must pass `data.dataset_dir=…` itself.)
- **Validation (required):** `os.path.join(dataset_dir, dataset_file)` **must exist** — fail
  fast with a clear error if the named template isn't in the experiment's directory (see R10).

### D. Schema (`hfmt/hydra_config.py`) — repurpose `DataConfig` (settled)
`DataConfig` is repurposed to hold the **source/target train/dev/test file paths** — the same
structure the template captures — plus the inflation inputs. Proposed shape:
```python
@dataclass
class DataSplit:
    """Source/target file lists for one split (post-inflation absolute paths)."""
    src: List[str] = field(default_factory=list)
    trg: List[str] = field(default_factory=list)

@dataclass
class DataConfig:
    src: str = MISSING            # source language code, e.g. "fr"  (was src_lang)
    trg: str = MISSING            # target language code, e.g. "en"  (was trg_lang)
    data_root: str = MISSING      # absolute dir where the data FILES live (inflation input)
    dataset_file: str = "dataset.yaml"  # template file NAME (basename, incl. .yaml suffix)
    dataset_dir: str = MISSING    # experiment's own dir (injected by run.sh); holds dataset_file
    instruction: str = ""
    # Filled in code by inflating dataset.yaml with {data_root, src, trg}:
    train: DataSplit = field(default_factory=DataSplit)   # required
    dev: DataSplit = field(default_factory=DataSplit)     # required
    test: Optional[DataSplit] = None                      # optional (skip inference if missing)
```
- **Removed:** `train_yaml`, `evalset` (terminology retired — see note below).
- `train`/`dev`/`test` are declared in the schema (so they're typed and appear in the logged
  resolved config) and **populated in code** after compose by the inflation step.
- The `data: DataConfig` field stays on `Config`. This is *schema*, not a `conf/data/` group
  file, so it satisfies "remove the conf/data group" while keeping type validation.

> **Terminology:** the name **`evalset` is retired**. The end-of-run held-out inference now
> reads **`cfg.data.test.src`** (with `test.trg` as the external-scoring reference). Rename the
> corresponding code locals (`eval_data` → `test_data`, `eval.pred.trg` output stays as the
> dump filename unless you want it renamed too).

### E. `conf/` changes
- **Delete** `conf/data/` (the `mmtc_fr-en.yaml` group file).
- **`conf/config.yaml`:** remove the `- data: mmtc_fr-en` default line. Keep `base_config`.
- Decide experiment addressing + output root (below).

### F. Experiment addressing (run.sh + sweep_probe.py) — **chosen approach (F3)**
`+experiment=<name>` works today only because the files live in a directory literally
named `experiment` on the searchpath. Moving them to `egs/<dataset>/<src>-<trg>/` breaks that
one group name — but we don't need a fixed group. **Pass the experiment's path relative to
`egs/` (the actual `.yaml` file) as the sole positional arg** and let Hydra's nested-group
selection do the rest, with the existing searchpath untouched:
```
bash egs/run.sh mmtc/fr-en/sft1.yaml
#   group=$(dirname $1)            = mmtc/fr-en
#   name=$(basename $1 .yaml)      = sft1        # trailing .yaml stripped for the Hydra option
python -m hfmt.sft_translation --multirun \
    +mmtc/fr-en=sft1 \
    data.dataset_dir=${HFMT_ROOT}/egs/mmtc/fr-en \
    output_root=egs/mmtc/fr-en/outputs ...
```
(The template *file* name comes from the experiment's `data.dataset_file` (default
`dataset.yaml`); `run.sh` only supplies the *directory*.)
`run.sh` strips a trailing `.yaml` if present, so `mmtc/fr-en/sft1` is accepted too. The arg
points at the real file (tab-completable); Hydra's `+<group>=<option>` just needs the
extensionless option name.
Why this is the simplest option:
- **Searchpath is unchanged.** `conf/config.yaml` keeps `hydra.searchpath:
  [file://${HFMT_ROOT}/egs]`. Relative to that root, `mmtc/fr-en` *is* a valid nested Hydra
  config group and `sft1` its option, so `+mmtc/fr-en=sft1` is a standard nested-group
  selection (the experiment's `# @package _global_` still applies). No per-run searchpath
  rewrite.
- **run.sh stays trivial and depth-agnostic.** Just `dirname`/`basename` of one arg — no
  hard-coded dataset/langpair/exp parsing; works for any nesting.
- **`dataset.yaml` path + `output_root` derive from `dirname`** (`${HFMT_ROOT}/egs/<group>/
  dataset.yaml`, `egs/<group>/outputs`). `dataset.yaml` is an inert sibling Hydra never loads.
- **`sweep_probe.py` gets the identical change** (compose with `+<group>=<name>`,
  unchanged searchpath).

**The one real code change (small, pattern-consistent):** the selector key is now
`mmtc/fr-en`, which contains a `/`. `_label_pairs` (`sft_translation.py:201`) already drops
path-ish tokens via `if key.startswith("data.") or "/" in val`; add the symmetric guard
`or "/" in key` so any-depth group selector is filtered out of run labels and the `is_sweep`
check generically. Precedent: `hydra/launcher` (also a slash key) is already handled one line
up by the `hydra` prefix check, so slash-keyed group overrides are known to flow through here.

**Caveat (validate, not a blocker):** the static `hydra.job.config.override_dirname.
exclude_keys` list can't name the dynamic `mmtc/fr-en` key, so the *raw*
`${hydra.job.override_dirname}` will still contain `+mmtc/fr-en=sft1`. That's fine — its only
consumer is the `hfmt_runlabel` resolver (→ `_label_pairs`, which now filters it); nothing
uses the raw string as a path (`hydra.run.dir`/`sweep.dir` use `${now:}` + the resolver). The
stale `experiment` entry in `exclude_keys` can be left as a harmless no-op.

### G. Outputs (request §OUTPUTS)
Set the run/sweep dirs under the lang-pair dir. `run.sh` passes
`output_root=egs/<dataset>/<src>-<trg>/outputs`; `conf/config.yaml`'s existing
`hydra.run.dir` / `hydra.sweep.dir` templates (`${output_root}/${experiment}/...`) then
produce `egs/mmtc/fr-en/outputs/<experiment>/<ts>/...`. No change to the W&B↔output-dir
alignment logic. **Confidentiality:** these dirs hold `*.pred` / `eval.pred.trg` dumps —
add `egs/**/outputs/` to `.gitignore` (see Risk R4).

## Affected files
- `hfmt/hydra_config.py` — repurpose `DataConfig` (D).
- `hfmt/sft_translation.py` — resolve+validate the template path
  (`join(data.dataset_dir, data.dataset_file)` must exist, R10); inflate it into
  `cfg.data.train/dev/test` (early in `main()`); `get_data()` reads `cfg.data.train/dev`; final
  inference reads `cfg.data.test.src` **guarded by existence** (skip when `test` is None or its
  file(s) are absent); rename `eval_data`→`test_data`, retire `evalset`; add `"/" in key` guard
  to `_label_pairs` (drops the slashed group selector).
- `hfmt/sweep_probe.py` — compose with `+<group>=<name>` (unchanged searchpath); it already
  composes the full experiment, so it's the natural place for the **login-node fast-fail**
  template-existence check too.
- `conf/config.yaml` — drop `- data:` default. Searchpath stays as-is; `experiment` entry in
  `exclude_keys` becomes a harmless no-op.
- `conf/data/mmtc_fr-en.yaml` — **delete**.
- `egs/run.sh` — single `egs`-relative `.yaml` path arg → `group=dirname`,
  `name=basename …  .yaml` (strip extension); emit `+<group>=<name>` and inject
  `data.dataset_dir=${HFMT_ROOT}/egs/<group>` + `output_root=egs/<group>/outputs`.
- `egs/experiment/*.yaml` (7 files) — **move** to `egs/<dataset>/<src>-<trg>/`, strip
  `override /data`, add inline `data:` block.
- `egs/<dataset>/<src>-<trg>/dataset.yaml` — **new** template(s).
- `.gitignore` — add `egs/**/outputs/`.
- Docs: `README.md` (§"conf/data", §"Add an experiment"), `decisions.md` (record the
  hierarchy + inflation decision), `overview.md` if it describes the data group.

## Risks & open questions

- **R1 — Hydra group addressing — VALIDATED (2026-10-16 compose spike).** `+mmtc/fr-en=spike`
  resolved a nested `# @package _global_` file at `egs/mmtc/fr-en/spike.yaml` through the
  unchanged `hydra.searchpath: [file://$HFMT_ROOT/egs]` (experiment name + `train.seed`
  applied, all `override /…` picks composed). The `"/" in key` guard is **confirmed
  necessary**: current `_label_pairs("+mmtc/fr-en=spike,train.seed=37,lora.lora_r=8")` →
  `['mmtc-fr-en-spike','seed-37','lora_r-8']` (selector leaks); with the guard →
  `['seed-37','lora_r-8']`. Residual risk now near-zero.
  - *Spike note:* the Hydra **1.3.7 argparse CLI crashes on this box's Python 3.14**
    (`LazyCompletionHelp` / "badly formed help string"), so `--cfg job` is unusable here;
    the spike used the `compose()` API (as `sweep_probe.py` does). Flag for the cluster env:
    confirm the real run uses a supported Python, or the CLI launch path needs attention
    independent of this feature.
- **R2 — resolved.** Repurpose `DataConfig` (keep the dataclass + `data:` field; delete only
  the `conf/data/` group). It now holds `src`/`trg`/`data_root`/`dataset_file`/`dataset_dir`/
  `instruction` + the inflated `train`/`dev`/`test` splits.
- **R2b — OmegaConf writeback (impl risk).** The config is confirmed **struct-mode** (spike:
  `data.dataset_dir=/foo` → `ConfigCompositionException: Could not override 'data.dataset_dir'`),
  so every field we read/write must exist in the schema — hence `dataset_dir`/`dataset_file`/
  `train`/`dev`/`test` must be declared (good: validation has teeth). Writing inflated paths
  into `cfg.data.train/dev` is fine (declared fields); for `test: Optional[DataSplit] = None`,
  assigning a populated split may need `OmegaConf.update(cfg, "data.test", {...})` or a brief
  `open_dict`. If fussy, keep the inflated structure local and read `test` existence from it.
- **R3 — resolved.** Experiment names the template *file* (`data.dataset_file`, default
  `dataset.yaml`); `run.sh` injects the *directory* (`data.dataset_dir`). A bare `python -m`
  run must pass `data.dataset_dir`.
- **R10 — template-existence validation (required behavior).** `join(dataset_dir,
  dataset_file)` must exist in the experiment's directory, else fail fast with a clear message.
  Put the check both at the start of `main()` and (for login-node fast-fail before the heavy
  submit) in the `sweep_probe.py` compose that `run.sh` already runs. Edge cases to cover:
  `dataset_file` given without `.yaml`, an absolute path sneaking in, or `dataset_dir` unset on
  a raw `python -m` run — all should yield an explicit error, not a deep stack trace.
- **R4 — Outputs now live inside `egs/` (tracked tree).** Run dirs contain prediction dumps
  (data-derived). Must gitignore `egs/**/outputs/` *before* any run, or proprietary
  hypotheses risk being committed. Also confirm nothing globs `egs/**` into packaging.
- **R5 — resolved.** `evalset` terminology retired → final inference uses `cfg.data.test.src`;
  `test.trg` is reference-only (external scoring). **Test is optional:** skip the inference
  pass when `test` is unset or `test.src` file(s) don't exist (existence check at runtime).
- **R6 — Template typo** (`train.trg` → `.$TRG`). Assuming corrected. **→ confirm.**
- **R7 — Breaking change / migration.** All 7 existing experiments move and change shape;
  anyone with muscle-memory `bash egs/run.sh mmtc_fr-en_sft1` must switch to the new
  address. One-time migration; note in `decisions.md`. The untracked old shell recipes under
  `egs/mmtc/*/` (`sft1.sh`, `inf.sh`) are unrelated legacy scripts — leave as-is unless asked.
- **R8 — `load_dataset("text", data_files=...)` expects lists.** The template values are
  single-element lists already; keep them lists through inflation so the existing loader is
  unchanged.
- **R9 — Multiple lang-pairs / datasets.** Design should generalize to `egs/mmtc/de-en/`,
  `egs/mmtc/zh-en/` with their own `dataset.yaml`. F1/F2 both handle this; only `mmtc/fr-en`
  gets fully populated now unless you want more scaffolded.

## Open decisions
1. ~~**Addressing**~~ — **settled: F3** (pass the `egs`-relative `.yaml`; `+<group>=<name>` on
   the unchanged searchpath). Literal `egs/mmtc/fr-en/` layout, least machinery.
2. ~~**`DataConfig`**~~ — **settled: repurpose** to hold src/trg train/dev/test paths (+
   `data_root`/`dataset_yaml`/`instruction`); delete only the `conf/data/` group.
4. ~~**Test data**~~ — **settled:** `evalset` retired → `cfg.data.test.src`; `test`/`test.trg`
   optional; skip inference when the test source is missing.

3. ~~**`dataset_yaml` path**~~ — **settled:** experiment sets `data.dataset_file` (name, default
   `dataset.yaml`); `run.sh` injects `data.dataset_dir`; the resolved file must exist in the
   experiment's directory (validated, R10).

All design decisions are settled — ready to generate `todo.md` on your go.

## Validation (once decisions land)
- Compose spike: `--cfg job` prints resolved config with inflated train/dev/test paths; no
  `conf/data` reference remains.
- `get_data()` unit check against a tiny fixture `dataset.yaml` + `egs/data/example.*` files,
  asserting inflated absolute paths and that src≠trg.
- Dry single run (`sweep.enabled=false`) and a 2-axis sweep land under
  `egs/<dataset>/<src>-<trg>/outputs/...`; W&B run name == run-dir basename still holds.
- `git status` shows no `egs/**/outputs/` tracked; grep for stale `conf/data` / `train_yaml`
  / `evalset` references returns nothing but intended code.
