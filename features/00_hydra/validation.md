# On-cluster validation checklist — 00_hydra

Everything in this feature was validated on a CPU box (config composition, schema, sbatch
rendering, offline W&B). The items below **require a GPU + the full env on the Slurm cluster**
and must be run by a human. Check off as you confirm each.

## 1. Environment
- [ ] Build/refresh the conda env with the new deps: `bash install/install_hf_default.sh`
      (or `pip install hydra-core hydra-submitit-launcher hydra-colorlog` into the `hfmt` env).
- [ ] `python install/check_versions.py` prints hydra/omegaconf/submitit alongside the rest,
      with no import errors.
- [ ] `export HFMT_ROOT=$(pwd)` and confirm the fr-en data referenced by
      `conf/data/mmtc_fr-en.yaml` exists (`train.yaml`, `test.fr-en.fr`).

## 2. Config composition (on the cluster)
- [ ] `python hfmt/sft_translation.py +experiment=mmtc_fr-en_sft1 --cfg job` prints the
      resolved config (seed 37, lr 2e-4, r=8/qv, data paths resolved via HFMT_ROOT).
- [ ] `python hfmt/sft_translation.py +experiment=mmtc_fr-en_sft1 hydra/launcher=slurm --cfg hydra`
      shows the submitit `SlurmLauncher` with the expected partition/gres/time/setup.

## 3. Quick GPU smoke (in-process, short)
On a GPU node (interactive or `hydra/launcher=local`), run a truncated job to shake out the
real deps (torch/bitsandbytes/trl) end to end:
- [ ] `python hfmt/sft_translation.py +experiment=mmtc_fr-en_sft1 train.max_steps=20 train.eval_steps=10`
- [ ] Training starts, the eval callback decodes the dev set and logs BLEU/CHRF/TER.
- [ ] Outputs appear under `outputs/mmtc_fr-en_sft1/<timestamp>/` (checkpoints, `*.pred`,
      `eval.pred.trg`, the adapter at `<outdir>_b`).

## 4. Slurm submission (the real path)
- [ ] From a **login node**: `bash egs/run.sh mmtc_fr-en_sft1 train.max_steps=50`
      (do NOT `sbatch` it — submitit submits).
- [ ] A Slurm job is queued (`squeue -u $USER`); the generated sbatch lives under
      `outputs/mmtc_fr-en_sft1/.submitit/`.
- [ ] The job's log shows `install/path.sh` ran on the node (conda activated, CUDA modules
      loaded) **before** Python — i.e. the R2 bootstrap works.
- [ ] `HFMT_ROOT` resolved correctly inside the job (data paths found). If not, ensure it is
      exported at submit time (Slurm `--export=ALL` should carry it).

## 5. W&B linkage & confidentiality
- [ ] A W&B run appears in project `hfmt`, `group=mmtc_fr-en_sft1`, name `mmtc_fr-en_sft1-<...>`.
- [ ] `wandb.config` contains `run_dir` (the on-disk output dir) and the `hfmt` config;
      spot-check that `data.train_yaml`/`evalset` are **paths, not content**.
- [ ] The run logs metrics only (loss, lr, eval/bleu, eval/chrf, eval/ter) — **no** sample
      text, tables, or dataset-bearing artifacts, and **no** model artifact upload.

## 6. Requeue / resume (optional but recommended)
- [ ] Trigger a requeue (preempt or `scontrol requeue <jobid>`); confirm the job resumes into
      the **same** output dir and the **same** W&B run (not a duplicate).

## 7. Sign-off
- [ ] Full run (`bash egs/run.sh mmtc_fr-en_sft1`) completes and reproduces the pre-Hydra
      baseline numbers within noise (earlier fr-en run peaked ~64 BLEU).
- [ ] Record any surprises / config fixes back in `conf/` and note them in `decisions.md`.
