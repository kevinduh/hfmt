# 12_resume_sft — request

Fix the implementation of SFT resume training using `egs/launch.sh sft <run folder>/run_config.yaml`
-- which currently doesn't work. I'd clarify that resume training is meant only for jobs with
checkpoints cancelled midway and continues training using TRL's `resume_from_checkpoint`. It
shouldn't work with different run_names, output to different directories, etc.
