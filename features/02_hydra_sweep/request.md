I want to implement hydra's parameter sweep so we can experiment with multiple parameters.
I want to use the submitit package approach that we're currently using. Multiple slurm jobs should be sent out.
Job results are currently written to wandb and they should not clobber each other.
Enough information should be pushed to wandb so that run results are identifiable. I think this is already the case, but good to confirm.