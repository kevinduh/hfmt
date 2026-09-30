We want to update this repo so the hydra launcher can be used to run experiments.
We will need to update dependencies. Set up conda and pull dependencies if needed. You do not have access to a gpu, but the machine where this code runs will, so ensure all dependencies needed for running with a gpu are accounted for.
The goal of adding hydra is so that we can more easily account for all tweakable parameters and isolate them to configuration
files instead of in code.
We will be running everything via slurm so ensure that the solutions are slurm compatible.
We will be using wandb to log results so ensure there is some way to clearly link experiments to their output.
Our goal is to perform parameters sweeps but we will focus on this in a future request.