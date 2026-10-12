We want to update the hydra configuration structure.
We want the hydra conf directory to only include base level experimental components such as:
* data: where the data that we are training and evaluating on lives
* decode: the decoding parameters for the model
* model: this should be renamed to basemodel and this should contain all the parameters for the basemodel that we are finetuning
* train: should contain the general training parameters
* lora: should contain parameters related to finetuning with lora

The goal is that each of these configs can be mixed and match in our experimental config. These will essentially be atomistic components.

We want to merge the current sweep and wandb configurations into the experiment config. We want to move the experiment config into the egs directory. The goal is for the experiment config to be a collection of atomistic config components. We also should be able to "turn off" lora during training to fully finetune the basemodel.

Is this possible? Will hydra be able to find it our can we just update our egs/run.sh script?

We want to be able to turn sweeps on and off.
We also want to be able to sweep across model configs. For example if we have a model config for qwen-7b and qwen-2.5b, we want an experiment to be able to sweep across both of those entire configs.