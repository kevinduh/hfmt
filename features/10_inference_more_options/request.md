I'd like to add more options for the inference code (inf_translation.py)

1. Currently "launch.sh inf" takes a run_config which means run inference on a trained model. I'd like to add an option "launch.sh inf experiment.yaml" which means do inference on the vanilla checkpoint, using settings in experiment.yaml. This allows us to compare zero-shot inference of downloaded base models without training.
2. I want to allow dataset.yaml to work with multiple test files. For multiple train/dev files, they are concatenated by the sft_translation.py script and treated as one blob. However, for test files, I want inference to run on each src file independently, and do bleu calculation on trg if provided. This will be useful if we want to measure translation accuracy on multiple distinct test sets. 
3. Not strictly an inference thing, but I want to implement extra script for merging the model/ of sft_translation.py result as model_merged (in the same folder) and then allow either further rounds of SFT using model_merged/ as a starting checkpoint. The inference code to run of the model/ of the 1st-round SFT and 2nd-round SFT should work the same.
4. I imagine code will be added to grid.py and launch.sh. Those are getting difficult to understand, so please add comments to explain. 
5. Running inference in MT_callback is expensive. Add a hydra option train.inference_callback_interval (int: default 1). If this value is N, then inference will run every N x train.eval_steps. I imagine changing on_evaluate() code so that the expensive part only runs at that interval, but the super().on_evaluate(args, state, control, **kwargs) is run everytime.


