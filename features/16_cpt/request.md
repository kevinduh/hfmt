Create a continued pre-training (CPT) script that is analagous to hfmt/sft_translation.py
This script hfmt/cpt_monotext_bitext.py has the ability to train on both monolingual text and bilingual text.
The DataConfig in hfmt/hydra_config should be extended with an optional monotext field which is a list of filepaths to text files. The script will take in options to train just on monotext files, just on bitext, or on both. Additionally, for bitext files, there's an option that specifies whether src and trg side should be treated as indepent sentences or concatenated line-by-line. All these extra options should be optional as to not interfere with the standard sft_translation.py runs.

The training itself should use SFTTrainer. Use packing=True and metric_for_best_model=eval_loss. We won't be using chat templates, even though we will likely be training on Instruct models. The usage scenario is to first run cpt_monotext_bitext.py, save the model, and then trun sft_translation.py.

The current launch.sh script and job.sh will need to be updated too to allow for cpt. 