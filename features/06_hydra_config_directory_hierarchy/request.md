We want to update our hydra configuration in that we want more information to be embedded in the experiment hiearchy.
Currently we have experiment yamls in egs/experiment. This can be confusing and cluttered when we're workign with different datasets and languages.

## DIRECTORY HIERARCHY
We want to move to the following: egs/$DATASET_NAME/$SRC_TARGET_LANGUAGE. an example would be `egs/mmtc/fr-en` or `egs/mmtc/de-en` where `mmtc` is the name of our dataset.

## DATASET.YAML
In egs/$DATASET_NAME/$SRC_TRG_LANGUAGE there will be a file titled `dataset.yaml`. This file will contain a template for where the train, dev and test datasets live.
Here is the template:

```
train:
  src:
    - $DATA_ROOT/train.$SRC-$TRG.$SRC
  trg:
    - $DATA_ROOT/train.$SRC-$TRG.$SRC
dev:
  src:
    - $DATA_ROOT/dev.$SRC-$TRG.$SRC
  trg:
    - $DATA_ROOT/dev.$SRC-$TRG.$TRG
test:
  src:
    - $DATA_ROOT/test.$SRC-$TRG.$SRC
  trg:
    - $DATA_ROOT/test.$SRC-$TRG.$TRG
```

## EXPERIMENT.YAML
We want to update our experiment.yaml file structure (a current example: in egs/experiment/mmtc_fr-en_sft1.yaml) with the following changes:

- We want to remove all of the conf/data files, data structures, and mentions.
- We want to capture the variables in our dataset.yaml template:
    * $DATA_ROOT: the absolute path where the dataset lives
    * $SRC: the source language
    * $TRG: the target language

During dataset creation the dataset template should be inflated with the variables in the experiment.yaml fiel to give us the absolute paths of our datasets.

## REMOVING THE HYDRA CONF/DATA
Since we're going to be embedding the dataset information directly in the experiment yaml, we can remove all hydra configs related to data and datasets.

## OUTPUTS
We want the outputs directory to save results to a directory hierarchy similar to experiments.
E.g if we're running an experiment in `egs/mmtc/fr-en/sft1_lora.yaml`, we want the results to be saved to `egs/mmtc/fr-en/outputs/`.