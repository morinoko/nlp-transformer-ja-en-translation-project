# NLP Project: Using Transformers for Japanese-English Translation

This repository contains the scripts (and outputs) for evaluating four transformer models for Japanese-English translation, as well as a fine-tuning experiment using the Helsinki Opus model. 

The models evaluated are:
- [Helsinki OPUS JA-EN (Helsinki-NLP/opus-mt-ja-en)](https://huggingface.co/Helsinki-NLP/opus-mt-ja-en)
- [ElanMT (Mitsua/elan-mt-bt-ja-en)](https://huggingface.co/Mitsua/elan-mt-bt-ja-en)
- [FuguMT (staka/fugumt-ja-en)](https://huggingface.co/staka/fugumt-ja-en)
- [MBart JA-EN (ken11/mbart-ja-en)](https://huggingface.co/ken11/mbart-ja-en)

The following three datasets were used for evaluation:
- [Tatoeba JA-EN](https://huggingface.co/datasets/Helsinki-NLP/tatoeba)
- [Kyoto Free Translation Task (KFTT)](https://huggingface.co/datasets/Hoshikuzu/KFTT)
- [Custom-built Ruby Dataset](https://huggingface.co/datasets/morinoko-inari/ruby-rails-ja-en)

## Directory Structure

- `model_evaluation/`: Contains scripts and output files from model evaluation
  - `scripts/`: Contains the scripts for evaluating each model.
  - `script_outputs/`: Contains text files of the outputs from the evaluation scripts, including the BLEU and chrF scores for each model and translation samples. Each text file is named after the model it corresponds to.
  - `translation_output_csvs/`: Contains CSVs for each model with the machine translation outputs, including the original Japanese sentences, the machine translation outputs, the references, and the dataset from which each sentence came. Each CSV is named after the model it corresponds to.
- `fine_tuning_experiment/`: Contains the fine-tuning experiment scripts and output files
  - `scripts/`: Contains the scripts for fine-tuning the Helsinki Opus model.
  - `script_outputs/`: Contains text files of the outputs from the fine-tuning scripts, including the BLEU and chrF scores for each model and translation samples.
- `ruby_dataset/`: Contains the Ruby dataset used for training and evaluation
  - `data.json`: The full dataset, including synthetic and real data
  - `synthetic.json`: Dataset with only the synthetic data
- `report_and_presentation_slides/`: Contains the project report and presentation slides. Link to the presentation video is in the README in this directory.

## Running Evaluation Scripts

Each model has its own evaluation script, which evaluates the model using the BLEU and chrF metrics based on 100 randomly selected sentences from each dataset.

Results of the evaluation will be printed to the console and a CSV of the test translations will be saved in the `model_evaluation/translation_output_csvs/` directory.

To save the output of the evaluation to a text file instead of printing to the screen, redirect the output to a file using `>` in the command line. For example:

```bash
python ./model_evaluation/scripts/model_helsinki.py > `output.txt`
```

Outputs are already saved in the `model_evaluation/script_outputs/` directory if you wish to inspect them.

### Helsinki Opus MT Model Evaluation

```bash
python ./model_evaluation/scripts/model_helsinki.py
```

### Mitsua Elan MT Model Evaluation

```bash
python ./model_evaluation/scripts/model_mitsua_elan.py
```

### Staka Fugu MT Model Evaluation

```bash
python ./model_evaluation/scripts/model_staka_fugu.py
```

### MBart JA-EN Model Evaluation

```bash
python ./model_evaluation/scripts/model_mbart_ja_en.py
```

### Ruby Fine-tuned Version of Helsinki Opus Model Evaluation

This is a version of the Helsinki Opus model that has been fine-tuned on the entire Ruby dataset.
```bash
python ./model_evaluation/scripts/model_helsinki_fine_tuned.py
```

## Running the Fine-tuning Experiment Scripts

The fine-tuning experiment scripts are used to train the Helsinki Opus model on the Ruby dataset. The training process is done using the Hugging Face `transformers` library.

The output of the fine-tuning process will be printed to the console, but if you do not wish to run the script, you can inspect the output files in the `fine_tuning_experiment/script_outputs/` directory:
- `output_training_helsinki.txt`: Contains the output from fine-tuning the model on the entire Ruby dataset
- `output_training_helsinki_experiment.txt`: Contains the output from the fine-tuning experiment, where four versions of the Helsinki Opus model were trained.

For fine-tuning the Helsinki Opus model on the **entire** Ruby dataset:
```bash
python ./fine_tuning_experiment/scripts/training_helsinki.py
```

For running the fine-tuning experiment, in which four versions of the Helsinki Opus model were training on the Ruby dataset, each with an increasing number of training samples:
```bash
python ./fine_tuning_experiment/scripts/training_helsinki_experiment.py
```

For you reference, here are the BLEU and chrF scores from the fine-tuning experiment (based on an exerpt from the output file):
```
-----------------------------------------
Fine-tuned Model Evaluation scores vs original model
-----------------------------------------
BLEU - Original Ruby: 0.250
chrF - Original Ruby: 59.38

BLEU - Fine-tuned Ruby 100 samples: 0.275
chrF - Fine-tuned Ruby 100 samples: 60.47

BLEU - Fine-tuned Ruby 200 samples: 0.317
chrF - Fine-tuned Ruby 200 samples: 62.37

BLEU - Fine-tuned Ruby 300 samples: 0.326
chrF - Fine-tuned Ruby 300 samples: 63.31

BLEU - Fine-tuned Ruby all samples: 0.337
chrF - Fine-tuned Ruby all samples: 63.32
```
