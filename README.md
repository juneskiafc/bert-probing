# Installation
1. Download Miniconda and create an environment with python 3.8.
2. pip install -r requirements.txt
3. Replace transformer source code with ones provided in bert_src:
    site-packages/transformers/models/bert/modelling_bert.py
    site-packages/transformers/modelling-outputs.py
    site-packages/transformers/trainer.py

# Data Prep
For each dataset, the script that generates the train/test tsv files is in {dataset_dir}/prepare_data.py. First go to this file, and for the \__main__ method, make sure it will only run the methods you want. For most datasets, methods for making specific train/test files exist - for example, generating train/test datasets for each language in a list of languages (__make_per_language()__), or generating them for all languages (__make_multilingual()__).
* The languages that these methods will consider are currently hardcoded, so make sure those are the ones you want and add/remove languages if you need to.

After you have done this, run the script. For example, to make the train/test tsv files for the NER dataset, run 
```console
python experiments/NER/prepare_data.py
```
This will generate the tsv files in the correct format for your task. The task for the dataset is specified in the task_def.yaml file, under the key *data_format*. Make sure you have this yaml file, or make one if one doesn't exist (take a look at the ones that do for a hint on how to make one). Then, you can run prepro_std.py. Carrying on with the NER example, run
```console
python prepro_std.py --dataset NER --task_def path/to/task_def/yaml/file
```
This by default creates json data files that BERT can consume. If you want data files that XLM-R can consume, add *--model xlm-roberta-base* to the command above (these model names correspond to the ones registred on huggingface).
# To train a model
Take a look at train.py. The arguments are listed in at the very top as parser arguments; set the ones that are under the comment \#SET THESE that you need to set. Don't worry about the ones under the comment \# DON"T NEED THESE. For example, to finetune a mBERT model on the NER task with NER/en/ner_train.json on GPU 0:
```console
python train.py --devices 0 --exp_name NER-en --dataset_name NER/en
```
To do the same but with XLM-R, do
```console
python train.py --devices 0 --exp_name NER-en --dataset_name NER/en --bert_model_type xlm-roberta-base
```
# Gradient Probing
Gradient probing means performing multiple forward passes thorugh the network, accumulating the individual gradients obtained during each forward pass, and then using the normalized accumulated gradients as a measure to compare the relative importance of each transformer cell in performing the task for that dataset.

To gradient-probe a fine-tuned model saved in checkpoint/NER-el/model_5_15000.pt on GPU 0 for the NER task for the language el, run
```console
python gradient_probing.py --model_ckpt checkpoint/NER-el/model_5_15000.pt --finetuned_task NER --finetuned_setting el --downstream_task NER --downstream_setting el  --device_id 0
```

