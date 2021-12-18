#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, CTRL, BERT, RoBERTa, XLNet).
GPT, GPT-2 and CTRL are fine-tuned using a causal language modeling (CLM) loss. BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss. XLNet is fine-tuned using a permutation language modeling (PLM) loss.
"""


import logging
import os
from dataclasses import dataclass, field
from typing import Optional
import torch

import transformers
from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import is_main_process


logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default='bert-base-multilingual-cased',
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default='bert',
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    mtdnn_checkpoint: Optional[str] = field(
        default=''
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    train_data_files: Optional[str] = field(
        default=None,
        metadata={
            "help": "The input training data files (multiple files in glob format). "
            "Very often splitting large files to smaller files can prevent tokenizer going out of memory"
        },
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    train_ref_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input train ref data file for whole word mask in Chinese."},
    )
    eval_ref_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input eval ref data file for whole word mask in Chinese."},
    )
    line_by_line: bool = field(
        default=True,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )

    mlm: bool = field(
        default=True, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    whole_word_mask: bool = field(default=False, metadata={"help": "Whether ot not to use whole word mask."})
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    plm_probability: float = field(
        default=1 / 6,
        metadata={
            "help": "Ratio of length of a span of masked tokens to surrounding context length for permutation language modeling."
        },
    )
    max_span_length: int = field(
        default=5, metadata={"help": "Maximum length of a span of masked tokens for permutation language modeling."}
    )
    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    device_id: str = field(default='0')
    setting: str = field(default='cross')

def get_dataset(
    args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizer,
    evaluate: bool = False,
    cache_dir: Optional[str] = None,
    ):
    def _dataset(file_path, ref_path=None):
        return LineByLineTextDataset(
            tokenizer=tokenizer,
            file_path=file_path,
            block_size=args.block_size)

    return _dataset(args.train_data_file, args.train_ref_file)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # env variables
    os.environ['WANDB_PROJECT'] = 'soroush'

    training_args.disable_tqdm = True
    training_args.num_train_epochs = 6
    training_args.do_train = True
    training_args.output_dir = training_args.output_dir + f"/mlm_finetuned/huggingface/{training_args.run_name}/"

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased', cache_dir=model_args.cache_dir)

    model = AutoModelWithLMHead.from_pretrained(
        model_args.model_name_or_path,
        from_tf=False,
        config=config,
        cache_dir=model_args.cache_dir,
    )
    
    if model_args.mtdnn_checkpoint != '':
        mtdnn = torch.load(model_args.mtdnn_checkpoint, map_location=f'cuda:{data_args.device_id}')['state']
        # these are leftovers from NLI finetuning, get rid
        for param in [
            'bert.pooler.dense.weight',
            'bert.pooler.dense.bias',
            'scoring_list.0.weight',
            'scoring_list.0.bias',
            'pooler.dense.weight',
            'pooler.dense.bias']:
            del mtdnn[param]
        
        # init MLM head with pretrained BERT params
        for param in [
            "cls.predictions.bias",
            "cls.predictions.transform.dense.weight",
            "cls.predictions.transform.dense.bias",
            "cls.predictions.transform.LayerNorm.weight",
            "cls.predictions.transform.LayerNorm.bias",
            "cls.predictions.decoder.weight",
            "cls.predictions.decoder.bias"]:
            mtdnn[param] = model.state_dict()[param]

        # freeze everything except MLM head
        model.load_state_dict(mtdnn, strict=True)
        for n, p in model.named_parameters():
            if 'cls' not in n:
                p.requires_grad = False

        model.resize_token_embeddings(len(tokenizer))

    # set max seq len = max for BERT
    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.model_max_length
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Get datasets
    # only train
    train_dataset = (
        get_dataset(
            data_args,
            tokenizer,
            model_args.cache_dir) if training_args.do_train else None
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=data_args.mlm,
        mlm_probability=data_args.mlm_probability
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=None
    )

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.train(model_path=model_path)
        trainer.save_model()

if __name__ == "__main__":
    main()