# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import yaml
import os
import numpy as np
import argparse
import json
import sys
from data_utils import load_data
from data_utils.task_def import TaskType, DataFormat
from data_utils.log_wrapper import create_logger
from experiments.exp_def import TaskDefs
from transformers import AutoTokenizer

DEBUG_MODE = False
MAX_SEQ_LEN = 512
DOC_STRIDE = 180
MAX_QUERY_LEN = 64
MRC_MAX_SEQ_LEN = 384

logger = create_logger(
    __name__,
    to_disk=True,
    log_file='mt_dnn_data_proc_{}.log'.format(MAX_SEQ_LEN))

def feature_extractor(tokenizer, text_a, text_b=None, max_length=512, do_padding=False):
    inputs = tokenizer(
        text_a,
        text_b,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding=do_padding
    )
    input_ids = inputs["input_ids"]
    token_type_ids = inputs["token_type_ids"] if "token_type_ids" in inputs else [0] * len(input_ids)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    attention_mask = inputs["attention_mask"]
    if do_padding:
        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)
    return input_ids, attention_mask, token_type_ids

def build_data(data, dump_path, tokenizer, data_format=DataFormat.PremiseOnly,
               max_seq_len=MAX_SEQ_LEN, lab_dict=None, do_padding=False, truncation=True):
    def build_data_premise_only(
            data, dump_path, max_seq_len=MAX_SEQ_LEN, tokenizer=None):
        """Build data of single sentence tasks
        """
        with open(dump_path, 'w', encoding='utf-8') as writer:
            for idx, sample in enumerate(data):
                ids = sample['uid']
                premise = sample['premise']
                label = sample['label']
                input_ids, input_mask, type_ids = feature_extractor(tokenizer, premise, max_length=max_seq_len)
                features = {
                    'uid': ids,
                    'label': label,
                    'token_id': input_ids,
                    'type_id': type_ids,
                    'attention_mask': input_mask}
                writer.write('{}\n'.format(json.dumps(features)))

    def build_data_premise_and_one_hypo(
            data, dump_path, max_seq_len=MAX_SEQ_LEN, tokenizer=None):
        """Build data of sentence pair tasks
        """
        with open(dump_path, 'w', encoding='utf-8') as writer:
            for idx, sample in enumerate(data):
                ids = sample['uid']
                premise = sample['premise']
                hypothesis = sample['hypothesis']
                label = sample['label']
                input_ids, input_mask, type_ids = feature_extractor(tokenizer, premise, text_b=hypothesis, max_length=max_seq_len)
                features = {
                    'uid': ids,
                    'label': label,
                    'token_id': input_ids,
                    'type_id': type_ids,
                    'attention_mask': input_mask}
                writer.write('{}\n'.format(json.dumps(features)))

    def build_data_premise_and_multi_hypo(
            data, dump_path, max_seq_len=MAX_SEQ_LEN, tokenizer=None):
        """Build QNLI as a pair-wise ranking task
        """
        with open(dump_path, 'w', encoding='utf-8') as writer:
            for idx, sample in enumerate(data):
                ids = sample['uid']
                premise = sample['premise']
                hypothesis_list = sample['hypothesis']
                label = sample['label']
                input_ids_list = []
                type_ids_list = []
                attention_mask_list = []
                for hypothesis in hypothesis_list:
                    input_ids, input_mask, type_ids = feature_extractor(tokenizer,
                                                                        premise, hypothesis, max_length=max_seq_len)
                    input_ids_list.append(input_ids)
                    type_ids_list.append(type_ids)
                    attention_mask_list.append(input_mask)
                features = {
                    'uid': ids,
                    'label': label,
                    'token_id': input_ids_list,
                    'type_id': type_ids_list,
                    'ruid': sample['ruid'],
                    'olabel': sample['olabel'],
                    'attention_mask': attention_mask_list}
                writer.write('{}\n'.format(json.dumps(features)))

    def build_data_sequence(data, dump_path, max_seq_len=MAX_SEQ_LEN, tokenizer=None, label_mapper=None):
        with open(dump_path, 'w', encoding='utf-8') as writer:
            for idx, sample in enumerate(data):
                ids = sample['uid']
                premise = sample['premise']
                tokens = []
                labels = []
                for i, word in enumerate(premise):
                    subwords = tokenizer.tokenize(word)
                    tokens.extend(subwords)
                    for j in range(len(subwords)):
                        if j == 0:
                            label = label_mapper[sample['label']]
                            raise ValueError(label)
                            labels.append(label)
                        else:
                            labels.append(label_mapper['X'])
                if len(premise) >  max_seq_len - 2:
                    tokens = tokens[:max_seq_len - 2]
                    labels = labels[:max_seq_len - 2]

                label = [label_mapper['CLS']] + labels + [label_mapper['SEP']]
                input_ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + tokens + [tokenizer.sep_token])
                assert len(label) == len(input_ids)
                type_ids = [0] * len(input_ids)
                features = {'uid': ids, 'label': label, 'token_id': input_ids, 'type_id': type_ids}
                writer.write('{}\n'.format(json.dumps(features)))

    if data_format == DataFormat.PremiseOnly:
        build_data_premise_only(
            data,
            dump_path,
            max_seq_len,
            tokenizer)
    elif data_format == DataFormat.PremiseAndOneHypothesis:
        build_data_premise_and_one_hypo(
            data, dump_path, max_seq_len, tokenizer)
    elif data_format == DataFormat.PremiseAndMultiHypothesis:
        build_data_premise_and_multi_hypo(
            data, dump_path, max_seq_len, tokenizer)
    elif data_format == DataFormat.Seqence:
        build_data_sequence(data, dump_path, max_seq_len, tokenizer, lab_dict)
    elif data_format == DataFormat.MRC:
        pass
        # build_data_mrc(data, dump_path, max_seq_len, tokenizer)
    else:
        raise ValueError(data_format)

def route_by_dataset(args):
<<<<<<< HEAD
    prepare_data(args, args.dataset)
=======
    if args.dataset == 'POS':
        pos_main(args)
    else:
        raise NotImplementedError
>>>>>>> f18cf1b5d82862452f86f02f278673d80d36cca4

def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocessing GLUE/SNLI/SciTail dataset.')
    parser.add_argument('--model', type=str, default='bert-base-multilingual-cased',
                        help='support all BERT and ROBERTA family supported by HuggingFace Transformers')
    parser.add_argument('--do_lower_case', action='store_true')
    parser.add_argument('--do_padding', action='store_true')
    parser.add_argument('--root_dir', type=str, default='data/canonical_data')
    parser.add_argument('--head_probe', action='store_true')
    parser.add_argument('--dataset', type=str)

    args = parser.parse_args()
    return args

def build_data_from_task_defs(task_defs, root, mt_dnn_root, tokenizer):
    for task in task_defs.get_task_names():
        task_def = task_defs.get_task_def(task)
        logger.info("Task %s" % task)
        for split_name in task_def.split_names:
            file_path = os.path.join(root, "%s_%s.tsv" % (task, split_name))
            if not os.path.exists(file_path):
                logger.warning(f"File {file_path} doesnot exit")
                sys.exit(1)
            rows = load_data(file_path, task_def)
            dump_path = os.path.join(mt_dnn_root, "%s_%s.json" % (task, split_name))
            logger.info(dump_path)
            build_data(
                rows,
                dump_path,
                tokenizer,
                task_def.data_type,
                lab_dict=task_def.label_vocab)

<<<<<<< HEAD
def prepare_data(args, task_name):
    def _build_huggingface_data_from_root(root):
=======
def pos_main(args):
    def _pos_prepare_data(root):
>>>>>>> f18cf1b5d82862452f86f02f278673d80d36cca4
        task_def = os.path.join(root, 'task_def.yaml')

        mt_dnn_root = os.path.join(root, args.model)
        if not os.path.isdir(mt_dnn_root):
            os.makedirs(mt_dnn_root)
        
        task_defs = TaskDefs(task_def)
        build_data_from_task_defs(task_defs, root, mt_dnn_root, tokenizer)
<<<<<<< HEAD

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.head_probe:
        root = f'/home/june/mt-dnn/experiments/{task_name}/head_probe'
        _build_huggingface_data_from_root(root)
    
    else:
        for setting in ['cross', 'multi']:
            root = f'/home/june/mt-dnn/experiments/{task_name}/{setting}'
            _build_huggingface_data_from_root(root)
=======
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.head_probe:
        root = f'/home/june/mt-dnn/experiments/POS/head_probe'
        _pos_prepare_data(root)
    
    else:
        for setting in ['cross', 'multi']:
            root = f'/home/june/mt-dnn/experiments/POS/{setting}'
            _pos_prepare_data(root)

def nli_main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # 'ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh'
    datasets = ['ltr']
    rootr = '/home/june/mt-dnn/experiments/attention-probing/'

    for dataset in datasets:
        root = rootr + f'multi-{dataset}'
        task_def = root + '/task_def.yaml'

        mt_dnn_root = os.path.join(root, args.model)
        if not os.path.isdir(mt_dnn_root):
            os.makedirs(mt_dnn_root)

        task_defs = TaskDefs(task_def)
        build_data_from_task_defs(task_defs, root, mt_dnn_root, tokenizer)  

def ner_main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    root = 'experiments/NER/'
    task_def = root + 'task_def.yaml'

    mt_dnn_root = os.path.join(root, args.model)
    if not os.path.isdir(mt_dnn_root):
        os.makedirs(mt_dnn_root)

    task_defs = TaskDefs(task_def)
    build_data_from_task_defs(task_defs, root, mt_dnn_root, tokenizer)

def marc_main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    root = 'experiments/MARC/'
    task_def = root + 'task_def.yaml'

    mt_dnn_root = os.path.join(root, args.model)
    if not os.path.isdir(mt_dnn_root):
        os.makedirs(mt_dnn_root)

    task_defs = TaskDefs(task_def)
    build_data_from_task_defs(task_defs, root, mt_dnn_root, tokenizer)

def pawsx_main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    root = 'experiments/PAWSX/'
    task_def = root + 'task_def.yaml'

    mt_dnn_root = os.path.join(root, args.model)
    if not os.path.isdir(mt_dnn_root):
        os.makedirs(mt_dnn_root)

    task_defs = TaskDefs(task_def)
    build_data_from_task_defs(task_defs, root, mt_dnn_root, tokenizer)
>>>>>>> f18cf1b5d82862452f86f02f278673d80d36cca4

if __name__ == '__main__':
    args = parse_args()
    route_by_dataset(args)