from collections import OrderedDict
from gluonnlp import vocab
from mxnet.gluon import data
import pandas as pd
from mlm.scorers import MLMScorerPT
from torch import nn
from mlm.models import get_pretrained
from transformers import AutoTokenizer
from mlm.loaders import Corpus
import mxnet as mx
from transformers import BertForMaskedLM
from pathlib import Path
import pickle
from mt_dnn.model import MTDNNModel
from experiments.exp_def import TaskDefs
import torch
from mxnet.gluon.data import SimpleDataset
import json
import numpy as np
from typing import List, Tuple
from math import exp
import tasks
from experiments.exp_def import TaskDefs
import seaborn as sns
from time import time
from module.san import MaskLmHeader

def load_data_file(path, path_to_task_def='', prefix='', maxlen=512):
    task_defs = TaskDefs(path_to_task_def)
    task_def = task_defs.get_task_def(prefix)

    with open(path, 'r', encoding='utf-8') as reader:
        data = []
        cnt = 0

        for line in reader:
            sample = json.loads(line)
            cnt += 1
            if len(sample['token_id']) < maxlen:
                data.append(sample)
    
    return data

def ids_to_masked(token_ids: np.ndarray, tokenizer) -> List[Tuple[np.ndarray, List[int]]]:
    # every word can be masked
    # We don't mask the [CLS], [SEP] for now for PLL
    # 101 is [CLS], 102 is [SEP]
    mask_indices = []
    for mask_pos in range(len(token_ids)):
        if token_ids[mask_pos] != 101 and token_ids[mask_pos] != 102:
            mask_indices.append(mask_pos)

    # get the id for [MASK]
    mask_token_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    token_ids_masked_list = []
    for mask_set in mask_indices:
        token_ids_masked = token_ids.copy()
        token_ids_masked[mask_set] = mask_token_id
        token_ids_masked_list.append((token_ids_masked, mask_set))

    return token_ids_masked_list

def compute_pppl(scores, num_words):
    return exp(-(sum(scores) / num_words))

def get_true_tok_lens_from_dataset(dataset):
    prev_sent_idx = None
    true_tok_lens = []
    for (curr_sent_idx, _, valid_length, _, _, _) in dataset:
        if curr_sent_idx != prev_sent_idx:
            prev_sent_idx = curr_sent_idx
            true_tok_lens.append(valid_length - 2)

    return sum(true_tok_lens)

def construct_dataset(data_file, path_to_task_def, prefix, tokenizer):
    print('loading data and constructing dataset...')
    data = load_data_file(data_file, path_to_task_def, prefix)

    sents_expanded = []
    for sent_idx, example in enumerate(data):
        token_ids = example['token_id']
        # token_type_ids = example['type_id']
        ids_masked = ids_to_masked(token_ids, tokenizer)
        for ids, mask_set in ids_masked:
            sents_expanded.append((sent_idx, ids, len(token_ids), mask_set, token_ids[mask_set], 1))

    print(f"{sent_idx + 1} sentences loaded.")
    dataset = SimpleDataset(sents_expanded)
    return dataset

def mlm_score(model_name, ckpt_file, data_file, path_to_task_def, prefix, scores_out_file, n_words, device_id, load_huggingface=True):
    if Path(scores_out_file).is_file():
        with open(scores_out_file, 'rb') as f:
            scores = pickle.load(f)
        # vocab_size = 119547 # bert base multilingual cased

        if n_words is None:
            _, _, tokenizer = get_pretrained([mx.gpu(device_id)], 'bert-base-multilingual-cased')
            dataset = construct_dataset(data_file, path_to_task_def, prefix, tokenizer)
            n_words = get_true_tok_lens_from_dataset(dataset)
    else:
        # data file is json.
        ctxs = [mx.gpu(device_id)] # or, e.g., [mx.gpu(0), mx.gpu(1)]

        tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
        vocab = tokenizer.vocab

        if ckpt_file is not None:
            print(f'loading ckpt from {ckpt_file}.')
            state_dict = torch.load(ckpt_file)
            if not load_huggingface:
                config = state_dict['config']

                # cuda settings.
                config["cuda"] = True
                config['device'] = device_id

                # task def init.
                task_defs = TaskDefs(path_to_task_def)
                task_def = task_defs.get_task_def(prefix)
                task_def_list = [task_def]
                config['task_def_list'] = task_def_list

                # temp fix
                config['fp16'] = False
                config['answer_opt'] = 0
                config['adv_train'] = False

                # don't need these
                del state_dict['optimizer']
                for param in ['scoring_list.0.weight', 'scoring_list.0.bias',
                            'pooler.dense.weight', 'pooler.dense.bias',
                            'scoring_list.0.decoder.weight']:
                    if param in state_dict['state']:
                        del state_dict['state'][param]

                model = MTDNNModel(config, device=config['device'])
                model.load_state_dict(state_dict, strict=True)
                model = model.network
            else:
                model = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased')
                # load huggingface weights
                model.load_state_dict(state_dict, strict=True)
        else:
            model = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased')
        
        scorer = MLMScorerPT(model, vocab, tokenizer, ctxs, device=device_id, tokenizer_fast=True)
        dataset = construct_dataset(data_file, path_to_task_def, prefix, tokenizer)

        print('scoring (pll)...')
        scores, true_token_lens = scorer.score(corpus=None, dataset=dataset, split_size=64) # PLL, returns list of PLLs for each sentence in dataset
        print(f'len(scores) = {len(scores)}, len(ttl) = {len(true_token_lens)}')
        n_words = sum(true_token_lens)

        print(f'saving at {scores_out_file}')
        Path(scores_out_file).parent.mkdir(exist_ok=True, parents=True)
        with open(scores_out_file, 'wb') as f:
            pickle.dump(scores, f)

    # compute PPPL pseudo perplexity for corpus
    print('scoring (pppl)...')
    pseudo_perplexity = compute_pppl(scores, n_words)
    
    return scores, pseudo_perplexity, n_words

if __name__ == '__main__':
    # corpuses to compute PPPLs for.
    datasets = [
        'multi-ar',
        'multi-bg',
        'multi-de',
        'multi-el',
        'multi-es',
        'multi-fr',
        'multi-hi',
        'multi-ru',
        'multi-sw',
        'multi-th',
        'multi-tr',
        'multi-ur',
        'multi-vi',
        'multi-zh',
        'multi-en'
    ]

    # settings under which BERT was trained to evaluate PPPL for each corpus.
    settings = [
        'cross',
        # 'multi',
        # 'base',
    ]

    results = np.zeros((len(datasets), len(settings)))
    data_root = Path('experiments/NLI')
    base_model_name = "bert-base-multilingual-cased"
    true_tok_lens_saved_file = data_root.joinpath('true_tok_lens.csv')
    mlm_scores_out_root = Path('mlm_scores/NLI')
    mlm_scores_out_file = mlm_scores_out_root.joinpath('mlm_all.npy')

    if mlm_scores_out_file.is_file():
        results = np.load(mlm_scores_out_file)
    else:
        new_n_words = {}
        if true_tok_lens_saved_file.is_file():
            true_tok_lens_saved = pd.read_csv(true_tok_lens_saved_file).to_dict()
            saved_n_words = len(true_tok_lens_saved.keys())
        else:
            true_tok_lens_saved = None
            saved_n_words = 0
        
        for i, dataset in enumerate(datasets):
            language = dataset.split("-")[1]
            data_file = data_root.joinpath(f'{language}/{base_model_name}/{language}_test.json')

            for j, setting in enumerate(settings):
                print(f'evaluating mlm for {setting} on {dataset}.')

                if setting != 'base':
                    prefix = setting
                    ckpt_file = f'checkpoint/mlm_finetuned/huggingface/{setting}/pytorch_model.bin'
                    path_to_task_def = Path('experiments/MLM').joinpath(f'{setting}/task_def.yaml')
                else:
                    path_to_task_def = Path('experiments/MLM').joinpath(f'base/task_def.yaml')
                    ckpt_file = None
                    prefix = 'base'

                scores_out_file = mlm_scores_out_root.joinpath(f'{setting}-{language}_scores.pkl')

                n_words_saved = true_tok_lens_saved is not None and dataset in true_tok_lens_saved
                if n_words_saved:
                    n_words = true_tok_lens_saved[language]
                else:
                    n_words = None

                start = time()
                plls, pppl, n_words = mlm_score(
                    setting,
                    ckpt_file,
                    data_file,
                    path_to_task_def,
                    prefix,
                    scores_out_file,
                    n_words,
                    device_id=2)
                end = time() - start

                results[i, j] = pppl
                print(f'pppl for {setting}-{language}: {pppl}, in {end:.4f}s')
                if not n_words_saved:
                    new_n_words[language] = n_words

        if len(new_n_words.keys()) > 0:
            if true_tok_lens_saved is not None:
                true_tok_lens_saved = {**true_tok_lens_saved, **new_n_words}
            else:
                true_tok_lens_saved = new_n_words
            
            index = list(range(len(true_tok_lens_saved.keys())))
            print(f'saving true_tok_lens at {true_tok_lens_saved_file}.')
            pd.DataFrame(true_tok_lens_saved, index=index).to_csv(true_tok_lens_saved_file)
        
        np.save(mlm_scores_out_file, results)
    
    heatmap = sns.heatmap(results, cmap='RdYlGn_r', xticklabels=settings, yticklabels=datasets, annot=True, fmt='.1f')
    heatmap.set_title('PPPL scores')
    heatmap.xaxis.tick_top()
    heatmap.xaxis.set_label_position('top')
    fig = heatmap.get_figure()
    fig.savefig(mlm_scores_out_file.with_suffix('.png'))