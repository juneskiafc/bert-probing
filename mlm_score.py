from typing import List, Tuple

from time import time
from argparse import ArgumentParser 
from pathlib import Path
from collections import Counter
import pickle
from math import exp
import json
from matplotlib import pyplot as plt 

import seaborn as sns
import pandas as pd
import numpy as np

import torch
from transformers import AutoTokenizer, BertForMaskedLM
import mxnet as mx
from mxnet.gluon.data import SimpleDataset

from experiments.exp_def import TaskDefs
from experiments.exp_def import Experiment
from mlm.scorers import MLMScorerPT
from mlm.models import get_pretrained

def create_heatmap(
    results: np.ndarray,
    xticklabels: List[str],
    yticklabels: List[str],
    xlabel: str,
    ylabel: str,
    out_file: str,
    figsize: Tuple[int, int] = (14, 14),
    fontsize: int = 20,
    cmap: str = 'RdYlGn_r'
    ):

    plt.figure(figsize=figsize)
    annot_kws = {
        "fontsize":fontsize,
    }

    heatmap = sns.heatmap(
        np.reshape(results, (-1, 1)),
        cmap=cmap,
        cbar=False,
        annot_kws=annot_kws,
        annot=True,
        fmt='.2f')
    
    heatmap.set_xticklabels(xticklabels, rotation=0, fontsize=fontsize)
    heatmap.set_yticklabels(yticklabels, rotation=0, fontsize=fontsize)
    heatmap.set_xlabel(xlabel, fontsize=fontsize)
    heatmap.set_ylabel(ylabel, fontsize=fontsize)

    heatmap.xaxis.tick_top()
    heatmap.xaxis.set_label_position('top')

    fig = heatmap.get_figure()
    fig.savefig(out_file, bbox_inches='tight')

def load_data_file(path, maxlen=512):
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

def construct_dataset(data_file, tokenizer):
    print('loading data and constructing dataset...')
    data = load_data_file(data_file)

    sents_expanded = []
    for sent_idx, example in enumerate(data):
        token_ids = example['token_id']
        ids_masked = ids_to_masked(token_ids, tokenizer)
        for ids, mask_set in ids_masked:
            sents_expanded.append((sent_idx, ids, len(token_ids), mask_set, token_ids[mask_set], 1))

    print(f"{sent_idx + 1} sentences loaded.")
    dataset = SimpleDataset(sents_expanded)
    return dataset

def mlm_score(
    state_dict,
    is_huggingface_ckpt,
    data_file,
    scores_out_file,
    device_id):

    if Path(scores_out_file).is_file():
        print(f'PLL scores exists at {scores_out_file}')
        with open(scores_out_file, 'rb') as f:
            scores = pickle.load(f)

        _, _, tokenizer = get_pretrained([mx.gpu(device_id)], 'bert-base-multilingual-cased')
        dataset = construct_dataset(data_file, tokenizer)
        n_words = get_true_tok_lens_from_dataset(dataset)

    else:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
        vocab = tokenizer.vocab
        model = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased')

        if state_dict is not None:
            if not is_huggingface_ckpt and "cls.predictions.bias" not in state_dict:
                og_state_dict = model.state_dict()
                for param in [
                        "cls.predictions.bias",
                        "cls.predictions.transform.dense.weight",
                        "cls.predictions.transform.dense.bias", 
                        "cls.predictions.transform.LayerNorm.weight",
                        "cls.predictions.transform.LayerNorm.bias",
                        "cls.predictions.decoder.weight",
                        "cls.predictions.decoder.bias"
                    ]:
                        state_dict[param] = og_state_dict[param]

            model.load_state_dict(state_dict, strict=True)
        
        scorer = MLMScorerPT(
            model,
            vocab,
            tokenizer,
            ctxs=[mx.gpu(device_id)],
            device=device_id,
            tokenizer_fast=True
        )

        dataset = construct_dataset(data_file, tokenizer)

        # PLL, returns list of PLLs, one for each sentence in dataset
        print('Scoring PLL...')
        scores, true_token_lens = scorer.score(corpus=None, dataset=dataset, split_size=64)

        print(f'len(scores) = {len(scores)}, len(ttl) = {len(true_token_lens)}')
        n_words = sum(true_token_lens)

        print(f'Saving PLL scores at {scores_out_file}.')
        Path(scores_out_file).parent.mkdir(exist_ok=True, parents=True)
        with open(scores_out_file, 'wb') as f:
            pickle.dump(scores, f)

    # compute PPPL pseudo perplexity for corpus
    print('Scoring PPPL...')
    pseudo_perplexity = compute_pppl(scores, n_words)
    
    return scores, pseudo_perplexity, n_words

def main(
    task,
    model_name,
    model_ckpt,
    is_huggingface_ckpt,
    datasets,
    do_individual=True,
    do_combined=True,
    device_id=0):

    data_root = Path(f'experiments/{task.name}')
    mlm_scores_out_file = Path(f'mlm_scores/{task.name}/{model_name}/scores.npy')
    n_words_path = Path(f'mlm_scores/{task.name}/n_words.csv')

    # load n_words if it exists.
    # contains the number of words in each dataset.
    if n_words_path.is_file():
        n_words = pd.read_csv(n_words_path)
        n_words = {n_words.iloc[i, 0]: n_words.iloc[i, 1] for i in range(len(n_words))}
        n_words = Counter(n_words)
    else:
        n_words = Counter()
    
    if mlm_scores_out_file.is_file():
        results = np.load(mlm_scores_out_file)
    else:
        if do_combined:
            results = np.zeros((len(datasets)+1,))
        else:
            results = np.zeros((len(datasets),))
        
        # load model.
        print(f'loading ckpt from {model_ckpt}.')
        state_dict = torch.load(model_ckpt)
        if not is_huggingface_ckpt:
            if 'optimizer' in state_dict:
                del state_dict['optimizer']
                del state_dict['config']
                state_dict = state_dict['state']

                for param in [
                    "scoring_list.0.weight",
                    "scoring_list.0.bias",
                    "pooler.dense.weight",
                    "pooler.dense.bias",
                    "bert.pooler.dense.weight",
                    "bert.pooler.dense.bias"
                ]:
                    del state_dict[param]

        if do_individual:
            for i, dataset in enumerate(datasets):
                print(f'Evaluating mlm for {model_name} on {dataset}.')

                data_file = data_root.joinpath(f'{dataset}/bert-base-multilingual-cased/{task.name.lower()}_test.json')
                scores_for_dataset_out_file = Path(f'mlm_scores/{task.name}').joinpath(model_name, f'{dataset}_scores.pkl')

                start = time()
                _, pppl, nwords = mlm_score(
                    state_dict,
                    is_huggingface_ckpt,
                    data_file,
                    scores_for_dataset_out_file,
                    device_id)
                end = time() - start

                results[i] = pppl
                print(f'\nPPPL for {model_name}->{dataset}: {pppl}, in {end:.4f}s\n')

                # update n_words if we haven't saved it yet.
                if not n_words_path.is_file():
                    n_words[dataset] += nwords
            
            if not n_words_path.is_file():
                index = list(n_words.keys())
                values = [n_words[i] for i in index]
                pd.DataFrame(values, index=index).to_csv(n_words_path)
        
        if do_combined:
            n_combined_words = sum(n_words.values())
            combined_scores = []

            for k, v in n_words.items():
                print(f'{k}: {v}')
            print(f'combined: {n_combined_words}')

            for i, dataset in enumerate(datasets):
                scores_for_dataset_out_file = Path(f'mlm_scores').joinpath(
                    task.name,
                    model_name,
                    f'{dataset}_scores.pkl')
                
                with open(scores_for_dataset_out_file, 'rb') as f:
                    scores_for_dataset = pickle.load(f)
                    combined_scores.extend(scores_for_dataset)
            
            # compute PPPL pseudo perplexity for corpus
            print('Scoring PPPL for combined...')
            combined_pppl = compute_pppl(combined_scores, n_combined_words)
            print(f'\nPPPL for {model_name}->combined: {combined_pppl}\n')
            results[-1] = combined_pppl
        
        np.save(mlm_scores_out_file, results)
    
    if do_individual:
        yticklabels = [d.upper() for d in datasets]
    if do_combined:
        yticklabels = [d.upper() for d in datasets] + ['combined']

    create_heatmap(
        results,
        xticklabels=[model_name],
        yticklabels=yticklabels,
        xlabel='',
        ylabel='languages',
        out_file=mlm_scores_out_file.with_suffix('.pdf')
    )

def combine_split_scores(dataset):
    all_scores = []
    for score_pkl in Path('mlm_scores/MARC/MARC_EN-FR-DE-ES').rglob(f'{dataset}_*_scores.pkl'):
        with open(score_pkl, 'rb') as f:
            score_for_split = pickle.load(f)
        all_scores.extend(score_for_split)
    
    out_file = Path('mlm_scores/MARC/MARC_EN-FR-DE-ES').joinpath(f'{dataset}_scores.pkl')
    with open(out_file, 'wb') as f:
        pickle.dump(all_scores, f)
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_ckpt', type=str)
    parser.add_argument('--huggingface_ckpt', action='store_true')
    parser.add_argument('--task', type=str)
    parser.add_argument('--device_id', type=int, default=0)
    args = parser.parse_args()

    task = Experiment[args.task]
    checkpoint_dir = Path(f'checkpoint/mlm_finetuned/{task.name}/')
    base_model_name = "bert-base-multilingual-cased"

    if task is Experiment['NLI']:
        datasets = [
            'ar',
            'bg',
            'de',
            'el',
            'es',
            'fr',
            'hi',
            'ru',
            'sw',
            'th',
            'tr',
            'ur',
            'vi',
            'zh',
            'en'
        ]
        main(
            task,
            'NLI_all-lang',
            checkpoint_dir.joinpath('15lang/model_5_294528.pt'),
            args.huggingface_ckpt,
            datasets,
            device_id=args.device_id
        )

        datasets = [
            'de',
            'es',
            'fr',
            'en'
        ]
        main(
            task, 
            'NLI_EN-FR-DE-ES',
            checkpoint_dir.joinpath('4lang/model_5_294528.pt'),
            args.huggingface_ckpt,
            datasets,
            device_id=args.device_id
        )
        
    else: 
        model_name = f'{task.name}_EN-FR-DE-ES'
        main(
            task,
            model_name,
            args.model_ckpt,
            args.huggingface_ckpt,
            ['en', 'es', 'fr', 'de'],
            device_id=args.device_id
        )