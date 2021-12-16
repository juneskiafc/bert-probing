from typing import Tuple
import pandas as pd
from collections import Counter
from mlm.scorers import MLMScorerPT
from mlm.models import get_pretrained
from transformers import AutoTokenizer
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
from matplotlib import pyplot as plt 
from experiments.exp_def import TaskDefs
import seaborn as sns
from time import time
from experiments.exp_def import LingualSetting, Experiment
from argparse import ArgumentParser 

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
        results,
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

def mlm_score(
    ckpt_file,
    data_file,
    path_to_task_def, 
    prefix,
    scores_out_file,
    n_words,
    device_id):

    if Path(scores_out_file).is_file():
        print(f'PLL scores exists at {scores_out_file}')
        with open(scores_out_file, 'rb') as f:
            scores = pickle.load(f)

        _, _, tokenizer = get_pretrained([mx.gpu(device_id)], 'bert-base-multilingual-cased')
        dataset = construct_dataset(data_file, path_to_task_def, prefix, tokenizer)
        n_words = get_true_tok_lens_from_dataset(dataset)

    else:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
        vocab = tokenizer.vocab

        if ckpt_file is not None:
            print(f'loading ckpt from {ckpt_file}.')
            state_dict = torch.load(ckpt_file)
            model = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased')
            model.load_state_dict(state_dict, strict=True)
        else:
            model = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased')
        
        scorer = MLMScorerPT(
            model,
            vocab,
            tokenizer,
            ctxs=[mx.gpu(device_id)],
            device=device_id,
            tokenizer_fast=True
        )

        dataset = construct_dataset(data_file, path_to_task_def, prefix, tokenizer)

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

def main(task, model_name, model_ckpt, datasets, do_individual=True, do_combined=True, device_id=0):
    data_root = Path(f'experiments/{task.name}')
    mlm_scores_out_file = Path(f'mlm_scores/{task.name}/scores.npy')
    n_words_path = Path(f'mlm_scores/{task.name}/n_words.csv')

    # load n_words if it exists.
    # contains the number of words in each dataset.
    if n_words_path.is_file():
        n_words = pd.read_csv(n_words_path, index_col=0).to_dict()
    else:
        n_words = Counter()
    
    if mlm_scores_out_file.is_file():
        results = np.load(mlm_scores_out_file)
    else:
        if do_combined:
            results = np.zeros((len(datasets)+1,))
        else:
            results = np.zeros((len(datasets),))

        if do_individual:
            for i, dataset in enumerate(datasets):
                print(f'Evaluating mlm for {model_name} on {dataset}.')

                data_file = data_root.joinpath(f'{dataset}/bert-base-multilingual-cased/{dataset}_test.json')
                path_to_task_def = Path(f'experiments/MLM/{task.name}/task_def.yaml')
                scores_for_dataset_out_file = Path(f'mlm_scores/{task.name}').joinpath(model_name, f'{dataset}_scores.pkl')

                start = time()
                plls, pppl, n_words = mlm_score(
                    model_ckpt,
                    data_file,
                    path_to_task_def,
                    task.name.lower(),
                    scores_for_dataset_out_file,
                    n_words,
                    device_id)
                end = time() - start

                results[i] = pppl
                print(f'PPPL for {model_name}->{dataset}: {pppl}, in {end:.4f}s')

                # update n_words if we haven't saved it yet.
                if not n_words_path.is_file():
                    n_words[dataset] += n_words
            
            if not n_words_path.is_file():
                pd.DataFrame(n_words).to_csv(n_words_path)
        
        if do_combined:
            n_combined_words = sum(n_words)
            combined_scores = []

            for i, dataset in enumerate(datasets):
                scores_for_dataset_out_file = Path(f'mlm_scores/{task.name}').joinpath(model_name, f'{dataset}_scores.pkl')
                with open(scores_for_dataset_out_file, 'rb') as f:
                    scores_for_dataset = pickle.load(f)
                    combined_scores.extend(scores_for_dataset)
            
            # compute PPPL pseudo perplexity for corpus
            print('Scoring PPPL for combined...')
            combined_pppl = compute_pppl(combined_scores, n_combined_words)
            results[-1] = combined_pppl
        
        np.save(mlm_scores_out_file, results)
    
    if do_combined:
        yticklabels = [d.upper() for d in datasets] + 'combined'
    if do_individual:
        yticklabels = [d.upper() for d in datasets]

    create_heatmap(
        results,
        xticklabels=[model_name],
        yticklabels=yticklabels,
        xlabel='',
        ylabel='languages',
        out_file=mlm_scores_out_file.with_suffix('.pdf')
    )

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_ckpt', type=str)
    parser.add_argument('--task', type=str)
    args = parser.parse_args()

    task = Experiment[args.task]
    checkpoint_dir = Path('checkpoint/mlm_finetuned/huggingface/')
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

        model_ckpts = [
            checkpoint_dir.joinpath('nli_15lang_finetuned'),
            checkpoint_dir.joinpath('nli_4lang_finetuned')
        ]
        model_names = [
            'NLI_all-lang',
            'NLI_EN/FR/DE/ES'
        ]

        for i, model_ckpt in enumerate(model_ckpts):
            main(task, model_names[i], model_ckpt, datasets)
        
    else: 
        model_ckpt = checkpoint_dir.joinpath(f'mlm_{task.name.lower()}_finetuned')
        model_name = f'{task.name}_EN/FR/DE/ES'
        main(task, model_name, model_ckpt, ['en', 'fr', 'de', 'es'])