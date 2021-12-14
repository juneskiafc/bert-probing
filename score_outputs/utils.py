# compute by-language probing scores.
from collections import defaultdict
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from conllu import parse_incr
import numpy as np
from datasets import Dataset, load_dataset
from argparse import ArgumentParser

FONTSIZE = 25

def save_heatmap(data, out_file):
    row_labels = list(range(1, 13))
    column_labels = list(range(1, 13))

    plt.figure(figsize=(14, 14))
    annot_kws = {
        "fontsize":FONTSIZE,
    }
    heatmap = sns.heatmap(
        data,
        cbar=False,
        annot=True,
        annot_kws=annot_kws,
        fmt=".2f",
        # xticklabels=column_labels,
        # yticklabels=row_labels,
        cmap='RdYlGn')

    heatmap.invert_yaxis()
    heatmap.set_yticklabels(row_labels, rotation=0, fontsize=FONTSIZE)
    heatmap.set_xticklabels(column_labels, rotation=0, fontsize=FONTSIZE)
    heatmap.set_xlabel('heads', fontsize=FONTSIZE)
    heatmap.set_ylabel('layers', fontsize=FONTSIZE)

    fig = heatmap.get_figure()
    fig.savefig(Path(out_file).with_suffix('.pdf'), bbox_inches='tight')

def get_lang_to_id(task):
    if task == 'POS':
        return get_lang_to_id_pos()
    elif task == 'NER':
        return get_lang_to_id_ner()
    elif task == 'PAWSX':
        return get_lang_to_id_pawsx()

def get_lang_to_id_pawsx():
    lang_to_id = defaultdict(list)

    i = 0
    for lang in ['en', 'fr', 'de', 'es']:
        dataset = load_dataset('paws-x', lang, split='test')
        lang_to_id[lang] = list(range(i, i + len(dataset)))
        i += len(dataset)
    
    return lang_to_id

def get_lang_to_id_pos():
    data_root = Path('/home/june/mt-dnn/experiments/POS/data')

    dataset_dirs = [
        data_root.joinpath('en/UD_English-EWT'),
        data_root.joinpath('fr/UD_French-FTB'),
        data_root.joinpath('de/UD_German-GSD'),
        data_root.joinpath('es/UD_Spanish-AnCora')
    ]

    langs = ['en', 'fr', 'de', 'es']
    lang_to_id = defaultdict(list)
    cumulative_sent_idx = 0

    for i, data_dir in enumerate(dataset_dirs):
        lang = langs[i]
        with open(data_dir.joinpath(f'test.conllu'), 'r', encoding='utf-8') as f:
            for _, _ in enumerate(parse_incr(f)):
                lang_to_id[lang].append(cumulative_sent_idx)
                cumulative_sent_idx += 1
    
    return lang_to_id

def get_lang_to_id_ner():
    root = 'experiments/NER/ner_test_tmp.json'
    df = Dataset.from_json(root)
    lang_to_id = defaultdict(list)

    for i, row in enumerate(df):
        lang = row['langs'][0]
        lang_to_id[lang].append(i)
    
    return lang_to_id

def get_final_probing_result(task, languages, do_individual=True, do_combined=True):
    def _get_individual_probing_result(setting, combined=False, lang=None):
        assert combined or (not combined and lang is not None)

        # read base.
        base_root = f'score_outputs/{task}/head_probe/base/xnli_base-{task.lower()}'
        setting_root = f'score_outputs/{task}/head_probe/{setting}/xnli_{setting}-{task.lower()}'
        out_file_root = f'score_outputs/{task}/head_probe/results/xnli_{setting}-{task.lower()}'

        if not combined:
            base_root += f'-{lang}-pure'
            setting_root += f'-{lang}-pure'
            out_file_root += f'-{lang}'
        else:
            base_root += '-combined'
            setting_root += f'-combined'
            out_file_root += f'-combined'
        
        out_file_root = Path(out_file_root)
        base_root = Path(base_root)
        setting_root = Path(setting_root)

        if out_file_root.with_suffix('.pdf').is_file() and out_file_root.with_suffix('.csv').is_file():
            return 
        else:
            out_file_root.parent.mkdir(parents=True, exist_ok=True)

            base_df = pd.read_csv(base_root.with_suffix('.csv'), index_col=0)
            setting_df = pd.read_csv(setting_root.with_suffix('.csv'), index_col=0)

            diff = setting_df - base_df
            diff.to_csv(out_file_root.with_suffix('.csv'))
            save_heatmap(diff, out_file_root)

    if do_individual:
        for lang in languages:
            for setting in ['cross', 'multi']:
                _get_individual_probing_result(setting, combined=False, lang=lang)
    
    if do_combined:
        for setting in ['cross', 'multi']:
            _get_individual_probing_result(setting, combined=True)
    

def get_lang_csvs(task, model, languages, do_individual=True, do_combined=True):
    def _get_acc_for_heads(labels, preds):
        results = np.zeros((12, 12))
        
        for hl in range(12):
            for hi in range(12):
                preds_for_head = preds.iloc[:, hl*12+hi]
                acc = accuracy_score(labels, preds_for_head)
                results[hl, hi] = acc
        
        return results
    
    root = f'score_outputs/{task}/head_probe/{model}/results.csv'
    data = pd.read_csv(root, index_col=0)
    
    if do_individual:
        language_to_ids = get_lang_to_id(task)

        # individual languages
        for lang in languages:
            out_file = f'score_outputs/{task}/head_probe/{model}/xnli_{model}-{task.lower()}-{lang}-pure.csv'
            if Path(out_file).is_file():
                continue
                
            ids = language_to_ids[lang]
            data_for_lang = data.iloc[ids, :]

            # compute acc
            labels = data_for_lang.iloc[:, 1]
            preds = data_for_lang.iloc[:, 2:]

            results = _get_acc_for_heads(labels, preds)
            
            out_df = pd.DataFrame(results)
            out_df.to_csv(out_file)
    
    # combined
    if do_combined:
        combined_out_file = f'score_outputs/{task}/head_probe/{model}/xnli_{model}-{task.lower()}-combined.csv'
        if not Path(combined_out_file).is_file():
            combined_labels = data.iloc[:, 1]
            combined_preds = data.iloc[:, 2:]

            combined_results = _get_acc_for_heads(combined_labels, combined_preds)
            out_df = pd.DataFrame(combined_results)
            out_df.to_csv(combined_out_file)
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--task', type=str)
    args = parser.parse_args()
    
    task = args.task
    if task != 'NLI':
        languages = ['en', 'fr', 'es', 'de']
    
    for model in ['cross', 'multi', 'base']:
        get_lang_csvs(task, model, languages)
    
    get_final_probing_result(task, languages)