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

def get_final_probing_result(task, languages):
    for lang in languages:
        cross = pd.read_csv(f'score_outputs/{task}/head_probe/cross/xnli_cross-{task.lower()}-{lang}-pure.csv', index_col=0)
        multi = pd.read_csv(f'score_outputs/{task}/head_probe/multi/xnli_multi-{task.lower()}-{lang}-pure.csv', index_col=0)
        base = pd.read_csv(f'score_outputs/{task}/head_probe/base/xnli_base-{task.lower()}-{lang}-pure.csv', index_col=0)

        cross = cross - base
        multi = multi - base

        cross.to_csv(f'score_outputs/{task}/head_probe/xnli_cross-{task.lower()}-{lang}.csv')
        multi.to_csv(f'score_outputs/{task}/head_probe/xnli_multi-{task.lower()}-{lang}.csv')
        save_heatmap(cross, f'score_outputs/{task}/head_probe/xnli_cross-{task.lower()}-{lang}')
        save_heatmap(multi, f'score_outputs/{task}/head_probe/xnli_multi-{task.lower()}-{lang}')

def get_lang_csvs(task, model, languages):
    root = f'score_outputs/{task}/head_probe/{model}/results.csv'

    data = pd.read_csv(root, index_col=0)
    language_to_ids = get_lang_to_id(task)

    # load up language_to_ids.
    for lang in languages:
        ids = language_to_ids[lang]
        data_for_lang = data.iloc[ids, :]

        # compute acc
        labels = data_for_lang.iloc[:, 1]
        preds = data_for_lang.iloc[:, 2:]
        results = np.zeros((12, 12))
        
        for hl in range(12):
            for hi in range(12):
                preds_for_head = preds.iloc[:, hl*12+hi]
                acc = accuracy_score(labels, preds_for_head)
                results[hl, hi] = acc
        
        out_file = f'score_outputs/{task}/head_probe/{model}/xnli_{model}-{task.lower()}-{lang}-pure.csv'
        out_df = pd.DataFrame(results)
        out_df.to_csv(out_file)
    
if __name__ == '__main__':
    task = 'PAWSX'
    if task != 'NLI':
        languages = ['en', 'fr', 'es', 'de']
    
    for model in ['cross', 'multi', 'base']:
        get_lang_csvs(task, model, languages)
    
    get_final_probing_result(task, languages)