from datasets import load_dataset, concatenate_datasets, Dataset
from pathlib import Path
from collections import OrderedDict
import numpy as np
import csv

label_map = {
    'O': 0,
    'B-PER': 1,
    'I-PER': 2,
    'B-ORG': 3,
    'I-ORG': 4,
    'B-LOC': 5,
    'I-LOC': 6
}
label_map = {v:k for k,v in label_map.items()}

def _prepare_data(out_dir, langs_by_split):
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ['train', 'test']:
        if split not in langs_by_split:
            continue
    
        final_out_file = out_dir.joinpath(f'ner_{split}.tsv')
        out_file = out_dir.joinpath(f'ner_{split}_tmp.json')

        if Path(final_out_file).is_file():
            continue
        
        if not Path(out_file).is_file():
            datasets = []
            for lang in langs_by_split[split]:
                dataset = load_dataset('wikiann', lang, split=split)
                datasets.append(dataset)

            dataset = concatenate_datasets(datasets)
            dataset.to_json(out_file)
        
        # load and save as tsv in mtdnn format
        df = Dataset.from_json(str(out_file))
        with open(final_out_file, 'w') as f:
            for i, row in enumerate(df):
                premise = ' '.join(row['tokens'])
                label = label_map[int(row['ner_tags'][0])]
                f.write(f'{i}\t{label}\t{premise}\n')

def prepare_head_probe_data():
    out_dir = Path(f'experiments/NER/head_probe/')
    langs_by_split = {
        'train': ['en'],
        'test': ['en', 'fr', 'de', 'es']
    }

    _prepare_data(out_dir, langs_by_split)

def prepare_finetune_data():
    for setting in ['cross', 'multi']:
        out_dir = Path(f'experiments/NER/{setting}')
        langs_by_split = {
            'train': ['en'],
            'test': ['en', 'fr', 'de', 'es']
        }
        if setting == 'multi':
            langs_by_split['train'].extend(['fr', 'de', 'es'])
        
        _prepare_data(out_dir, langs_by_split)

def prepare_per_language_test_data():
    for lang in ['en', 'fr', 'de', 'es']:
        out_dir = Path(f'experiments/MLM/NER/{lang}')
        langs_by_split = {
            'test': [lang]
        }        
        _prepare_data(out_dir, langs_by_split)

def subsample_and_combine(foreign_dataset, ps):
    def read_rows(filename):
        with open(filename, 'r') as f:
            rows = []
            for row in f:
                id_, label, premise = row.split("\t")
                premise = premise.strip('\n')
                rows.append(OrderedDict({'id': id_, 'label': label, 'premise': premise}))
        return rows

    fieldnames = ['id', 'label', 'premise']
    mnli_rows = read_rows('experiments/NER/cross/ner_train.tsv')

    seeds = [list(range(500, 900, 100)), list(range(900, 1300, 100)), list(range(1300, 1700, 100))]
    rows = read_rows(foreign_dataset)
    for i, seed_collection in enumerate(seeds):
        for p_idx, p in enumerate(ps):
            np.random.seed(seed_collection[p_idx])
            subsampled_idxs = np.random.choice(
                np.arange(len(rows)),
                size=int(len(rows)*p),
                replace=False)
            subsampled_rows = [rows[i] for i in subsampled_idxs]

            out_file = Path(f'experiments/NER/foreign_{p}_{i}/ner_train.tsv')
            out_file.parent.mkdir(parents=True, exist_ok=True)

            with open(out_file, 'w') as fw:
                writer = csv.DictWriter(fw, fieldnames, delimiter='\t')
                for row in subsampled_rows:
                    writer.writerow(row)
            
                for r in mnli_rows:
                    writer.writerow(r)
if __name__ == '__main__':
    foreign_dataset = 'experiments/NER/foreign/ner_train.tsv'
    subsample_and_combine(foreign_dataset, [0.2, 0.4, 0.6, 0.8])

    # out_dir = Path(f'experiments/NER/foreign')
    # langs = {'train': ['fr', 'de', 'es'], 'test': ['fr', 'de', 'es']}
    # _prepare_data(out_dir, langs)

    # out_dir = Path(f'experiments/NER/en')
    # langs = {'train': ['en'], 'test': ['en']}
    # _prepare_data(out_dir, langs)



