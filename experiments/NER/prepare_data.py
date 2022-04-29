from datasets import load_dataset, concatenate_datasets, Dataset
from pathlib import Path
from collections import OrderedDict
import numpy as np
import csv
import shutil

LABEL_MAP = {
    'O': 0,
    'B-PER': 1,
    'I-PER': 2,
    'B-ORG': 3,
    'I-ORG': 4,
    'B-LOC': 5,
    'I-LOC': 6
}
LABEL_MAP= {v:k for k,v in LABEL_MAP.items()}

def _combine_datasets(datasets, out_file):
    with open(out_file, 'w') as f:
        for dataset in datasets:
            with open(dataset, 'r') as fr:
                for line in dataset:
                    f.write(line)

def copy_master_task_def(target_dir):
    master = Path('experiments/NER/task_def.yaml')
    shutil.copy(master, target_dir)

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
                labels = [LABEL_MAP[int(row['ner_tags'][j])] for j in range(len(row['tokens']))]
                labels_as_str = ' '.join(labels)
                f.write(f'{i}\t{labels_as_str}\t{premise}\n')

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

    # list of 3 different sets of seeds to make the fractional training data.
    # Different seeds for each fractional generation, too.
    seeds = [
        list(range(500, 900, 100)),
        list(range(900, 1300, 100)),
        list(range(1300, 1700, 100))
    ]

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

# MAIN DATA GENERATION
def make_multilingual():
    out_dir = Path(f'experiments/NER/multi')
    if not out_dir.is_dir():
        langs = {'train': ['en', 'fr', 'de', 'es'], 'test': ['en', 'fr', 'de', 'es']}
        _prepare_data(out_dir, langs)
    copy_master_task_def(out_dir)

def make_per_language():
    for lang in ['en', 'fr', 'de', 'es']:
        out_dir = Path(f'experiments/NER/{lang}')
        if not out_dir.is_dir():
            langs = {'train': [lang], 'test': [lang]}
            _prepare_data(out_dir, langs)
        copy_master_task_def(out_dir)

def make_crosslingual():
    out_dir = Path(f'experiments/NER/cross')
    if not out_dir.is_dir():
        langs = {'train': ['en'], 'test': ['en']}
        _prepare_data(out_dir, langs)
    copy_master_task_def(out_dir)
    
def make_foreign():
    out_dir = Path(f'experiments/NER/foreign')
    if not out_dir.is_dir():
        langs = {'train': ['fr', 'de', 'es'], 'test': ['fr', 'de', 'es']}
        _prepare_data(out_dir, langs)
    copy_master_task_def(out_dir)

# FRACTIONAL TRAINING GENERATION
def make_fractional_training():
    foreign_dataset = 'experiments/NER/foreign/ner_train.tsv'
    subsample_and_combine(foreign_dataset, [0.2, 0.4, 0.6, 0.8])

if __name__ == '__main__':
    make_per_language()
    make_multilingual()
    make_crosslingual()
    make_foreign()





