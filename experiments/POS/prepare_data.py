import sys
from pathlib import Path
import pandas as pd
import csv
import numpy as np
from conllu import parse_incr
import shutil

DATA_ROOT = Path('/home/june/mt-dnn/experiments/POS/data')
DATASETS = {
    'en':  DATA_ROOT.joinpath('en/UD_English-EWT'),
    'fr': DATA_ROOT.joinpath('fr/UD_French-GSD'),
    'de': DATA_ROOT.joinpath('de/UD_German-GSD'),
    'es': DATA_ROOT.joinpath('es/UD_Spanish-AnCora')
}

def copy_master_task_def(target_dir):
    master = Path('experiments/POS/task_def.yaml')
    shutil.copy(master, target_dir)

def simplify_name(dataset_dirs):
    for data_dir in dataset_dirs[1:]:
        for file in data_dir.iterdir():
            if file.suffix in ['.txt', '.conllu']:
                if file.name != 'LICENSE.txt' and file.name != 'README.txt':
                    split = file.with_suffix("").name.split("-")[-1]
                    new_name = file.parent.joinpath(f'{split}{file.suffix}')
                    print(f'{file.name} -> {new_name}')
                    file.rename(file.parent.joinpath(new_name))

def _prepare_data(dataset_dirs, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'test']:
        if split not in dataset_dirs:
            continue
            
        out_file = out_dir.joinpath(f'pos_{split}.tsv')
        
        data = [[], []]
        print(f'making {split} data: {out_file}')
        if out_file.is_file():
            return

        for data_dir in dataset_dirs[split]:
            print(f'\tprocessing {data_dir.name}')
            data_file = list(data_dir.rglob(f'*{split}.conllu'))[0]

            with open(data_file, 'r', encoding='utf-8') as f:
                for i, tokenlist in enumerate(parse_incr(f)):
                    labels = []
                    tokens = []
                    words = [tokenlist[i]['form'] for i in range(len(tokenlist))]
                    for i in range(len(tokenlist)):
                        word = tokenlist[i]['form']
                        if " " in word:
                            word = word.replace(" ", "")
                        
                        upos = tokenlist[i]['upos']
                        if upos == '_':
                            upos = 'X'
                        
                        labels.append(upos)
                        tokens.append(word)
                    
                    assert len(labels) == len(tokens)
                    data[0].append(' '.join(labels))
                    data[1].append(' '.join(tokens))

        data = [pd.Series(d) for d in data]
        df = pd.concat(data, axis=1)
        df.to_csv(out_file, sep='\t', header=None)

def subsample_and_combine(foreign_dataset, ps):
    fieldnames = ['id', 'label', 'premise']
    with open(f'experiments/POS/cross/pos_train.tsv', 'r') as f:
        reader = csv.DictReader(f, delimiter='\t', fieldnames=fieldnames)
        mnli_rows = [row for row in reader]

    seeds = [list(range(500, 900, 100)), list(range(900, 1300, 100)), list(range(1300, 1700, 100))]
    for i, seed_collection in enumerate(seeds):
        with open(foreign_dataset, 'r') as fr:
            reader = csv.DictReader(fr, delimiter='\t', fieldnames=fieldnames)
            rows = [r for r in reader]
            for p_idx, p in enumerate(ps):
                np.random.seed(seed_collection[p_idx])
                subsampled_idxs = np.random.choice(
                    np.arange(len(rows)),
                    size=int(len(rows)*p),
                    replace=False)
                subsampled_rows = [rows[i] for i in subsampled_idxs]

                out_file = Path(f'experiments/POS/foreign_{p}_{i}/pos_train.tsv')
                out_file.parent.mkdir(parents=True, exist_ok=True)

                with open(out_file, 'w') as fw:
                    writer = csv.DictWriter(fw, fieldnames, delimiter='\t')
                    for row in subsampled_rows:
                        writer.writerow(row)
                
                    for r in mnli_rows:
                        writer.writerow(r)

def combine_datasets(datasets, out_file):
    if out_file.is_file():
        return
    
    with open(out_file, 'w') as f:
        for dataset in datasets:
            with open(dataset, 'r') as fr:
                for line in fr:
                    f.write(line)
    
def make_multilingual(langs=['en', 'fr', 'de', 'es'], out_dir=Path('experiments/POS/multi')):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    trains = []
    tests = []
    for lang in langs:
        make_for_lang(lang)
        trains.append(Path(f'experiments/POS/{lang}/pos_train.tsv'))
        tests.append(Path(f'experiments/POS/{lang}/pos_test.tsv'))
    
    for split in ['train', 'test']:
        if not out_dir.joinpath(f'pos_{split}.tsv').is_file():
            datasets = [Path(f'experiments/POS/{lang}/pos_{split}.tsv') for lang in langs]
            combine_datasets(datasets, out_dir.joinpath(f'pos_{split}.tsv'))
        
    copy_master_task_def(out_dir)

def make_for_lang(lang):
    out_dir = Path(f'experiments/POS/{lang}')
    if lang == 'en' and Path(f'experiments/POS/cross').is_dir():
        shutil.copytree('experiments/POS/cross', 'experiments/POS/en')
    else:
        dataset_dirs = {
            'train': [DATASETS[lang]],
            'test': [DATASETS[lang]]
        }
        _prepare_data(dataset_dirs, out_dir)
    copy_master_task_def(out_dir)

def make_per_language():
    for lang in ['en', 'fr', 'de', 'es']:
        make_for_lang(lang)

def make_crosslingual():
    out_dir = Path(f'experiments/POS/cross')
    if Path(f'experiments/POS/en').is_dir():
        shutil.copytree('experiments/POS/en', out_dir)
    else:
        dataset_dirs = {
            'train': [DATASETS['en']],
            'test': [DATASETS['en']]
        }
        _prepare_data(dataset_dirs, out_dir)
    copy_master_task_def(out_dir)

def make_foreign():
    make_multilingual(langs=['fr', 'de', 'es'], out_dir=Path('experiments/POS/foreign'))

def make_fractional_training():
    foreign_dataset = 'experiments/POS/foreign/pos_train.tsv'
    subsample_and_combine(foreign_dataset, [0.2, 0.4, 0.6, 0.8])

if __name__ == '__main__':
    make_per_language()
    make_multilingual()
    make_crosslingual()
    make_foreign()
            