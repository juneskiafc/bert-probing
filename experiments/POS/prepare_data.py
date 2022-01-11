import sys
sys.path.append('/home/june/mt-dnn')
from experiments.exp_def import LingualSetting
from pathlib import Path
import pandas as pd
from conllu import parse_incr

DATA_ROOT = Path('/home/june/mt-dnn/experiments/POS/data')

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
    for split in ['train', 'test']:
        if split not in dataset_dirs:
            continue
            
        out_file = out_dir.joinpath(f'pos_{split}.tsv')
        
        data = [[], []]
        print(f'making {split} data: {out_file}')

        for data_dir in dataset_dirs[split]:
            print(f'\tprocessing {data_dir.name}')
            data_file = list(data_dir.rglob(f'*{split}.conllu'))[0]

            with open(data_file, 'r', encoding='utf-8') as f:
                for i, tokenlist in enumerate(parse_incr(f)):
                    first_word_idx = 0
                    while tokenlist[first_word_idx]['upos'] == '_':
                        first_word_idx += 1

                    first_word = tokenlist[first_word_idx]
                    label = first_word['upos']
                    premise = tokenlist.metadata['text']
                    data[0].append(label)
                    data[1].append(premise)

        data = [pd.Series(d) for d in data]
        df = pd.concat(data, axis=1)
        df.to_csv(out_file, sep='\t', header=None)

def prepare_data_finetune(setting: LingualSetting):
    out_dir = DATA_ROOT.parent.joinpath(setting.name.lower())
    out_dir.mkdir(parents=True, exist_ok=True)

    if setting is LingualSetting.MULTI:
        train_datasets = [
            DATA_ROOT.joinpath('en/UD_English-EWT'),
            DATA_ROOT.joinpath('fr/UD_French-GSD'),
            DATA_ROOT.joinpath('de/UD_German-GSD'),
            DATA_ROOT.joinpath('es/UD_Spanish-AnCora')
        ]
    elif setting is LingualSetting.CROSS:
        train_datasets = [DATA_ROOT.joinpath('en/UD_English-EWT')]
    
    dataset_dirs = {
        'train': train_datasets,
        'test': [
            DATA_ROOT.joinpath('en/UD_English-EWT'),
            DATA_ROOT.joinpath('fr/UD_French-GSD'),
            DATA_ROOT.joinpath('de/UD_German-GSD'),
            DATA_ROOT.joinpath('es/UD_Spanish-AnCora')
        ]
    }
    
    _prepare_data(dataset_dirs, out_dir)

def prepare_per_language_test_data():
    datasets = [
        DATA_ROOT.joinpath('en/UD_English-EWT'),
        DATA_ROOT.joinpath('fr/UD_French-GSD'),
        DATA_ROOT.joinpath('de/UD_German-GSD'),
        DATA_ROOT.joinpath('es/UD_Spanish-AnCora')
    ]

    for dataset in datasets:
        dataset_dirs = {'test': [dataset]}
        language = dataset.parent.name
        out_dir = DATA_ROOT.parent.joinpath(language)
        out_dir.mkdir(parents=True, exist_ok=True)
        _prepare_data(dataset_dirs, out_dir)

if __name__ == '__main__':
    prepare_data_finetune(LingualSetting.MULTI)
            