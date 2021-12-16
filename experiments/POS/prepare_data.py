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

def _prepare_data(dataset_dirs, out_dir, out_file=None):
    for split in ['train', 'test']:
        if split not in dataset_dirs:
            continue
            
        if out_file is None:
            out_file = out_dir.joinpath(f'pos_{split}.tsv')
        
        data = [[], []]
        print(f'making {split} data.')

        for data_dir in dataset_dirs[split]:
            print(f'\tprocessing {data_dir.name}')
            with open(data_dir.joinpath(f'{split}.conllu'), 'r', encoding='utf-8') as f:
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

def prepare_data_finetune():
    for setting in ['cross', 'multi']:
        out_dir = DATA_ROOT.parent.joinpath(setting)
        if out_dir.is_dir():
            continue
        else:
            out_dir.mkdir(parents=True)

        if setting == 'multi':
            train_datasets = [
                DATA_ROOT.joinpath('en/UD_English-EWT'),
                DATA_ROOT.joinpath('fr/UD_French-FTB'),
                DATA_ROOT.joinpath('de/UD_German-GSD'),
                DATA_ROOT.joinpath('es/UD_Spanish-AnCora')
            ]
        elif setting == 'cross':
            train_datasets = [DATA_ROOT.joinpath('en/UD_English-EWT')]
        
        dataset_dirs = {
            'train': train_datasets,
            'test': [
                DATA_ROOT.joinpath('en/UD_English-EWT'),
                DATA_ROOT.joinpath('fr/UD_French-FTB'),
                DATA_ROOT.joinpath('de/UD_German-GSD'),
                DATA_ROOT.joinpath('es/UD_Spanish-AnCora')
            ]
        }
        
        _prepare_data(dataset_dirs, out_dir)

def prepare_data_head_probe():
    dataset_dirs = {
    'train': [DATA_ROOT.joinpath('en/UD_English-EWT')],
    'test': [
        DATA_ROOT.joinpath('en/UD_English-EWT'),
        DATA_ROOT.joinpath('fr/UD_French-FTB'),
        DATA_ROOT.joinpath('de/UD_German-GSD'),
        DATA_ROOT.joinpath('es/UD_Spanish-AnCora')]
    }

    out_dir = DATA_ROOT.parent.joinpath('head_probe')
    if not out_dir.is_dir():
        _prepare_data(dataset_dirs, out_dir)

if __name__ == '__main__':
    prepare_data_finetune()
            