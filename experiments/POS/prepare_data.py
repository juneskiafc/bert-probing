from pathlib import Path
import pandas as pd
from conllu import parse_incr

DATA_ROOT = Path('/home/june/mt-dnn/experiments/pos/data')

dataset_dirs = [
    DATA_ROOT.joinpath('en/UD_English-EWT'),
    DATA_ROOT.joinpath('fr/UD_French-FTB'),
    DATA_ROOT.joinpath('de/UD_German-GSD'),
    DATA_ROOT.joinpath('es/UD_Spanish-AnCora')
]

def simplify_name():
    for data_dir in dataset_dirs[1:]:
        for file in data_dir.iterdir():
            if file.suffix in ['.txt', '.conllu']:
                if file.name != 'LICENSE.txt' and file.name != 'README.txt':
                    split = file.with_suffix("").name.split("-")[-1]
                    new_name = file.parent.joinpath(f'{split}{file.suffix}')
                    print(f'{file.name} -> {new_name}')
                    file.rename(file.parent.joinpath(new_name))

task_name = 'pos'
id_ = 0

for split in ['train', 'test']:
    out_file = DATA_ROOT.parent.joinpath(f'{task_name}_{split}.tsv')
    data = [[], []]

    print(f'making {split} data.')
    for data_dir in dataset_dirs:
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
                id_ += 1

    data = [pd.Series(d) for d in data]
    df = pd.concat(data, axis=1)
    df.to_csv(out_file, sep='\t', header=None)
            