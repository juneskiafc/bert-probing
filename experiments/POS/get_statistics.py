import sys
sys.path.append('/home/june/mt-dnn')
from experiments.exp_def import LingualSetting
from pathlib import Path
import pandas as pd
from conllu import parse_incr

DATA_ROOT = Path('/home/june/mt-dnn/experiments/POS/data')

def get_num_instances():
    datasets = [
        DATA_ROOT.joinpath('en/UD_English-EWT'),
        DATA_ROOT.joinpath('fr/UD_French-GSD'),
        DATA_ROOT.joinpath('de/UD_German-GSD'),
        DATA_ROOT.joinpath('es/UD_Spanish-AnCora')
    ]

    for split in ['train', 'test']:                    
        for data_dir in datasets:
            n = 0
            data_file = list(data_dir.rglob(f'*{split}.conllu'))[0]

            with open(data_file, 'r', encoding='utf-8') as f:
                for i, tokenlist in enumerate(parse_incr(f)):
                    n += 1
            
            print(f'{data_dir.parent.name}, {split}, {n}')
    
if __name__ == '__main__':
    get_num_instances()
            