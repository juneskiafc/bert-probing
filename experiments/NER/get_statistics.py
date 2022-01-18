from datasets import load_dataset, concatenate_datasets, Dataset
from pathlib import Path

def get_num_instances():
    for split in ['train', 'test']:
        for lang in ['en', 'fr', 'es', 'de']:
            dataset = load_dataset('wikiann', lang, split=split)
            print(f'{lang}, {split}, {len(dataset)}')

if __name__ == '__main__':
    get_num_instances()



