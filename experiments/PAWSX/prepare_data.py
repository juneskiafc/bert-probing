from datasets import load_dataset, concatenate_datasets, Dataset
from pathlib import Path
import sys
sys.path.append('/home/june/mt-dnn/')
from experiments.exp_def import LingualSetting
import csv
from collections import OrderedDict
import numpy as np

def _prepare_data(train_langs, test_langs, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = []
    if train_langs is not None:
        splits.append('train')
    if test_langs is not None:
        splits.append('test')
    
    for split in splits:
        out_file = out_dir.joinpath(f'pawsx_{split}_tmp.json')
        final_out_file = out_dir.joinpath(f'pawsx_{split}.tsv')

        if final_out_file.is_file():
            continue
        else:
            if not Path(out_file).is_file():
                datasets = []
                if split == 'train':
                    langs = train_langs
                else:
                    langs = test_langs
                
                for lang in langs:
                    dataset = load_dataset('paws-x', lang, split=split)
                    datasets.append(dataset)

                dataset = concatenate_datasets(datasets)
                dataset.to_json(out_file)

            # load and save as tsv in mtdnn format
            df = Dataset.from_json(str(out_file))
            with open(final_out_file, 'w') as f:
                for i, row in enumerate(df):
                    premise = row['sentence1']
                    hypo = row['sentence2']
                    label = row['label']
                    f.write(f'{i}\t{label}\t{premise}\t{hypo}\n')

def prepare_finetune_data():
    train_langs_per_setting = {
        LingualSetting.CROSS: ['en'],
        LingualSetting.MULTI: ['en', 'fr', 'de', 'es']
    }
    test_langs = ['en', 'fr', 'de', 'es']

    for setting in [LingualSetting.CROSS, LingualSetting.MULTI]:
        out_dir = Path(f'experiments/PAWSX/{setting.name.lower()}')
        train_langs = train_langs_per_setting[setting]

        _prepare_data(train_langs, test_langs, out_dir)

def subsample_and_combine(foreign_dataset, ps):
    def read_rows(filename):
        with open(filename, 'r') as f:
            rows = []
            for row in f:
                id_, label, premise, hypothesis = row.split("\t")
                hypothesis = hypothesis.strip('\n')
                rows.append(OrderedDict({'id': id_, 'label': label, 'premise': premise, 'hypothesis': hypothesis}))
        return rows

    fieldnames = ['id', 'label', 'premise', 'hypothesis']
    mnli_rows = read_rows('experiments/PAWSX/cross/pawsx_train.tsv')

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

            out_file = Path(f'experiments/PAWSX/foreign_{p}_{i}/pawsx_train.tsv')
            out_file.parent.mkdir(parents=True, exist_ok=True)

            with open(out_file, 'w') as fw:
                writer = csv.DictWriter(fw, fieldnames, delimiter='\t')
                for row in subsampled_rows:
                    writer.writerow(row)
            
                for r in mnli_rows:
                    writer.writerow(r)

if __name__ == '__main__':
    # out_dir = Path(f'experiments/PAWSX/cross')
    # langs = ['en']
    # _prepare_data(langs, langs, out_dir)
    
    # out_dir = Path(f'experiments/PAWSX/fr')
    # langs = ['fr']
    # _prepare_data(langs, langs, out_dir)
    
    foreign_dataset = 'experiments/PAWSX/foreign/pawsx_train.tsv'
    subsample_and_combine(foreign_dataset, [0.2, 0.4, 0.6, 0.8])





