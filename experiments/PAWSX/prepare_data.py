from datasets import load_dataset, concatenate_datasets, Dataset
from pathlib import Path
import sys
sys.path.append('/home/june/mt-dnn/')
from experiments.exp_def import LingualSetting
import csv
from collections import OrderedDict
import subprocess
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

def subsample_and_combine(foreign_dataset, ps, seeds):
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

def make_per_language():
    for lang in ['en', 'es', 'de', 'fr']:
        out_dir = Path(f'experiments/PAWSX/{lang}')
        langs = [lang]
        _prepare_data(langs, langs, out_dir)

def make_multilingual():
    out_dir = Path(f'experiments/PAWSX/multi')
    langs = ['en', 'es', 'de', 'fr']
    _prepare_data(langs, langs, out_dir)

def make_crosslingual():
    out_dir = Path(f'experiments/PAWSX/cross')
    langs = ['en']
    _prepare_data(langs, langs, out_dir)

def make_foreign():
    out_dir = Path(f'experiments/PAWSX/cross')
    langs = ['fr', 'de', 'es']
    _prepare_data(langs, langs, out_dir)

def make_fractional_training():
    foreign_dataset = 'experiments/PAWSX/foreign/pawsx_train.tsv'
    seeds = [list(range(500, 900, 100)), list(range(900, 1300, 100)), list(range(1300, 1700, 100))]
    subsample_and_combine(foreign_dataset, [0.2, 0.4, 0.6, 0.8], seeds)

def prepro_wrapper_for_foreign():
    for i in range(3):
        for frac in [0.2, 0.4, 0.6, 0.8]:
            dataset_name = f'PAWSX/foreign_{frac}_{i}'
            task_def = 'experiments/PAWSX/task_def.yaml'

            cmd = f'python prepro_std.py --dataset {dataset_name} --task_def {task_def}'
            split_cmd = cmd.split(" ")
            subprocess.run(split_cmd)

if __name__ == '__main__':
    # make_per_language()
    # make_multilingual()
    # make_crosslingual()
    # make_foreign()
    # make_fractional_training()
    prepro_wrapper_for_foreign()





