import csv
import sys
import jsonlines
import numpy as np
from pathlib import Path
import subprocess
csv.field_size_limit(sys.maxsize)

RAW_DATA_PATH = Path('./experiments/NLI/data_raw')
DATA_PATH = Path('./experiments/NLI')

MNLI_TRAIN = RAW_DATA_PATH.joinpath('multinli_1.0_train.jsonl')
XNLI_DEV = RAW_DATA_PATH.joinpath('xnli.dev.jsonl')
XNLI_TEST = RAW_DATA_PATH.joinpath('xnli.test.jsonl')

CROSS_TRAIN = DATA_PATH.joinpath('cross/nli_train.tsv')
CROSS_TEST = DATA_PATH.joinpath('cross/nli_test.tsv')
CROSS_TEST_TMP = DATA_PATH.joinpath('cross/cross_test_tmp.tsv')
MULTI_TRAIN = DATA_PATH.joinpath('multi/nli_train.tsv')
MULTI_TEST = DATA_PATH.joinpath('multi/nli_test.tsv')

def combine_datasets(in_files, out_file):
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, 'w') as fw:
        fieldnames = ['id', 'label', 'premise', 'hypothesis']
        writer = csv.DictWriter(fw, fieldnames, delimiter='\t')

        for in_file in in_files:
            with open(in_file, 'r') as fr:
                reader =csv.DictReader(fr, delimiter='\t', fieldnames=fieldnames)
                for row in reader:
                    writer.writerow(row)

def raw_tsv_to_mtdnn_format(in_files, out_file, languages=None, excl_langs=None):
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, 'w') as fw:
        fieldnames = ['id', 'label', 'premise', 'hypothesis']
        writer = csv.DictWriter(fw, fieldnames, delimiter='\t')

        line_no = 0

        for in_file in in_files:
            with jsonlines.open(in_file) as fr:
                for row in fr:
                    if 'language' not in row:
                        row['language'] = 'en'
                    
                    # check language specific request
                    c1 = (languages is None) or (languages is not None and row['language'] in languages)
                    # then check excl_lang request
                    c2 = (excl_langs is None) or (excl_langs is not None and row['language'] not in excl_langs)
                    
                    if c1 and c2:
                        label = row['gold_label']
                        premise = row['sentence1']
                        hypo = row['sentence2']
                        
                        writer.writerow({
                            'id': line_no,
                            'label': label,
                            'premise': premise,
                            'hypothesis': hypo
                        })
                    line_no += 1


def make_multilingual():
    # multi-lingual train set is MNLI train + XNLI dev
    if not MULTI_TRAIN.is_file():
        print('making train dataset for multi')
        raw_tsv_to_mtdnn_format([MNLI_TRAIN, XNLI_DEV], MULTI_TRAIN)
    
    # # multi-lingual test set is XNLI test
    if not MULTI_TEST.is_file():
        print('making test dataset for multi')
        raw_tsv_to_mtdnn_format([XNLI_TEST], MULTI_TEST)
    
    cmd = f'python prepro_std.py --dataset NLI/multi --task_def experiments/NLI/task_def.yaml'
    cmd = cmd.split(' ')
    subprocess.run(cmd)

def make_crosslingual():
    # MNLI train set is also cross-lingual training set
    if not CROSS_TRAIN.is_file():
        print('making train dataset for cross')
        raw_tsv_to_mtdnn_format([MNLI_TRAIN], CROSS_TRAIN)
    
    # cross-lingual test set is XNLI test set
    if not CROSS_TEST.is_file():
        print('making test dataset for cross')
        raw_tsv_to_mtdnn_format([XNLI_TEST], CROSS_TEST)
    
    cmd = f'python prepro_std.py --dataset NLI/cross --task_def experiments/NLI/task_def.yaml'
    cmd = cmd.split(' ')
    subprocess.run(cmd)

def make_foreign_3lang():
    langs = [
            'de',
            'es',
            'fr',
        ]

    for split in ['train', 'test']:
        print(f'making {split} dataset for foreign_3lang')
        out_file = DATA_PATH.joinpath(f'foreign_3lang/nli_{split}.tsv')
        out_file.parent.mkdir(exist_ok=True)

        raw_tsv_to_mtdnn_format(
            [MNLI_TRAIN, XNLI_DEV],
            out_file,
            languages=langs
        )

    cmd = f'python prepro_std.py --dataset NLI/foreign_3lang --task_def experiments/NLI/task_def.yaml'
    cmd = cmd.split(' ')
    subprocess.run(cmd)

def make_foreign_14lang():
    for split in ['train', 'test']:
        print(f'making {split} dataset for foreign_14lang')
        out_file = DATA_PATH.joinpath(f'foreign_14lang/nli_{split}.tsv')
        out_file.parent.mkdir(exist_ok=True)

        if not out_file.is_file():
            raw_tsv_to_mtdnn_format(
                [MNLI_TRAIN, XNLI_DEV],
                out_file,
                excl_langs=['en']
            )

    cmd = f'python prepro_std.py --dataset NLI/foreign_14lang --task_def experiments/NLI/task_def.yaml'
    cmd = cmd.split(' ')
    subprocess.run(cmd)

def make_ltr_only_data():
    out_file = DATA_PATH.joinpath(f'multi-ltr/multi-ltr_train.tsv')
    out_file.parent.mkdir(parents=True, exist_ok=True)
    raw_tsv_to_mtdnn_format([MNLI_TRAIN, XNLI_DEV], out_file, excl_langs=['ar', 'ur', 'tr'])

    out_file = DATA_PATH.joinpath(f'multi-ltr/multi-ltr_test.tsv')
    out_file.parent.mkdir(parents=True, exist_ok=True)
    raw_tsv_to_mtdnn_format([XNLI_TEST], out_file, excl_langs=['ar', 'ur', 'tr'])

def make_per_language():
    langs = [
            'ar',
            'bg',
            'de',
            'el',
            'es',
            'en',
            'fr',
            'hi',
            'ru',
            'sw',
            'th',
            'tr',
            'ur',
            'vi',
            'zh',
        ]

    for language in langs:
        for split in ['train', 'test']:
            if split == 'train':
                datasets = [XNLI_DEV]
            else:
                datasets = [XNLI_TEST]

            print(f'making {split} dataset for {language}')
            out_file = DATA_PATH.joinpath(f'{language}/nli_{split}.tsv')
            out_file.parent.mkdir(exist_ok=True)

            if not out_file.is_file():
                raw_tsv_to_mtdnn_format(
                    datasets,
                    out_file,
                    languages=[language])
        
        train_json_exists = DATA_PATH.joinpath(language, 'bert-base-multilingual-cased', 'nli_train.json').is_file()
        test_json_exists = DATA_PATH.joinpath(language, 'bert-base-multilingual-cased', 'nli_test.json').is_file()
        if not (train_json_exists and test_json_exists):
            cmd = f'python prepro_std.py --dataset NLI/{language} --task_def experiments/NLI/task_def.yaml'
            cmd = cmd.split(' ')
            subprocess.run(cmd)
            


def subsample_and_combine(foreign_dataset, ps):
    fieldnames = ['id', 'label', 'premise', 'hypothesis']
    with open(f'experiments/NLI/cross/nli_train.tsv', 'r') as f:
        reader =csv.DictReader(f, delimiter='\t', fieldnames=fieldnames)
        mnli_rows = [row for row in reader]

    seeds = list(range(500, 800, 100))
    with open(foreign_dataset, 'r') as fr:
        reader = csv.DictReader(fr, delimiter='\t', fieldnames=fieldnames)
        rows = [r for r in reader]
        for p_idx, p in enumerate(ps):
            np.random.seed(seeds[p_idx])
            subsampled_idxs = np.random.choice(
                np.arange(len(rows)),
                size=int(len(rows)*p),
                replace=False)
            subsampled_rows = [rows[i] for i in subsampled_idxs]

            out_file = Path(f'experiments/NLI/foreign_{p}_2/nli_train.tsv')
            out_file.parent.mkdir(parents=True, exist_ok=True)

            with open(out_file, 'w') as fw:
                writer = csv.DictWriter(fw, fieldnames, delimiter='\t')
                for row in subsampled_rows:
                    writer.writerow(row)
            
                for r in mnli_rows:
                    writer.writerow(r)

if __name__ == '__main__':
    make_per_language()
    make_multilingual()
    make_crosslingual()
    make_foreign_3lang()
    make_foreign_14lang()


