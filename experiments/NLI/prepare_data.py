import csv
import shutil
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

CROSS_TRAIN = DATA_PATH.joinpath('cross/cross_train.tsv')
CROSS_TEST = DATA_PATH.joinpath('cross/cross_test.tsv')
CROSS_TEST_TMP = DATA_PATH.joinpath('cross/cross_test_tmp.tsv')
MULTI_TRAIN = DATA_PATH.joinpath('multi/multi_train.tsv')
MULTI_TEST = DATA_PATH.joinpath('multi/multi_test.tsv')

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
                    # check language specific request
                    c1 = (languages is None) or (languages is not None and ('language' not in row or row['language'] in languages))
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

def make_main_data():
    # MNLI train set is also cross-lingual training set
    if not CROSS_TRAIN.is_file():
        raw_tsv_to_mtdnn_format([MNLI_TRAIN], CROSS_TRAIN)

    # cross-lingual test set is XNLI test set
    if not CROSS_TEST.is_file():
        raw_tsv_to_mtdnn_format([XNLI_TEST], CROSS_TEST)

    # multi-lingual train set is MNLI train + XNLI dev
    if not MULTI_TRAIN.is_file():
        raw_tsv_to_mtdnn_format([MNLI_TRAIN, XNLI_DEV], MULTI_TRAIN)

    # # multi-lingual test set is XNLI test
    if not MULTI_TEST.is_file():
        shutil.copy(CROSS_TEST, MULTI_TEST)

def make_ltr_only_data():
    out_file = DATA_PATH.joinpath(f'multi-ltr/multi-ltr_train.tsv')
    out_file.parent.mkdir(parents=True, exist_ok=True)
    raw_tsv_to_mtdnn_format([MNLI_TRAIN, XNLI_DEV], out_file, excl_langs=['ar', 'ur', 'tr'])

    out_file = DATA_PATH.joinpath(f'multi-ltr/multi-ltr_test.tsv')
    out_file.parent.mkdir(parents=True, exist_ok=True)
    raw_tsv_to_mtdnn_format([XNLI_TEST], out_file, excl_langs=['ar', 'ur', 'tr'])

def make_per_language_multilingual_data(languages, split='train'):
    if split == 'train':
        datasets = [XNLI_DEV]
    else:
        datasets = [XNLI_TEST]

    for language in languages:
        print(f'making {split} dataset for {language}')
        out_file = DATA_PATH.joinpath(f'{language}/nli_{split}.tsv')
        out_file.parent.mkdir(exist_ok=True)

        raw_tsv_to_mtdnn_format(
            datasets,
            out_file,
            languages=[language])

def make_evaluation_data():
    langs = [
            'ar',
            'bg',
            'de',
            'el',
            'en',
            'es',
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

    for lang in langs:
        print(f'making eval data for {lang}.')
        raw_tsv_to_mtdnn_format([XNLI_TEST], f'experiments/NLI/{lang}/nli_test.tsv', language=lang)

    print('combining 15 langs.')
    datasets = [f'experiments/NLI/{lang}/nli_test.tsv' for lang in langs]
    combine_datasets(datasets, 'experiments/NLI/combined/nli_test.tsv')

    print('combining en/es/fr/de.')
    fourlang_datasets = [f'experiments/NLI/{lang}/nli_test.tsv' for lang in langs if lang in ['en', 'fr', 'de', 'es']]
    combine_datasets(fourlang_datasets, 'experiments/NLI/4lang_combined/nli_test.tsv')

def subsample_and_combine(foreign_dataset, ps, seeds):
    fieldnames = ['id', 'label', 'premise', 'hypothesis']
    with open(f'experiments/NLI/cross/nli_train.tsv', 'r') as f:
        reader =csv.DictReader(f, delimiter='\t', fieldnames=fieldnames)
        mnli_rows = [row for row in reader]

    for i, seed_collection in enumerate(seeds):
        with open(foreign_dataset, 'r') as fr:
            reader = csv.DictReader(fr, delimiter='\t', fieldnames=fieldnames)
            rows = [r for r in reader]
            for p_idx, p in enumerate(ps):
                out_file = Path(f'experiments/NLI/foreign_{p}_{i}/nli_train.tsv')
                if out_file.is_file():
                    continue
                    
                np.random.seed(seed_collection[p_idx])
                subsampled_idxs = np.random.choice(
                    np.arange(len(rows)),
                    size=int(len(rows)*p),
                    replace=False)
                subsampled_rows = [rows[i] for i in subsampled_idxs]

                out_file.parent.mkdir(parents=True, exist_ok=True)
                with open(out_file, 'w') as fw:
                    writer = csv.DictWriter(fw, fieldnames, delimiter='\t')
                    for row in subsampled_rows:
                        writer.writerow(row)
                
                    for r in mnli_rows:
                        writer.writerow(r)

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
    make_per_language_multilingual_data(langs)

def make_multilingual():
    if not MULTI_TRAIN.is_file():
        raw_tsv_to_mtdnn_format([MNLI_TRAIN, XNLI_DEV], MULTI_TRAIN)
    if not CROSS_TEST.is_file():
        raw_tsv_to_mtdnn_format([XNLI_TEST], MULTI_TEST)

def make_crosslingual():
    # MNLI train set is also cross-lingual training set
    if not CROSS_TRAIN.is_file():
        raw_tsv_to_mtdnn_format([MNLI_TRAIN], CROSS_TRAIN)

    # cross-lingual test set is XNLI test set
    if not CROSS_TEST.is_file():
        raw_tsv_to_mtdnn_format([XNLI_TEST], CROSS_TEST)

def make_foreign():
    langs = [
            # 'ar',
            # 'bg',
            'de',
            # 'el',
            'es',
            'fr',
            # 'hi',
            # 'ru',
            # 'sw',
            # 'th',
            # 'tr',
            # 'ur',
            # 'vi',
            # 'zh',
        ]
    raw_tsv_to_mtdnn_format([XNLI_DEV], 'experiments/NLI/foreign/nli_train.tsv', langs)
    raw_tsv_to_mtdnn_format([XNLI_TEST], 'experiments/NLI/foreign/nli_test.tsv', langs)

def make_fractional_training():
    foreign_dataset = 'experiments/NLI/foreign/nli_train.tsv'
    seeds = [list(range(500, 900, 100)), list(range(900, 1300, 100)), list(range(1300, 1700, 100))]
    subsample_and_combine(foreign_dataset, [0.2, 0.4, 0.6, 0.8], seeds)
    prepro_wrapper_for_foreign()

def prepro_wrapper_for_foreign():
    for i in range(3):
        for frac in [0.2, 0.4, 0.6, 0.8]:
            dataset_name = f'NLI/foreign_{frac}_{i}'
            task_def = 'experiments/NLI/task_def.yaml'

            cmd = f'python prepro_std.py --dataset {dataset_name} --task_def {task_def}'
            split_cmd = cmd.split(" ")
            subprocess.run(split_cmd)

if __name__ == '__main__':
    # make_per_language()
    # make_multilingual()
    # make_crosslingual()
    # make_foreign()
    # make_fractional_training()
    datasets = ['experiments/NLI/cross/nli_train.tsv', 'experiments/NLI/foreign/nli_train.tsv']
    combine_datasets(datasets, 'experiments/NLI/multi_4lang/nli_train.tsv')


