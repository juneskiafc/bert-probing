import csv
import shutil
import sys
import jsonlines
from pathlib import Path
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

def make_per_language_multilingual_data(exclude_english=False, split='train'):
    languages = set()
    with jsonlines.open(XNLI_DEV) as f:
        for row in f:
            languages.add(row['language'])

    for language in languages:
        print(f'making dataset for {language}')
        if exclude_english:
            out_file = DATA_PATH.joinpath(f'{language}/{language}_{split}.tsv')
        else:
            out_file = DATA_PATH.joinpath(f'multi-{language}/multi-{language}_{split}.tsv')
        
        out_file.parent.mkdir(exist_ok=True)

        if split == 'train':
            if not exclude_english:
                datasets = [MNLI_TRAIN, XNLI_DEV]
            else:
                datasets = [XNLI_DEV]
        elif split == 'test':
            datasets = [XNLI_TEST]

        if exclude_english and language != 'en':
            raw_tsv_to_mtdnn_format(datasets, out_file, language=language, excl_langs=['en'])
        else:
            raw_tsv_to_mtdnn_format(datasets, out_file, language=language)

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

if __name__ == '__main__':
    raw_tsv_to_mtdnn_format([XNLI_TEST], Path('experiments/NLI/en/nli_test.tsv'), languages=['en'])