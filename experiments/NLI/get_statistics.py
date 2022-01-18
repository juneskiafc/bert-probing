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

def get_num_instances(in_files, language=None, excl_langs=None):
    n = 0
    for in_file in in_files:
        with jsonlines.open(in_file) as fr:
            for row in fr:
                # check language specific request
                c1 = (language is None) or (language is not None and row['language'] == language)
                # then check excl_lang request
                c2 = (excl_langs is None) or (excl_langs is not None and row['language'] not in excl_langs)
                
                if c1 and c2:
                    n += 1
    print(n)

if __name__ == '__main__':
    for lang in ['ar',
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
            'zh']:
        print(lang)
        get_num_instances([XNLI_DEV], language=lang)
        get_num_instances([XNLI_TEST], language=lang)
        print('\n')