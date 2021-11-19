from pathlib import Path
import csv
import json

ROOT = Path('experiments/NLI/')
OUT_ROOT = Path('experiments/MLM/')
CROSS_TRAIN = ROOT.joinpath('cross', 'cross_train.tsv')
MULTI_TRAIN = ROOT.joinpath('multi', 'multi_train.tsv')
CROSS_TEST = ROOT.joinpath('cross', 'cross_test.tsv')
MULTI_TEST = ROOT.joinpath('multi', 'multi_test.tsv')

CROSS_TRAIN_OUT = OUT_ROOT.joinpath('cross_train.json')
MULTI_TRAIN_OUT = OUT_ROOT.joinpath('multi_train.json')
CROSS_TEST_OUT = OUT_ROOT.joinpath('cross_test.json')
MULTI_TEST_OUT = OUT_ROOT.joinpath('multi_test.json')

def load_loose_json(load_path):
    rows = []
    with open(load_path, 'r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            rows.append(row)
    return rows

def make_mlm_json(in_file, out_file):
    if out_file.is_file():
        return out_file

    fieldnames = ['id', 'label', 'premise', 'hypothesis']
    
    with open(out_file, 'w') as fw:
        with open(in_file, 'r') as fr:
            reader = csv.DictReader(fr, delimiter='\t', fieldnames=fieldnames)
            for row in reader:
                for sent in [row['premise'], row['hypothesis']]:
                    fw.write(json.dumps({'text': sent}))
                    fw.write("\n")

    return out_file
    
if __name__ == '__main__':
    # make_mlm_json(CROSS_TRAIN, CROSS_TRAIN_OUT)
    # make_mlm_json(MULTI_TRAIN, MULTI_TRAIN_OUT)
    # make_mlm_json(CROSS_TEST, CROSS_TEST_OUT)
    make_mlm_json(MULTI_TEST, MULTI_TEST_OUT)
    
            

                
