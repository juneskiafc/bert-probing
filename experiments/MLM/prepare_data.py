from pathlib import Path
import csv
import json
import jsonlines

ROOT = Path('experiments/NLI/')
OUT_ROOT = Path('experiments/MLM/')
CROSS_TRAIN = ROOT.joinpath('cross', 'cross_train.tsv') # MNLI train
MULTI_TRAIN = ROOT.joinpath('multi', 'multi_train.tsv') # MNLI train + XNLI dev
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

def make_mlm_json_from_raw_xnli_dev(out_file, languages=None):
    if Path(out_file).is_file():
        return
    
    raw_nli_data_path = Path('./experiments/NLI/data_raw')
    xnli_dev_path = raw_nli_data_path.joinpath('xnli.dev.jsonl')

    Path(out_file).parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, 'w', encoding='utf-8') as fw:
        with jsonlines.open(xnli_dev_path) as fr:
            for row in fr:
                c1 = (languages is not None) and (row['language'] in languages)
                c2 = languages is None

                if c1 or c2:
                    premise = row['sentence1']
                    hypo = row['sentence2']
                    
                    for sent in [premise, hypo]:
                        fw.write(sent)
                        fw.write("\n")
                
    
if __name__ == '__main__':
    # make_mlm_json(CROSS_TRAIN, CROSS_TRAIN_OUT)
    # make_mlm_json(MULTI_TRAIN, MULTI_TRAIN_OUT)
    # make_mlm_json(CROSS_TEST, CROSS_TEST_OUT)
    # make_mlm_json(MULTI_TEST, MULTI_TEST_OUT)

    make_mlm_json_from_raw_xnli_dev('experiments/MLM/NLI/15lang_train.txt')
    make_mlm_json_from_raw_xnli_dev('experiments/MLM/NLI/4lang_train.txt', ['en', 'fr', 'de', 'es'])
            

                
