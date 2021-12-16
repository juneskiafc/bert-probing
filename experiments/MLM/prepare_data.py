from pathlib import Path
import csv
import json
import jsonlines
import sys
sys.path.append('/home/june/mt-dnn/')
from experiments.exp_def import Experiment
from datasets import Dataset
from conllu import parse_incr

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

def make_mlm_data_from_raw_xnli_dev(out_file, languages=None):
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

def make_mlm_data_from_pos(out_file):
    DATA_ROOT = Path('/home/june/mt-dnn/experiments/POS/data')

    train_data_files = [
        DATA_ROOT.joinpath('en/UD_English-EWT'),
        DATA_ROOT.joinpath('fr/UD_French-FTB'),
        DATA_ROOT.joinpath('de/UD_German-GSD'),
        DATA_ROOT.joinpath('es/UD_Spanish-AnCora')
    ]
    
    with open(out_file, 'w', encoding='utf-8') as fw:
        for data_dir in train_data_files:
            with open(data_dir.joinpath(f'train.conllu'), 'r', encoding='utf-8') as f:
                for i, tokenlist in enumerate(parse_incr(f)):
                    fw.write(tokenlist.metadata['text'])
                    fw.write('\n')

def make_mlm_data_from_pawsx(out_file):
    if Path(out_file).is_file():
        return

    tmp_out_file = 'experiments/PAWSX/multi/pawsx_train_tmp.json'

    df = Dataset.from_json(str(tmp_out_file))
    with open(out_file, 'w', encoding='utf-8') as f:
        for i, row in enumerate(df):
            premise = row['sentence1']
            hypo = row['sentence2']

            for sent in [premise, hypo]:
                f.write(sent)
                f.write('\n')

def make_mlm_data_from_marc(out_file):
    if Path(out_file).is_file():
        return

    tmp_out_file = 'experiments/MARC/multi/marc_train_tmp.json'

    df = Dataset.from_json(str(tmp_out_file))
    with open(out_file, 'w', encoding='utf-8') as f:
        for i, row in enumerate(df):
            f.write(row['review_body'])
            f.write('\n')

def make_mlm_data_from_ner(out_file):
    if Path(out_file).is_file():
        return
    
    tmp_out_file = 'experiments/NER/multi/ner_train_tmp.json'

    df = Dataset.from_json(str(tmp_out_file))
    with open(out_file, 'w', encoding='utf-8') as f:
        for i, row in enumerate(df):
            premise = ' '.join(row['tokens'])
            f.write(premise)
            f.write('\n')

def make_mlm_data(task: Experiment):
    out_dir = Path('experiments/MLM').joinpath(task.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if task is Experiment.NLI:
        make_mlm_data_from_raw_xnli_dev(out_dir.joinpath('15lang_train.txt'))
        make_mlm_data_from_raw_xnli_dev(out_dir.joinpath('4lang_train.txt'), languages=['en', 'fr', 'de', 'es'])
    elif task is Experiment.POS:
        make_mlm_data_from_pos(out_dir.joinpath('4lang_train.txt'))
    elif task is Experiment.PAWSX:
        make_mlm_data_from_pawsx(out_dir.joinpath('4lang_train.txt'))
    elif task is Experiment.MARC:
        make_mlm_data_from_marc(out_dir.joinpath('4lang_train.txt'))
    elif task is Experiment.NER:
        make_mlm_data_from_ner(out_dir.joinpath('4lang_train.txt'))

    
if __name__ == '__main__':
    # make_mlm_json(CROSS_TRAIN, CROSS_TRAIN_OUT)
    # make_mlm_json(MULTI_TRAIN, MULTI_TRAIN_OUT)
    # make_mlm_json(CROSS_TEST, CROSS_TEST_OUT)
    # make_mlm_json(MULTI_TEST, MULTI_TEST_OUT)

    for task in list(Experiment):
        make_mlm_data(task)


            

                
