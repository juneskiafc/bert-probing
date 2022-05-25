from pathlib import Path
import csv
import json
import jsonlines
import sys
sys.path.append('/home/june/mt-dnn/')
from experiments.exp_def import Experiment, LingualSetting
from datasets import Dataset
from argparse import ArgumentParser
from conllu import parse_incr
from argparse import ArgumentParser


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

def combine_txts(files, out_file):
    if out_file.is_file():
        return
    
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w') as fw:
        for file in files:
            with open(file, 'r') as fr:
                for line in fr:
                    if line != '\n':
                        fw.write(line)

def make_mlm_data_from_raw_mnli(out_file):
    if Path(out_file).is_file():
        return
    
    raw_nli_data_path = Path('./experiments/NLI/data_raw')
    xnli_dev_path = raw_nli_data_path.joinpath('multinli_1.0_train.jsonl')

    Path(out_file).parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, 'w', encoding='utf-8') as fw:
        with jsonlines.open(xnli_dev_path) as fr:
            for row in fr:
                premise = row['sentence1']
                hypo = row['sentence2']
                
                for sent in [premise, hypo]:
                    fw.write(sent)
                    fw.write("\n")

def make_mlm_data_from_raw_xnli(split, out_dir, languages=None, separate_premise_hypothesis=True):
    raw_nli_data_path = Path('./experiments/NLI/data_raw')

    if split == 'train':
        xnli_path = raw_nli_data_path.joinpath('xnli.dev.jsonl')
    else:
        xnli_path = raw_nli_data_path.joinpath('xnli.test.jsonl')

    out_files_per_lang = []
    for lang in ['en', 'de', 'es', 'fr']:
        languages = [lang]
        out_file = out_dir.joinpath(f'{lang}', f'nli_{split}.txt')
        out_files_per_lang.append(out_file)

        if not out_file.is_file():
            out_file.parent.mkdir(parents=True, exist_ok=True)
        
            with open(out_file, 'w', encoding='utf-8') as fw:
                with jsonlines.open(xnli_path) as fr:
                    for row in fr:
                        c1 = (languages is not None) and (row['language'] in languages)
                        c2 = languages is None

                        if c1 or c2:
                            premise = row['sentence1']
                            hypo = row['sentence2']
                            
                            if separate_premise_hypothesis:
                                for sent in [premise, hypo]:
                                    fw.write(sent)
                                    fw.write("\n")
                            else:
                                sent = f'{premise} [SEP] {hypo}'
                                fw.write(sent)
                                fw.write('\n')
    
    combine_txts(out_files_per_lang, out_dir.joinpath('multi', f'nli_{split}.txt'))

def make_mlm_data_from_pos(split, out_dir):
    DATA_ROOT = Path('experiments/POS/data')

    train_data_files = {
        'en': DATA_ROOT.joinpath('en/UD_English-EWT'),
        'fr': DATA_ROOT.joinpath('fr/UD_French-GSD'),
        'de': DATA_ROOT.joinpath('de/UD_German-GSD'),
        'es': DATA_ROOT.joinpath('es/UD_Spanish-AnCora')
    }
    
    out_files_per_lang = []
    for lang in ['en', 'es', 'de', 'fr']:
        out_file = out_dir.joinpath(lang, f'pos_{split}.txt')
        out_files_per_lang.append(out_file)

        if not out_file.is_file():
            out_file.parent.mkdir(parents=True, exist_ok=True)

            with open(out_file, 'w', encoding='utf-8') as fw:
                train_file = list(train_data_files[lang].rglob(f'*{split}.conllu'))[0]
                with open(train_file, 'r', encoding='utf-8') as f:
                    for i, tokenlist in enumerate(parse_incr(f)):
                        fw.write(tokenlist.metadata['text'])
                        fw.write('\n')
    
    combine_txts(out_files_per_lang, out_dir.joinpath('multi', f'pos_{split}.txt'))

def make_mlm_data_from_pawsx(split, out_dir, separate_premise_hypothesis=True):
    out_files_per_lang = []
    for lang in ['en', 'es', 'de', 'fr']:
        out_file = out_dir.joinpath(lang, f'pawsx_{split}.txt')
        out_files_per_lang.append(out_file)

        if not out_file.is_file():
            out_file.parent.mkdir(parents=True, exist_ok=True)

            with open(out_file, 'w', encoding='utf-8') as f:
                tmp_out_file = f'experiments/PAWSX/{lang}/pawsx_{split}_tmp.json'
                df = Dataset.from_json(str(tmp_out_file))

                premises = []
                hypos = []
                # n_lines = len(df)
                # n_aug = int(n_lines * 0.5)

                for i, row in enumerate(df):
                    premise = row['sentence1']
                    hypo = row['sentence2']
                    if separate_premise_hypothesis:
                        for sent in [premise, hypo]:
                            f.write(sent)
                            f.write("\n")
                    else:
                        sent = f'{premise} [SEP] {hypo}'
                        f.write(sent)
                        f.write('\n')
                    
                    premises.append(premise)
                    hypos.append(hypo)
                
                    # for i in range(n_aug):
                    #     premise_id = np.random.randint(0, n_lines)
                    #     hypo_id = np.random.randint(0, n_lines)
                    #     sent = f'{premises[premise_id]} [SEP] {hypos[hypo_id]}'
                    #     f.write(sent)
                    #     f.write('\n')
    combine_txts(out_files_per_lang, out_dir.joinpath('multi', f'pawsx_{split}.txt'))


def make_mlm_data_from_marc(split, out_dir):
    out_files_per_lang = []
    for lang in ['en', 'de', 'es', 'fr']:
        out_file = out_dir.joinpath(lang, f'marc_{split}.txt')
        out_files_per_lang.append(out_file)

        if not out_file.is_file():
            out_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_out_file = f'experiments/MARC/{lang}/marc_{split}_tmp.json'

            df = Dataset.from_json(str(tmp_out_file))
            with open(out_file, 'w', encoding='utf-8') as f:
                for i, row in enumerate(df):
                    f.write(row['review_body'])
                    f.write('\n')

    combine_txts(out_files_per_lang, out_dir.joinpath('multi', f'marc_{split}.txt'))

def make_mlm_data_from_ner(split, out_dir):
    out_files_per_lang = []
    for lang in ['en', 'de', 'es', 'fr']:
        out_file = out_dir.joinpath(lang, f'ner_{split}.txt')
        out_files_per_lang.append(out_file)

        if not out_file.is_file():
            out_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_out_file = f'experiments/NER/{lang}/ner_{split}_tmp.json'

            df = Dataset.from_json(str(tmp_out_file))
            with open(out_file, 'w', encoding='utf-8') as f:
                for i, row in enumerate(df):
                    premise = ' '.join(row['tokens'])
                    f.write(premise)
                    f.write('\n')
    
    combine_txts(out_files_per_lang, out_dir.joinpath('multi', f'ner_{split}.txt'))

def make_mlm_data(task: Experiment, separate_multiple_sentences_per_example=True):
    out_dir = Path('experiments/MLM').joinpath(task.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for split in ['train', 'test']:
        if task is Experiment.NLI:
            make_mlm_data_from_raw_xnli(
                split,
                out_dir,
                separate_premise_hypothesis=separate_multiple_sentences_per_example)
            
        elif task is Experiment.POS:
            make_mlm_data_from_pos(split,out_dir)

        elif task is Experiment.PAWSX:
            make_mlm_data_from_pawsx(
                split,
                out_dir,
                separate_premise_hypothesis=separate_multiple_sentences_per_example)

        elif task is Experiment.MARC:
            make_mlm_data_from_marc(split, out_dir)

        elif task is Experiment.NER:
            make_mlm_data_from_ner(split, out_dir)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--task', type=str, default='')
    parser.add_argument("--separate", action='store_true')
    args = parser.parse_args()
    
    if args.task == '':
        tasks = list(Experiment)
        tasks.remove(Experiment.BERT)
    else:
        tasks = [Experiment[args.task.upper()]]
    
    for task in tasks:
        make_mlm_data(task, args.separate)


            

                
