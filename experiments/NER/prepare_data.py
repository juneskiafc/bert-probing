from datasets import load_dataset, concatenate_datasets, Dataset
from pathlib import Path

label_map = {
    'O': 0,
    'B-ORG': 1,
    'I-ORG': 2,
    'B-PER': 3,
    'I-PER': 4,
    'B-LOC': 5,
    'I-LOC': 6
}
label_map = {v:k for k,v in label_map.items()}

def _prepare_data(out_dir, langs_by_split):
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ['train', 'test']:
        if split not in langs_by_split:
            continue
    
        final_out_file = out_dir.joinpath(f'ner_{split}.tsv')
        out_file = out_dir.joinpath(f'ner_{split}_tmp.json')

        if Path(final_out_file).is_file():
            continue
        
        if not Path(out_file).is_file():
            datasets = []
            for lang in langs_by_split[split]:
                dataset = load_dataset('wikiann', lang, split=split)
                datasets.append(dataset)

            dataset = concatenate_datasets(datasets)
            dataset.to_json(out_file)
        
        # load and save as tsv in mtdnn format
        df = Dataset.from_json(str(out_file))
        with open(final_out_file, 'w') as f:
            for i, row in enumerate(df):
                premise = ' '.join(row['tokens'])
                label = label_map[int(row['ner_tags'][0])]
                f.write(f'{i}\t{label}\t{premise}\n')

def prepare_head_probe_data():
    out_dir = Path(f'experiments/NER/head_probe/')
    langs_by_split = {
        'train': ['en'],
        'test': ['en', 'fr', 'de', 'es']
    }

    _prepare_data(out_dir, langs_by_split)

def prepare_finetune_data():
    for setting in ['cross', 'multi']:
        out_dir = Path(f'experiments/NER/{setting}')
        langs_by_split = {
            'train': ['en'],
            'test': ['en', 'fr', 'de', 'es']
        }
        if setting == 'multi':
            langs_by_split['train'].extend(['fr', 'de', 'es'])
        
        _prepare_data(out_dir, langs_by_split)

def prepare_per_language_test_data():
    for language in ['en', 'fr', 'de', 'es']:
        out_dir = Path(f'experiments/NER/{language}')
        langs_by_split = {'test': [language]}
        _prepare_data(out_dir, langs_by_split)

if __name__ == '__main__':
    prepare_per_language_test_data()



