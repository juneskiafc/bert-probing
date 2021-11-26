from datasets import load_dataset, concatenate_datasets, Dataset
import pandas as pd
from pathlib import Path
import json

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

# en fr de es
split_to_langs = {
    'train': ['en'],
    'test': ['en', 'fr', 'de', 'es']
}

for split in ['train', 'test']:
    out_file = f'experiments/NER/ner_{split}_tmp.json'
    if not Path(out_file).is_file():
        datasets = []
        for lang in split_to_langs[split]:
            dataset = load_dataset('wikiann', lang, split=split)
            datasets.append(dataset)

        dataset = concatenate_datasets(datasets)
        dataset.to_json(out_file)

    # load and save as tsv in mtdnn format
    df = Dataset.from_json(out_file)
    final_out_file = f'experiments/NER/ner_{split}.tsv'
    with open(final_out_file, 'w') as f:
        for i, row in enumerate(df):
            premise = ' '.join(row['tokens'])
            label = label_map[int(row['ner_tags'][0])]
            f.write(f'{i}\t{label}\t{premise}\n')





