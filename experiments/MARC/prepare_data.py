from datasets import load_dataset, concatenate_datasets, Dataset
import pandas as pd
from pathlib import Path

# en fr de es
split_to_langs = {
    'train': ['en'],
    'test': ['en', 'fr', 'de', 'es']
}

for split in ['train', 'test']:
    out_file = f'experiments/MARC/marc_{split}_tmp.json'
    if not Path(out_file).is_file():
        datasets = []
        for lang in split_to_langs[split]:
            dataset = load_dataset('amazon_reviews_multi', lang, split=split)
            datasets.append(dataset)

        dataset = concatenate_datasets(datasets)
        dataset.to_json(out_file)

    # load and save as tsv in mtdnn format
    df = Dataset.from_json(out_file)
    final_out_file = f'experiments/MARC/marc_{split}.tsv'
    with open(final_out_file, 'w') as f:
        for i, row in enumerate(df):
            premise = row['review_body']
            label = row['stars']
            f.write(f'{i}\t{label}\t{premise}\n')





