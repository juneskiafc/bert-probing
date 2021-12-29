from datasets import load_dataset, concatenate_datasets, Dataset
from pathlib import Path
import sys
sys.path.append('/home/june/mt-dnn/')
from experiments.exp_def import LingualSetting

def _prepare_data(train_langs, test_langs, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ['train', 'test']:
        if split == 'train' and train_langs is None:
            continue
        elif split == 'test' and test_langs is None:
            continue

        out_file = out_dir.joinpath(f'pawsx_{split}_tmp.json')
        final_out_file = out_dir.joinpath(f'pawsx_{split}.tsv')

        if final_out_file.is_file():
            continue
        else:
            if not Path(out_file).is_file():
                datasets = []
                if split == 'train':
                    langs = train_langs
                else:
                    langs = test_langs
                
                for lang in langs:
                    dataset = load_dataset('paws-x', lang, split=split)
                    datasets.append(dataset)

                dataset = concatenate_datasets(datasets)
                dataset.to_json(out_file)

            # load and save as tsv in mtdnn format
            df = Dataset.from_json(str(out_file))
            with open(final_out_file, 'w') as f:
                for i, row in enumerate(df):
                    premise = row['sentence1']
                    hypo = row['sentence2']
                    label = row['label']
                    f.write(f'{i}\t{label}\t{premise}\t{hypo}\n')

def prepare_finetune_data():
    train_langs_per_setting = {
        LingualSetting.CROSS: ['en'],
        LingualSetting.MULTI: ['en', 'fr', 'de', 'es']
    }
    test_langs = ['en', 'fr', 'de', 'es']

    for setting in [LingualSetting.CROSS, LingualSetting.MULTI]:
        out_dir = Path(f'experiments/PAWSX/{setting.name.lower()}')
        train_langs = train_langs_per_setting[setting]

        _prepare_data(train_langs, test_langs, out_dir)

def prepare_per_language_data():
    langs = ['en', 'fr', 'de', 'es']
    for lang in langs:
        out_dir = Path(f'experiments/PAWSX/{lang}')
        _prepare_data(None, [lang], out_dir)

if __name__ == '__main__':
    prepare_per_language_data()





