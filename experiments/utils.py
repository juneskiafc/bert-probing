import argparse
import subprocess
from pathlib import Path

def multi_dataset_prepro_wrapper(datasets):
    for dataset in datasets:
        command = f'python prepro_std.py --dataset {dataset}'
        print(command)
        command = command.split(' ')
        subprocess.run(command)

def combine_mtdnn_jsons(datasets, out_dir):
    out_dir = Path('experiments').joinpath(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    task = out_dir.parent.name

    for split in ['train', 'test']:
        out_file = out_dir.joinpath('bert-base-multilingual-cased', f'{task.lower()}_{split}.json')
        out_file.parent.mkdir(exist_ok=True)
        
        if not out_file.is_file():    
            with open(out_file, 'w') as f:
                for dataset in datasets:
                    dataset_file = Path('experiments').joinpath(
                        dataset,
                        'bert-base-multilingual-cased',
                        f'{task.lower()}_{split}.json')
                    with open(dataset_file, 'r') as fr:
                        for line in fr:
                            f.write(line)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+')
    parser.add_argument('--multi_dataset_prepro', action='store_true')
    parser.add_argument('--combine_jsons', action='store_true')
    parser.add_argument('--out_dir', default='')
    args = parser.parse_args()

    if args.multi_dataset_prepro:
        multi_dataset_prepro_wrapper(args.datasets)
    elif args.combine_jsons:
        combine_mtdnn_jsons(args.datasets, args.out_dir)
