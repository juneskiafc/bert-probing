from pathlib import Path

def split_dataset(n, task, datasets):
    for dataset in datasets:
        dataset_file = f'experiments/{task}/{dataset}/{task.lower()}_test.tsv'

        n_lines = 0
        with open(dataset_file, 'r') as f:
            for line in f:
                n_lines += 1

        n_split = 0
        lines_per_split = n_lines // n

        with open(dataset_file, 'r') as fr:
            for i, line in enumerate(fr):
                if i % lines_per_split == 0:
                    print(i, lines_per_split)
                    split_file_name = Path(f'experiments/{task}/{dataset}_{n_split}/{task.lower()}_test.tsv')
                    split_file_name.parent.mkdir(parents=True, exist_ok=True)
                    fw = open(split_file_name, 'w')
                    n_split += 1

                fw.write(line)

if __name__ == '__main__':
    split_dataset(2, 'MARC', ['en', 'es', 'fr'])
                








