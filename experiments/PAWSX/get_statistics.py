from datasets import load_dataset

def get_num_instances():
    for split in ['train', 'test']:
        for lang in ['en', 'fr', 'es', 'de']:
            dataset = load_dataset('paws-x', lang, split=split)
            print(f'{lang}, {split}, {len(dataset)}')

if __name__ == '__main__':
    get_num_instances()



