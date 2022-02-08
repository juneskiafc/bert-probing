from pathlib import Path
import pandas as pd

def combine_seeds():
    all_data = []
    model_name = 'NLI_multi'
    root = Path(f'model_probe_outputs/cross_training/{model_name}')
    for task in ['NLI', 'POS', 'NER', 'PAWSX', 'MARC']:
        data = []
        for seed in range(3):
            file = root.joinpath(f'{task}_results_{seed}.csv')
            data.append(pd.read_csv(file, index_col=0))
        data = pd.concat(data, axis=0)
        data.index = [f'{model_name}_{i}' for i in range(3)]
        all_data.append(data)

    all_data = pd.concat(all_data, axis=1)
    all_data.to_csv(root.joinpath('results.csv'))

def combine_individual_tasks():
    model_name = 'mBERT'
    all_data = []
    root = Path(f'model_probe_outputs/multi_training/{model_name}')

    for task in ['POS', 'NER', 'PAWSX', 'MARC']:
        file = root.joinpath(f'{task}_results.csv')
        data = pd.read_csv(file, index_col=0)
        all_data.append(data)
    
    all_data = pd.concat(all_data, axis=1)
    all_data.columns = ['POS', 'NER', 'PAWSX', 'MARC']
    all_data.index = [model_name]
    
    all_data.to_csv(f'model_probe_outputs/multi_training/{model_name}/results.csv')

if __name__ == '__main__':
    combine_individual_tasks()
