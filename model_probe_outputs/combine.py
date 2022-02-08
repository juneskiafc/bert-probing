from pathlib import Path
import pandas as pd

for setting in ['multi', 'cross']:
    for model_name in ['MARC', 'NLI', 'PAWSX']:
        model_name = f'{model_name}_{setting}'
        all_data = []
        root = Path(f'model_probe_outputs/{model_name}')
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