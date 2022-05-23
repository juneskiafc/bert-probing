from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# for subtask in Path('head_probe_outputs/NLI').iterdir():
#     for file in subtask.joinpath('results').iterdir():
#         split_file = file.name.split("_")
#         if split_file[0] == 'xnli':
#             split_file[0] = 'nli'
#             new_file_name = file.parent.joinpath("_".join(split_file))
#             shutil.move(file, new_file_name)

def get_sorted_head_idxs(finetuned_task, probe_task, setting, lang_setting):
    root_model = Path(f'head_probe_outputs/{finetuned_task}/{probe_task}/results/{finetuned_task.lower()}_{setting}-{probe_task.lower()}-{lang_setting}.csv')
    root_data = pd.read_csv(root_model, index_col=0).values.flatten()
    head_indices = np.array([i for i in range(144)])
    index_sorted = np.argsort(root_data)
    return head_indices[index_sorted]

colors = [f'tab:{c}' for c in ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']]
for lang_setting in ['combined', 'en', 'foreign']:
    for task in ['MARC', 'NER', 'POS', "NLI", 'PAWSX']:
        root_head_indices_sorted = get_sorted_head_idxs(task, task, 'multi', lang_setting)
        fig, ax = plt.subplots(figsize=(14, 14))
        
        i = 0
        for subtask in ['MARC', 'NER', 'POS', "NLI", 'PAWSX']:
            for setting in ['cross', 'multi']:
                if f'{subtask}_{setting}' == f'{task}_multi':
                    continue
            
                sub_head_indices_sorted = get_sorted_head_idxs(subtask, task, setting, lang_setting)
                data = [0 for _ in range(144)]
                for k in range(1, 145):
                    root_subset = root_head_indices_sorted[:k]
                    sub_subset = sub_head_indices_sorted[:k]
                    intersection = len(np.intersect1d(root_subset, sub_subset))
                    data[k-1] = intersection

                ax.plot(list(range(144)), data, label=f'{subtask}_{setting}', color=colors[i])
                i += 1
                
        plt.legend(loc='upper right')
        Path(f'head_probe_intersection/{lang_setting}').mkdir(exist_ok=True)
        plt.savefig(f'head_probe_intersection/{lang_setting}/{task}.png', bbox_inches='tight')
    raise ValueError
