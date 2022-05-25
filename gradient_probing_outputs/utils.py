import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

rhos = pd.read_csv('gradient_probing_outputs/correlations/rhos.csv', index_col=0)
new_rhos = pd.DataFrame(np.zeros((5, 10)))
new_rhos.index = ['POS', 'NER', 'PI', 'SA', 'XNLI']
new_rhos_cols = []
for c in ['POS', 'NER', 'PI', 'SA', 'XNLI']:
    for s in ['cross', 'multi']:
        new_rhos_cols.append(f'{c}-{s}')
new_rhos.columns = new_rhos_cols

task_dict = {
    'POS': 'POS',
    'NER': 'NER',
    'PI': 'PAWSX',
    'SA': 'MARC',
    'XNLI': 'NLI'
}
for model in new_rhos.columns:
    for task in ['POS', 'NER', 'PI', 'SA', 'XNLI']:
        model_task, model_setting = model.split("-")
        if model_task == 'XNLI':
            rho = rhos.loc[f'NLI_{model_setting}', f'{task_dict[task]}_{model_setting}']
        elif model_task == 'PI':
            rho = rhos.loc[f'PAWSX_{model_setting}', f'{task_dict[task]}_{model_setting}']
        elif model_task == 'SA':
            rho = rhos.loc[f'MARC_{model_setting}', f'{task_dict[task]}_{model_setting}']
        else:
            rho = rhos.loc[f'{model_task}_{model_setting}', f'{task_dict[task]}_{model_setting}']
        new_rhos.loc[task, model] = rho

new_rhos.columns = [f'{c}-ling' for c in new_rhos.columns]

font_size = 40
for data in [(f'rhos.pdf', rhos)]:
    plt.figure(figsize=(20, 20))
    annot_kws = {'fontsize': font_size}
    ax = sns.heatmap(
        new_rhos,
        cbar=False,
        annot=True,
        annot_kws=annot_kws,
        fmt=".2f",
        square=True)

    ax.tick_params(axis='y', labelsize=font_size, labelrotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    ax.tick_params(axis='x', labelsize=font_size)

    fig = ax.get_figure()
    fig.savefig('gradient_probe_outputs/correlations/rhos.pdf', bbox_inches='tight')