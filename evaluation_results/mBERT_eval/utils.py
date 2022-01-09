from pathlib import Path
import pandas as pd
from typing import Union, List, Tuple
from matplotlib import pyplot as plt
import seaborn as sns

def create_heatmap(
    data_csv_path: str = '',
    data_df: Union[pd.DataFrame, None] = None,
    row_labels: List[str] = None,
    column_labels: List[str] = None,
    xaxlabel: str = None,
    yaxlabel: str = None,
    invert_y: bool = False,
    figsize: Tuple[int, int] = (14, 14),
    fontsize: int = 14,
    cmap: str = 'RdYlGn',
    out_file: str= ''):

    # read data if dataframe not directly supplied.
    if data_df is None:
        data_df = pd.read_csv(data_csv_path, index_col=0)
        assert len(out_file) > 0, f'invalid csv: {data_csv_path}'
    
    plt.figure(figsize=figsize)
    annot_kws = {
        "fontsize":fontsize,
    }
    heatmap = sns.heatmap(
        data_df.to_numpy(),
        cbar=False,
        annot=True,
        annot_kws=annot_kws,
        fmt=".2f",
        cmap=cmap)

    if invert_y:
        heatmap.invert_yaxis()

    heatmap.set_xlabel(xaxlabel, fontsize=fontsize)
    heatmap.set_ylabel(yaxlabel, fontsize=fontsize)

    heatmap.set_yticklabels(row_labels, rotation=0, fontsize=fontsize)
    heatmap.set_xticklabels(column_labels, rotation=0, fontsize=fontsize)

    heatmap.tick_params(labeltop=True, labelbottom=False, bottom=False)

    fig = heatmap.get_figure()
    fig.savefig(Path(out_file).with_suffix('.pdf'), bbox_inches='tight')

dfs = []
indices = []
for task in ['MARC', 'NER', 'PAWSX', 'POS']:
    for setting in ['multi', 'cross']:
        results_file = f'evaluation_results/{task}_{setting}.csv'
        results_df = pd.read_csv(results_file, index_col=0)
        dfs.append(results_df)
        indices.append(f'{task}_{setting}')
        columns = results_df.columns

dfs = pd.concat(dfs, axis=1)
dfs = dfs.T
dfs.index = indices
dfs.to_csv('evaluation_results/downstream_task_f1macs.csv')

create_heatmap(data_df=dfs, row_labels=dfs.index, column_labels=dfs.columns, out_file='evaluation_results/downstream_task_f1macs')




