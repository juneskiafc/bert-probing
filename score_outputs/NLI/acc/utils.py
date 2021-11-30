from pathlib import Path
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

def save_acc_matrix_as_heatmap(input_csv='', file_df=None, invert_y=False, axlabels=None, out_file=''):
    if file_df is None:
        file_df = pd.read_csv(input_csv, index_col=0)
    row_labels = file_df.index
    column_labels = file_df.columns
    row_labels_new = []
    for l in row_labels:
        if l == 'cross' or l == 'multi':
            row_labels_new.append(f'XNLI-{l}-ling')
        else:
            lang = l.split("_")[1]
            row_labels_new.append(f'XNLI-multi-ling-{lang.upper()}')

    row_labels = row_labels_new
    column_labels = [l.split('-')[1].upper() for l in column_labels]
    data = file_df.to_numpy()

    annot_kws = {
        "fontsize":14,
    }

    fig, heatmap = plt.subplots(figsize=(14,1))
    heatmap = sns.heatmap(
        data,
        cbar=False,
        annot=True,
        annot_kws=annot_kws,
        fmt=".2f",
        xticklabels=column_labels,
        yticklabels=row_labels,
        cmap='RdYlGn',
        ax=heatmap)

    heatmap.set_yticklabels(row_labels, rotation=0, fontsize=14)
    heatmap.set_xticklabels(column_labels, rotation=0, fontsize=14)

    if invert_y:
        heatmap.invert_yaxis()
    if axlabels is not None:
        heatmap.set_ylabel(axlabels[1], loc='top')
        heatmap.set_xlabel(axlabels[0])

    fig = heatmap.get_figure()

    if len(input_csv) > 0: 
        out_file = Path(input_csv).with_suffix('.pdf')
        print(f'saving to {out_file}')
        fig.savefig(out_file, bbox_inches='tight')
    else:
        fig.savefig(Path(out_file).with_suffix('.png'), bbox_inches='tight')

if __name__ == "__main__":
    root = 'score_outputs/NLI/acc/acc_ml_ltr.csv'
    save_acc_matrix_as_heatmap(root)