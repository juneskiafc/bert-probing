from pathlib import Path
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def move_files():
    root = Path('head_probe_outputs')
    dst_root = Path('head_probe_ranking')
    for task in ['MARC', 'POS', 'NER', 'NLI', 'PAWSX']:
        for setting in ['en', 'foreign', 'combined']:
            for lingual_setting in ['cross', 'multi']:
                src = root.joinpath(task, task, 'results', f'{task.lower()}_{lingual_setting}-{task.lower()}-{setting}.csv')
                dst = dst_root.joinpath(setting, task, src.name)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dst)

def rank_heads():
    for setting in ['en', 'foreign', 'combined']:
        for task in ['MARC', 'POS', 'NER', 'NLI', 'PAWSX']:
            for ls in ['cross', 'multi']:
                path_to_data = Path(f'head_probe_ranking/{setting}/{task}/{task.lower()}_{ls}-{task.lower()}-{setting}.csv')
                data = pd.read_csv(path_to_data, index_col=0).values
                data = data.flatten()
                layer_indices = np.array([i for i in range(12) for j in range(12)])
                sorted_data_indices = np.argsort(data)[::-1]
                sorted_layer_indices = pd.Series(layer_indices[sorted_data_indices])
                sorted_layer_indices.to_csv(f'head_probe_ranking/{setting}/{task}/{task}_{ls}-{task}-{setting}_layer_ranks.np')

def plot_ranked_heads_1():
    colors = [f'tab:{c}' for c in ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']]
    for setting in ['en', 'foreign', 'combined']:
        plt.figure(figsize=(144, 50))
        i = 0
        for task in ['MARC', 'POS', 'NER', 'NLI', 'PAWSX']:
            for ls in ['cross', 'multi']:
                path_to_data = Path(f'head_probe_ranking/{setting}/{task}/{task}_{ls}-{task}-{setting}_layer_ranks.np')
                data = pd.read_csv(path_to_data, index_col=0)
                data += 1
                plt.plot(list(range(144)), data, color=colors[i], label=f'{task}_{ls}')
                i += 1
        
        plt.legend(loc='upper right')
        plt.savefig(f'head_probe_ranking/{setting}/ranks.pdf', bbox_inches='tight')

def plot_ranked_heads_2(setting):
    plt.rcParams["font.family"] = "Times New Roman"
    colors = [f'tab:{c}' for c in ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']]
    colors.extend(['k', 'm'])

    plot_label = False
    fig, ax = plt.subplots(nrows=5, ncols=1, sharex= True, sharey=True, figsize=(15, 30))
    for i, k in enumerate(range(24, 144, 24)):
        if i == 4:
            plot_label = True
        for task in ['MARC', 'POS', 'NER', 'NLI', 'PAWSX']:
            for ls in ['cross', 'multi']:
                accumulated = [0 for _ in range(12)]
                path_to_data = Path(f'head_probe_ranking/{setting}/{task}/{task}_{ls}-{task}-{setting}_layer_ranks.np')
                data = pd.read_csv(path_to_data, index_col=0)[:k]
                for _, d in data.iterrows():
                    accumulated[int(d)] += 1
                accumulated = [a / sum(accumulated) for a in accumulated]

                if task == 'PAWSX':
                    bar_name = f'PI_{ls}-ling'
                elif task == 'MARC':
                    bar_name = f'SA_{ls}-ling'
                elif task == 'NLI':
                    bar_name = f'XNLI_{ls}-ling'
                else:
                    bar_name = f'{task}_{ls}-ling'
                
                sum_ = 0
                for j in range(12):
                    if plot_label:
                        label = j+1
                    else:
                        label = f'_{j+1}'

                    ax[i].bar([bar_name], accumulated[j], color=colors[j], bottom=sum_, label=label)
                    ax[i].set_ylabel('Layer-wise distribution of contributive attention heads')
                    ax[i].set_title(f'k = {k}')
                    sum_ += accumulated[j]
                plot_label = False

    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.ylim(0, 1)
    plt.xlabel('Model')
    fig.legend(loc='lower center', ncol=12, bbox_to_anchor=(0.5, 0.08))
    plt.rcParams.update({'font.size': 20})
    plt.savefig(f'head_probe_ranking/{setting}/ranks.pdf', bbox_inches='tight')
    print(f'head_probe_ranking/{setting}/ranks.pdf')

def plot_ranked_heads_3():
    colors = [f'tab:{c}' for c in ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']]
    colors.extend(['k', 'm'])
    scores = [i for i in range(1, 145)][::-1]
    # scores = [(s-1)/(144-1) for s in scores]
    scores = [s/sum(scores) for s in scores]

    for setting in ['en', 'foreign', 'combined']:
        fig, ax = plt.subplots(figsize=(14, 14))
        plot_label = True
        for task in ['MARC', 'POS', 'NER', 'NLI', 'PAWSX']:
            for ls in ['cross', 'multi']:
                accumulated = [0 for _ in range(12)]
                path_to_data = Path(f'head_probe_ranking/{setting}/{task}/{task}_{ls}-{task}-{setting}_layer_ranks.np')
                data = pd.read_csv(path_to_data, index_col=0)
                for idx, d in data.iterrows():
                    accumulated[int(d)] += scores[idx]
                sum_ = 0
                for j in range(12):
                    if plot_label:
                        ax.bar([f'{task}_{ls}'], accumulated[j], color=colors[j], bottom=sum_, label=j)
                    else:
                        ax.bar([f'{task}_{ls}'], accumulated[j], color=colors[j], bottom=sum_, label=f'_{j}')
                    sum_ += accumulated[j]
                plot_label = False
            
        plt.legend(loc='upper right')
        plt.savefig(f'head_probe_ranking/{setting}/ranks_weighted.pdf', bbox_inches='tight')

if __name__ == '__main__':
    # rank_heads()
    for setting in ['en', 'foreign', 'combined']:
        plot_ranked_heads_2(setting)
