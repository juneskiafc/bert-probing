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

def plot_ranked_heads_2():
    colors = [f'tab:{c}' for c in ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']]
    colors.extend(['k', 'm'])

    for k in range(24, 144, 24):
        for setting in ['en', 'foreign', 'combined']:
            fig, ax = plt.subplots(figsize=(14, 14))
            plot_label = True
            for task in ['MARC', 'POS', 'NER', 'NLI', 'PAWSX']:
                for ls in ['cross', 'multi']:
                    accumulated = [0 for _ in range(12)]
                    path_to_data = Path(f'head_probe_ranking/{setting}/{task}/{task}_{ls}-{task}-{setting}_layer_ranks.np')
                    data = pd.read_csv(path_to_data, index_col=0)[:k]
                    for _, d in data.iterrows():
                        accumulated[int(d)] += 1
                    sum_ = 0
                    for j in range(12):
                        if plot_label:
                            ax.bar([f'{task}_{ls}'], accumulated[j], color=colors[j], bottom=sum_, label=j)
                        else:
                            ax.bar([f'{task}_{ls}'], accumulated[j], color=colors[j], bottom=sum_, label=f'_{j}')
                        sum_ += accumulated[j]
                    plot_label = False
            
            plt.legend(loc='upper right')
            plt.savefig(f'head_probe_ranking/{setting}/ranks_{k}.pdf', bbox_inches='tight')

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
    plot_ranked_heads_3()
