from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import numpy as np
import matplotlib.colors as mcolors

def move_files():
    root = Path('head_probe_outputs')
    dst_root = Path('head_probe_ranking')
    for upstream_task in ['MARC', 'POS', 'NER', 'NLI', 'PAWSX']:
        for downstream_task in ['MARC', 'POS', 'NER', 'NLI', 'PAWSX']:
            for lingual_setting in ['cross', 'multi']:
                for setting in ['combined']:
                    upstream_task_lower = upstream_task.lower()
                    src = root.joinpath(
                        upstream_task,
                        downstream_task,
                        'results',
                        f'{upstream_task_lower}_{lingual_setting}-{downstream_task.lower()}-{setting}.csv')
                    dst = dst_root.joinpath(setting, upstream_task, downstream_task, lingual_setting, src.name)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(src, dst)


def rank_heads():
    setting = 'combined'
    for upstream_task in ['MARC', 'POS', 'NER', 'NLI', 'PAWSX']:
        for downstream_task in ['MARC', 'POS', 'NER', 'NLI', 'PAWSX']:
            for ls in ['cross', 'multi']:
                path_to_data = Path('head_probe_ranking').joinpath(
                    setting,
                    upstream_task,
                    downstream_task,
                    ls,
                    f'{upstream_task.lower()}_{ls}-{downstream_task.lower()}-{setting}.csv'
                )
                out_file = Path('head_probe_ranking').joinpath(
                    setting,
                    upstream_task,
                    downstream_task,
                    ls,
                    f'{upstream_task.lower()}_{ls}-{downstream_task.lower()}-{setting}_layer_ranks.csv'
                )
                data = pd.read_csv(path_to_data, index_col=0).values
                data = data.flatten()
                layer_indices = np.array([i for i in range(12) for j in range(12)])
                sorted_data_indices = np.argsort(data)[::-1]
                sorted_layer_indices = pd.Series(layer_indices[sorted_data_indices])
                sorted_layer_indices.to_csv(out_file)


def preprocess_ranked_heads(setting):
    all_data = {}

    for k in range(24, 144, 24):
        data_for_k = {}
        for downstream_task in ['POS', 'NER', 'PAWSX', 'MARC', 'NLI']:
            for upstream_task in ['POS', 'NER', 'PAWSX', 'MARC', 'NLI']:
                for ls in ['cross', 'multi']:
                    path_to_data = Path('head_probe_ranking').joinpath(
                        setting,
                        upstream_task,
                        downstream_task,
                        ls,
                        f'{upstream_task.lower()}_{ls}-{downstream_task.lower()}-{setting}_layer_ranks.csv'
                    )
                    
                    accumulated = [0 for _ in range(12)]
                    data = pd.read_csv(path_to_data, index_col=0)[:k]

                    # get no. top heads per layer, normalize, and cumulative sum
                    for _, d in data.iterrows():
                        accumulated[int(d)] += 1
                    accumulated = [a / sum(accumulated) for a in accumulated]

                    if downstream_task not in data_for_k:
                        data_for_k[downstream_task] = {f'{upstream_task}_{ls}': accumulated}
                    else:
                        data_for_k[downstream_task][f'{upstream_task}_{ls}'] = accumulated
        all_data[k] = data_for_k
    
    return all_data


def get_colors_as_hex():
    mcolors.TABLEAU_COLORS
    mcolors.XKCD_COLORS
    mcolors.CSS4_COLORS

    cs = [
        "blue",
        "orange",
        "green",
        "red",
        "purple",
        'brown',
        'pink',
        'gray',
        'olive',
        'cyan']
    cs = [f'tab:{c}' for c in cs]
    cs.extend(['k', 'yellow'])

    #Base colors are in RGB so they need to be converted to HEX
    BASE_COLORS_hex = {name:mcolors.rgb2hex(color) for name,color in mcolors.BASE_COLORS.items()}

    all_named_colors = {}
    all_named_colors.update(mcolors.TABLEAU_COLORS)
    all_named_colors.update(BASE_COLORS_hex)
    all_named_colors.update(mcolors.CSS4_COLORS)
    all_named_colors.update(mcolors.XKCD_COLORS)

    return [all_named_colors[c] for c in cs]


def organize_data_self_probing(data, ks, ds_tasks):
    layer_dfs = []
    for layer in range(12):
        d = []
        for k in ks:
            data_for_k = []
            models = []
            for us_task in ds_tasks:
                for us_ls in ['cross', 'multi']:
                    data_for_k.append(data[k][us_task][f'{us_task}_{us_ls}'])
                    models.append(f'{us_task}_{us_ls}')
            df_for_k = pd.DataFrame(data_for_k)
            df_for_k.index = models
            d.append(df_for_k.iloc[:, layer])

        d = pd.concat(d, axis=1).T
        d.index = ks
        layer_dfs.append(d)
    
    return layer_dfs


def organize_data_cross_probing(data, k, ds_tasks):
    layer_dfs = []
    for layer in range(12):
        d = []
        for downstream_task in ds_tasks:
            data_for_ds_task = data[k][downstream_task]
            df_for_ds_task = pd.DataFrame(data_for_ds_task).T
            d.append(df_for_ds_task.iloc[:, layer])
        d = pd.concat(d, axis=1).T
        d.index = ['POS', 'NER', 'PAWSX', 'MARC', 'NLI']
        layer_dfs.append(d)

    return layer_dfs


def plot(setting='combined', self_probing=True, cross_probing=True):
    font_size = 14
    parameters = {
        'axes.labelsize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'axes.titlesize': font_size,
        'legend.fontsize': font_size,
        "legend.title_fontsize": font_size
    }
    plt.rcParams.update(parameters)
    colors = get_colors_as_hex()
    colors = colors[::-1]

    ds_tasks = ['POS', 'NER', 'PAWSX', 'MARC', 'NLI']
    ds_tasks_real_names = ['POS', 'NER', 'PI', 'SA', 'XNLI']
    ks = list(range(24, 144, 24))

    models_to_plot = itertools.product(ds_tasks, ['cross', 'multi'])
    model_names_to_plot = itertools.product(ds_tasks_real_names, ['cross', 'multi'])
    models_to_plot = [f'{m[0]}_{m[1]}' for m in models_to_plot]
    model_names_to_plot = [f'{m[0]}-{m[1]}-ling' for m in model_names_to_plot]

    data = preprocess_ranked_heads(setting)

    # self probing
    if self_probing:
        layer_dfs = organize_data_self_probing(data, ks, ds_tasks)
        _, super_ax = plot_groupby(layer_dfs, ks, models_to_plot, model_names_to_plot, colors, 'k=')
        super_ax.set_xlabel("Models", labelpad=100, fontweight='bold')
        super_ax.set_ylabel("Layer-wise distribution of top-k attention heads", labelpad=20, fontweight='bold')

        out_file = f'head_probe_ranking/self_probing.pdf'
        plt.savefig(out_file, bbox_inches='tight')
        print(out_file)

    # cross probing
    if cross_probing:
        for k in ks:
            layer_dfs = organize_data_cross_probing(data, k, ds_tasks)
            _, super_ax = plot_groupby(layer_dfs, ds_tasks, models_to_plot, model_names_to_plot, colors, '')

            super_ax.set_xlabel("Models", labelpad=100, fontweight='bold')
            top_x_ax = super_ax.secondary_xaxis('top')
            top_x_ax.set_xlabel('Task', labelpad=30, fontweight='bold')
            top_x_ax.spines['top'].set_visible(False)
            top_x_ax.set_xticks([])
            super_ax.set_ylabel(f"Layer-wise distribution of top-{k} attention heads", labelpad=20, fontweight='bold')

            out_file = f'head_probe_ranking/cross_probing_k={k}.pdf'
            plt.savefig(out_file, bbox_inches='tight')
            print(out_file)


def plot_groupby(
    layer_dfs,
    groupby,
    models_to_plot,
    model_names_to_plot,
    colors,
    groupby_prefix,
):
    global_fig, axes = plt.subplots(1, len(groupby), sharey=True, figsize=(20, 10))
    
    for i, ax_for_k in enumerate(axes):
        bottom = 0
        for layer in range(12):
            bar = ax_for_k.bar(
                model_names_to_plot,
                layer_dfs[layer].loc[groupby[i], models_to_plot],
                bottom=bottom,
                color=colors[layer]
            )
            if i == 0:
                bar.set_label(str(layer+1))
            bottom += layer_dfs[layer].loc[groupby[i], models_to_plot]

        ax_for_k.set_frame_on(False)
        ax_for_k.get_yaxis().set_visible(False)

        ax_for_k.set_ybound(lower=0, upper=1)
        ax_for_k.set_xticks(
            ax_for_k.get_xticks(),
            model_names_to_plot,
            rotation=45,
            ha='right',
            rotation_mode='anchor')

        ax_for_k.set_title(f'{groupby_prefix}{groupby[i]}')

    handles, labels = axes[0].get_legend_handles_labels()
    global_fig.legend(
        handles[::-1], labels[::-1],
        loc='right',
        title='Layers',
        bbox_to_anchor=(0.98, 0.5)
    )

    super_ax = global_fig.add_subplot(111, facecolor='none')
    super_ax.spines['top'].set_visible(False)
    super_ax.spines['right'].set_visible(False)
    super_ax.set_xticks([])
    super_ax.set_yticks(np.arange(0, 1.1, 0.1))

    return global_fig, super_ax

if __name__ == '__main__':
    move_files()
    rank_heads()
    plot()