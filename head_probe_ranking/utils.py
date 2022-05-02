from collections import defaultdict
from multiprocessing.sharedctypes import Value
from pathlib import Path
import shutil
import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import numpy as np
import matplotlib.colors as mcolors
import altair_saver

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

def cross_probing_move_files():
    root = Path('head_probe_outputs')
    dst_root = Path('head_probe_ranking')
    for upstream_task in ['MARC', 'POS', 'NER', 'NLI', 'PAWSX']:
        for downstream_task in ['MARC', 'POS', 'NER', 'NLI', 'PAWSX']:
            for lingual_setting in ['cross', 'multi']:
                for setting in ['combined']:
                    if upstream_task == 'NLI':
                        upstream_task_lower = 'xnli'
                    else:
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
    # for setting in ['en', 'foreign', 'combined']:
    setting = 'combined'
    for upstream_task in ['MARC', 'POS', 'NER', 'NLI', 'PAWSX']:
        for downstream_task in ['MARC', 'POS', 'NER', 'NLI', 'PAWSX']:
            for ls in ['cross', 'multi']:
                path_to_data = Path(f'head_probe_ranking/{setting}/{upstream_task}/{downstream_task}/{ls}/{upstream_task.lower()}_{ls}-{downstream_task.lower()}-{setting}.csv')
                data = pd.read_csv(path_to_data, index_col=0).values
                data = data.flatten()
                layer_indices = np.array([i for i in range(12) for j in range(12)])
                sorted_data_indices = np.argsort(data)[::-1]
                sorted_layer_indices = pd.Series(layer_indices[sorted_data_indices])
                sorted_layer_indices.to_csv(f'head_probe_ranking/{setting}/{upstream_task}/{downstream_task}/{ls}/{upstream_task.lower()}_{ls}-{downstream_task.lower()}-{setting}_layer_ranks.np')

def preprocess_ranked_heads(setting):
    dataframes = []

    for task in ['POS', 'NER', 'PAWSX', 'MARC', 'NLI']:
        for ls in ['cross', 'multi']:
            data_for_model = []

            for k in range(24, 144, 24):
                path_to_data = Path(f'head_probe_ranking/{setting}/{task}/{task}_{ls}-{task}-{setting}_layer_ranks.np')
                accumulated = [0 for _ in range(12)]
                data = pd.read_csv(path_to_data, index_col=0)[:k]

                # get no. top heads per layer, normalize, and cumulative sum
                for _, d in data.iterrows():
                    accumulated[int(d)] += 1
                accumulated = [a / sum(accumulated) for a in accumulated]
                data_for_model.append(accumulated)

            df_for_model = pd.DataFrame(data_for_model)
            df_for_model.index = list(range(24, 144, 24))
            dataframes.append(df_for_model)
    
    return dataframes

def preprocess_ranked_heads_cross_probing(setting):
    all_data = {}

    for k in range(24, 144, 24):
        data_for_k = {}
        for downstream_task in ['POS', 'NER', 'PAWSX', 'MARC', 'NLI']:
            for upstream_task in ['POS', 'NER', 'PAWSX', 'MARC', 'NLI']:
                if downstream_task != upstream_task:
                    for ls in ['cross', 'multi']:
                        path_to_data = Path(f'head_probe_ranking/{setting}/{upstream_task}/{downstream_task}/{ls}/{upstream_task.lower()}_{ls}-{downstream_task.lower()}-{setting}_layer_ranks.np')
                        
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

def plot_altair(dfs):
    columns = []
    for pair in itertools.product(['POS', 'NER', 'PI', 'SA', 'XNLI'], ['cross-ling', 'multi-ling']):
        columns.append(f'{pair[0]}-{pair[1]}')
    
    def prep_df(df, layer_idx):
        df = df.stack().reset_index()
        df.columns = ['k', 'model', 'nHeads']
        df['Layer'] = layer_idx
        return df

    layer_dfs = []
    for layer in range(12):
        d = []
        for df in dfs:
            d.append(df.iloc[:, layer])
        d = pd.concat(d, axis=1)
        d.index = [f'k={k}' for k in list(range(24, 144, 24))]
        d.columns = columns
        layer_dfs.append(d)
    
    dfs = []
    for i in range(12, 0, -1):
        prepped_df = prep_df(layer_dfs[i-1], i)
        dfs.append(prepped_df)
    
    df = pd.concat(dfs)
    colors = get_colors_as_hex()
    
    chart = alt.Chart(df).mark_bar().encode(
        # tell Altair which field to group columns on
        x=alt.X('model:N', title='', axis=alt.Axis(labelAngle=-45), sort=columns),

        # tell Altair which field to use as Y values and how to calculate
        y=alt.Y(
            'nHeads:Q',
            axis=alt.Axis(grid=False, title='layer-wise distribution of top-k attention heads'),
            scale=alt.Scale(domain=(0, 1))),

        # tell Altair which field to use to use as the set of columns to be  represented in each group
        # hack: title is model so we can move it to the bottom later
        column=alt.Column('k:O', title='model', sort=[f'k={k}' for k in list(range(24, 144, 24))]),

        # tell Altair which field to use for color segmentation 
        color=alt.Color('Layer:N', scale=alt.Scale(range=colors), sort=list(range(1, 13))[::-1]),

        order=alt.Order('Layer', sort='ascending')
    ).configure_view(
        strokeOpacity=0
    ).configure_header(
        titleOrient='bottom'
    )

    return chart

def plot_pyplot_cross_probing(data):
    font_size = 14
    parameters = {
        'axes.labelsize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'axes.titlesize': font_size,
        'legend.fontsize': font_size,
    }
    plt.rcParams.update(parameters)
    colors = get_colors_as_hex()
    colors = colors[::-1]

    for k in range(24, 144, 24):
        global_fig = plt.figure(constrained_layout=True, figsize=(20, 10))
        subfigs = global_fig.subfigures(1, 1)

        layer_dfs = []
        for layer in range(12):
            d = []
            for downstream_task in ['POS', 'NER', 'PAWSX', 'MARC', 'NLI']:
                data_for_ds_task = data[k][downstream_task]
                df_for_ds_task = pd.DataFrame(data_for_ds_task).T
                d.append(df_for_ds_task.iloc[:, layer])
            d = pd.concat(d, axis=1).T
            d.index = ['POS', 'NER', 'PAWSX', 'MARC', 'NLI']
            layer_dfs.append(d)
        
        ds_tasks = ['POS', 'NER', 'PAWSX', 'MARC', 'NLI']
        ds_tasks_real_names = ['POS', 'NER', 'PI', 'SA', 'XNLI']

        axes = subfigs.subplots(1, 5, sharey=True)
        plt.subplots_adjust(wspace=0.0001)
        bounds = list(axes[0]._position.bounds)
        bounds[1] += 0.02
        bounds[0] -= 0.081
        bounds[3] += 0.074
        bounds[2] += 0.813
        global_ax = subfigs.add_axes(bounds)
        global_ax.patch.set_alpha(0)
        global_ax.xaxis.set_ticks([])
        global_ax.yaxis.set_ticks([])

        for i, ax_for_ds_task in enumerate(axes):
            models_to_plot = itertools.product([ds_tasks[j] for j in range(5)], ['cross', 'multi'])
            model_names_to_plot = itertools.product([ds_tasks_real_names[j] for j in range(5)], ['cross', 'multi'])
            models_to_plot = [f'{m[0]}_{m[1]}' for m in models_to_plot]
            model_names_to_plot = [f'{m[0]}-{m[1]}-ling' for m in model_names_to_plot]

            bottom = 0
            for layer in range(12):
                bar = ax_for_ds_task.bar(
                    model_names_to_plot,
                    layer_dfs[layer].loc[ds_tasks[i], models_to_plot],
                    bottom=bottom,
                    color=colors[layer])
                if i == 0:
                    bar.set_label(str(layer+1))
                bottom += layer_dfs[layer].loc[ds_tasks[i], models_to_plot]

            ax_for_ds_task.set_frame_on(False)
            if i > 0:
                ax_for_ds_task.get_yaxis().set_visible(False)

            ax_for_ds_task.set_xticks(
                ax_for_ds_task.get_xticks(),
                model_names_to_plot,
                rotation=45,
                ha='right',
                rotation_mode='anchor')

            ax_for_ds_task.set_title(ds_tasks_real_names[i])

        handles, labels = axes[0].get_legend_handles_labels()
        global_fig.legend(handles[::-1], labels[::-1], loc='right', bbox_to_anchor=(1.05, 0.5))
        plt.savefig(f'head_probe_ranking/cross_probing_k={k}.pdf', bbox_inches='tight')
        raise ValueError

def plot_altair_cross_probing(data):
    charts = []
    for k in range(24, 144, 24):
        columns = []
        for pair in itertools.product(['POS', 'NER', 'PI', 'SA', 'XNLI'], ['cross-ling', 'multi-ling']):
            columns.append(f'{pair[0]}-{pair[1]}')
        
        def prep_df(df, layer_idx):
            df = df.stack().reset_index()
            df.columns = ['ds_task', 'model', 'nHeads']
            df['Layer'] = layer_idx
            return df

        layer_dfs = []
        for layer in range(12):
            d = []
            for downstream_task in ['POS', 'NER', 'PAWSX', 'MARC', 'NLI']:
                data_for_ds_task = data[k][downstream_task]
                df_for_ds_task = pd.DataFrame(data_for_ds_task).T
                d.append(df_for_ds_task.iloc[:, layer])
            d = pd.concat(d, axis=1).T
            d.index = ['POS', 'NER', 'PAWSX', 'MARC', 'NLI']
            layer_dfs.append(d)
        
        dfs = []
        for i in range(12, 0, -1):
            prepped_df = prep_df(layer_dfs[i-1], i)
            dfs.append(prepped_df)
        
        df = pd.concat(dfs)
        colors = get_colors_as_hex()
        
        chart = alt.Chart(df).mark_bar().encode(
            # tell Altair which field to group columns on
            x=alt.X('model:N', title='', axis=alt.Axis(labelAngle=-45), sort=columns),

            # tell Altair which field to use as Y values and how to calculate
            y=alt.Y(
                'nHeads:Q',
                axis=alt.Axis(grid=False, title='layer-wise distribution of top-k attention heads'),
                scale=alt.Scale(domain=(0, 1))),

            # tell Altair which field to use to use as the set of columns to be represented in each group
            # hack: title is model so we can move it to the bottom later
            column=alt.Column('ds_task:N', title='model', sort=['POS', 'NER', 'PAWSX', 'MARC', 'NLI']),

            # tell Altair which field to use for color segmentation 
            color=alt.Color('Layer:N', scale=alt.Scale(range=colors), sort=list(range(1, 13))[::-1]),

            order=alt.Order('Layer', sort='ascending')
        ).configure_view(
            strokeOpacity=0
        ).configure_header(
            titleOrient='bottom'
        )
        charts.append(chart)
    
    return charts

if __name__ == '__main__':
    # cross_probing_move_files()
    # rank_heads()
    # for setting in ['combined']:
    #     save_path = Path(f'head_probe_ranking/plots/plot_topk/{setting}_ranks.svg')
    #     save_path.parent.mkdir(parents=True, exist_ok=True)
    #     dataframes = preprocess_ranked_heads(setting)
    #     ax = plot_altair(dataframes)
    #     altair_saver.save(ax, str(save_path))

    cross_probing_move_files()
    # for setting in ['combined']:
    #     data = preprocess_ranked_heads_cross_probing(setting)
    #     plot_pyplot_cross_probing(data)
    for setting in ['combined']:
        data = preprocess_ranked_heads_cross_probing(setting)
        axes = plot_altair_cross_probing(data)
        ks = list(range(24, 144, 24))
        for i, ax in enumerate(axes):
            save_path = Path(f'head_probe_ranking/plots/cross_head_probing/{ks[i]}_{setting}_ranks.png')
            save_path.parent.mkdir(parents=True, exist_ok=True)
            altair_saver.save(ax, str(save_path))
            raise ValueError
