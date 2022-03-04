from pathlib import Path
import shutil
import seaborn as sns
import altair as alt
import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt
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

def preprocess_ranked_heads(setting):
    dataframes = []

    for task in ['MARC', 'POS', 'NER', 'NLI', 'PAWSX']:
        for ls in ['cross', 'multi']:
            data_for_model = []

            for i, k in enumerate(range(24, 144, 24)):
                path_to_data = Path(f'head_probe_ranking/{setting}/{task}/{task}_{ls}-{task}-{setting}_layer_ranks.np')
                accumulated = [0 for _ in range(12)]
                data = pd.read_csv(path_to_data, index_col=0)[:k]

                # get no. top heads per layer, normalize, and cumulative sum
                for _, d in data.iterrows():
                    accumulated[int(d)] += 1
                accumulated = [a / sum(accumulated) for a in accumulated]
                # accumulated = np.cumsum(accumulated)
                data_for_model.append(accumulated)

            df_for_model = pd.DataFrame(data_for_model)
            df_for_model.index = list(range(24, 144, 24))
            dataframes.append(df_for_model)
    
    return dataframes

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
    for pair in itertools.product(['SA', 'POS', 'NER', 'XNLI', 'PI'], ['cross-ling', 'multi-ling']):
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
        x=alt.X('model:N', title=''),

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

def plot_seaborn(dfs):
    for i, df in enumerate(dfs):
        df["Name"] = f"df{i}"
    dfall = pd.concat([pd.melt(i.reset_index(), id_vars=["Name", "index"]) for i in dfs], ignore_index=True)
    dfall.set_index(["Name", "index", "variable"], inplace=True)
    dfall["vcs"] = dfall.groupby(level=["Name", "index"]).cumsum()
    dfall.reset_index(inplace=True) 

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
    cs.extend(['k', 'w'])

    for i, g in enumerate(dfall.groupby("variable")):
        ax = sns.barplot(
            data=g[1],
            x="index",
            y="vcs",
            hue="Name",
            color=cs[i],
            zorder=-i, # so first bars stay on top
            edgecolor="k")
    ax.legend_.remove() # remove the redundant legends 
    return ax

def plot_clustered_stacked(dfall, labels=None, title="multiple stacked bar plot",  H="/", **kwargs):
    """
    Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
    labels is a list of the names of the dataframe, used for the legend
    title is a string for the title of the plot
    H is the hatch used for identification of the different dataframe

    For us, each dataframe needs to be:
    different k's on the index
    different layers on the columns
    each dataframe represents different model

    From: https://stackoverflow.com/questions/22787209/how-to-have-clusters-of-stacked-bars-with-python-pandas
    """

    n_df = len(dfall)
    n_col = len(dfall[0].columns) 
    n_ind = len(dfall[0].index)
    axe = plt.subplot(111)

    for df in dfall : # for each data frame
        axe = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False,
                      **kwargs)  # make bar plots

    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_width(1 / float(n_df + 1))

    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe.set_xticklabels(df.index, rotation = 0)
    axe.set_title(title)

    # Add invisible data to add another legend
    n=[]        
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="gray", hatch=H * i))

    l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.01, 0.5])
    if labels is not None:
        l2 = plt.legend(n, labels, loc=[1.01, 0.1]) 
    axe.add_artist(l1)
    return axe

if __name__ == '__main__':
    # rank_heads()
    for setting in ['en', 'foreign', 'combined']:
        save_path = f'head_probe_ranking/plots/plot_topk/{setting}_ranks.svg'
        dataframes = preprocess_ranked_heads(setting)
        ax = plot_altair(dataframes)
        altair_saver.save(ax, save_path)
