""" Fine-tuning the library models for named entity recognition on CoNLL-2003. """
import torch
import einops
from mt_dnn.batcher import Collater
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import spearmanr, pearsonr
from itertools import product, combinations
from experiments.exp_def import LingualSetting, Experiment

def min_max_norm(matrix):
    min_ = torch.min(matrix)
    max_ = torch.max(matrix)
    norm_matrix = (matrix - min_)/(max_ - min_)
    return norm_matrix

def max_norm(matrix):
    max_ = torch.max(matrix)
    norm_matrix = matrix / max_
    return norm_matrix

def raw_to_final_form(raw_attention_gradients):
    # layer norm
    for layer in range(12):
        raw_attention_gradients[layer, :] = max_norm(raw_attention_gradients[layer, :])
    
    # sum across training instances and global norm
    attention_gradients = torch.sum(raw_attention_gradients, dim=0)
    attention_gradients = min_max_norm(attention_gradients)

    return attention_gradients

def plot_heatmap(attention_gradients, output_path):  
    font_size = 45
    plt.figure(figsize=(20, 16))  
    annot_kws = {'fontsize': font_size}
    ax = sns.heatmap(
        attention_gradients,
        cbar=False,
        annot=False,
        annot_kws=annot_kws,
        fmt=".2f")

    ax.invert_yaxis()
    ax.set_xticklabels(list(range(1, 13)))
    ax.set_yticklabels(list(range(1, 13)))
    ax.set_xlabel('heads', fontsize=font_size)
    ax.set_ylabel('layers', fontsize=font_size)
    ax.tick_params(axis='x', labelsize=font_size)
    ax.tick_params(axis='y', labelsize=font_size)

    fig = ax.get_figure()
    fig.savefig(output_path, bbox_inches='tight')

def compute_correlation(method='pearson'):
    root = Path('gradient_probe_outputs')
    settings = list(LingualSetting)
    settings.remove(LingualSetting.BASE)
    conversion = {
        'NLI': "XNLI",
        'PAWSX': 'PI',
        'MARC': 'SA'
    }

    if not root.joinpath(f'rhos_{method}.csv').is_file():
        # only cross-cross, multi-multi pairs
        pairs = []
        all_models = []
        all_tasks = ['POS', 'NER', 'PI', 'SA', 'XNLI']
        for setting in ['cross', 'multi']:
            pool = []
            for task in ['POS', 'NER', 'PAWSX', 'MARC', 'NLI']:
                pool.append(f'{task}/{task}_{setting}')
            pool = list(combinations(pool, r=2))
            pairs.extend(pool)
        
        for task in all_tasks:
            for setting in ['cross', 'multi']:
                all_models.append(f'{task}-{setting}-ling') 
        
        shape_ = (len(all_tasks), len(all_models))
        rhos = pd.DataFrame(np.zeros(shape_))
        ps = pd.DataFrame(-1 * np.ones(shape_))

        rhos.index = all_tasks
        rhos.columns = all_models
        ps.index = all_tasks
        ps.columns = all_models

        for pair in pairs:
            task_a, setting = pair[0].split("/")[1].split("_")
            task_b = pair[1].split("/")[1].split("_")[0]

            if task_a in conversion:
                task_a = conversion[task_a]
            if task_b in conversion:
                task_b = conversion[task_b]
            
            model = f'{task_a}-{setting}-ling'

            if ps.loc[task_a, f'{task_b}-{setting}-ling'] > -1:
                rho = rhos.loc[task_a, f'{task_b}-{setting}-ling']
                p = ps.loc[task_a, f'{task_b}-{setting}-ling']
            else:
                a = root.joinpath(f'{pair[0]}_gp', 'grad.pt')
                b = root.joinpath(f'{pair[1]}_gp', 'grad.pt')
                
                a = raw_to_final_form(torch.load(a)).numpy().flatten()
                b = raw_to_final_form(torch.load(b)).numpy().flatten()

                if method == 'spearman':
                    a = np.expand_dims(a, axis=1)
                    b = np.expand_dims(b, axis=1)
                    rho, p = spearmanr(a, b)
                elif method == 'pearson':
                    rho, p = pearsonr(a, b)
            
            rhos.loc[task_b, model] = rho
            ps.loc[task_b, model] = p

            # mirror image
            rhos.loc[task_a, f'{task_b}-{setting}-ling'] = rho
            ps.loc[task_a, f'{task_b}-{setting}-ling'] = p

            print(pair, (task_b, model), rho)
        
        for task in all_tasks:
            for setting in ['cross', 'multi']:
                rhos.loc[task, f'{task}-{setting}-ling'] = 1
                ps.loc[task, f'{task}-{setting}-ling'] = 0
        
        rhos.to_csv(root.joinpath(f'rhos_{method}.csv'))
        ps.to_csv(root.joinpath(f'ps_{method}.csv'))

    rhos = pd.read_csv(root.joinpath(f'rhos_{method}.csv'), index_col=0)
    ps = pd.read_csv(root.joinpath(f'ps_{method}.csv'), index_col=0)

    font_size = 40
    for data in [(f'rhos_{method}.pdf', rhos), (f'ps_{method}.pdf', ps)]:
        plt.figure(figsize=(20, 20))
        annot_kws = {'fontsize': font_size}
        ax = sns.heatmap(
            data[1],
            cbar=False,
            annot=True,
            annot_kws=annot_kws,
            fmt=".2f",
            square=True)

        ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor', fontsize=font_size)
        ax.tick_params(axis='y', labelsize=font_size, labelrotation=0)

        fig = ax.get_figure()
        fig.savefig(root.joinpath(data[0]), bbox_inches='tight')

def prediction_gradient(args, model, dataloader, save_path):
    if not save_path.joinpath('grad.pt').is_file():
        save_path.mkdir(parents=True, exist_ok=True)

        n_batches = len(dataloader)
        attention_gradients = torch.zeros((len(dataloader) * args.batch_size, 12, 12))

        for i, (batch_meta, batch_data) in enumerate(dataloader):
            batch_meta, batch_data = Collater.patch_data(
                torch.device(args.devices[0]),
                batch_meta,
                batch_data)
            model.get_update_gradients(batch_meta, batch_data)

            for layer in range(12):
                attention_layer = model.get_head_probe_layer(layer)
                k = attention_layer.key.weight.grad.detach()
                v = attention_layer.value.weight.grad.detach()
                k = einops.rearrange(k, 'h (n d) -> n d h', n=12, d=64)
                v = einops.rearrange(v, 'h (n d) -> n d h', n=12, d=64)
                grad_ = torch.abs(k + v)
                grad_ = torch.sum(grad_, dim=(-1, -2)) # sum across 64 * 768 (attention_head_size, hidden_size)
                attention_gradients[i, layer] = grad_

            model.network.zero_grad()
            if (i + 1) % 500 == 0:
                print(f'{i+1}/{n_batches}')

        # save raw
        attention_gradients = attention_gradients[:i+1, ...] # in case # training examples isn't exactly len(dataloader) * batch_size
        torch.save(attention_gradients, save_path.joinpath('grad.pt'))
    
    else:
        attention_gradients = torch.load(save_path.joinpath('grad.pt'))

    attention_gradients = raw_to_final_form(attention_gradients)
    plot_heatmap(attention_gradients, save_path.joinpath(f'{save_path.name}.pdf'))

if __name__ == '__main__':
    prediction_gradient()
    for method in ['pearson', 'spearman']:
        compute_correlation(method)
