""" Fine-tuning the library models for named entity recognition on CoNLL-2003. """
import torch
import einops
from mt_dnn.batcher import Collater
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import spearmanr
from itertools import product
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

def compute_correlation():
    root = Path('gradient_probe_outputs')
    # settings = list(LingualSetting)
    # settings.remove(LingualSetting.BASE)

    # all_tasks_settings = list(product(list(Experiment), settings))
    # all_tasks_settings_a = [f'{n[0].name}/{n[0].name}_{n[1].name.lower()}_en' for n in all_tasks_settings]
    # all_tasks_settings_a.extend([f'{n[0].name}/{n[0].name}_{n[1].name.lower()}_foreign' for n in all_tasks_settings])
    # all_tasks_settings_a.extend(['NLI/NLI_multi_foreign3', 'NLI/NLI_cross_foreign3'])
    # all_tasks_settings = all_tasks_settings_a
    # n = len(all_tasks_settings)

    # pairs = product(all_tasks_settings, all_tasks_settings)
    # rhos = pd.DataFrame(np.zeros((n, n)))
    # ps = pd.DataFrame(-1 * np.ones((n, n)))

    # rhos.index = all_tasks_settings
    # rhos.columns = all_tasks_settings
    # ps.index = all_tasks_settings
    # ps.columns = all_tasks_settings

    # for pair in pairs:
    #     print(pair)
    #     if pair[0] == pair[1]:
    #         rho, p = 1, 0
    #     else:
    #         if ps.loc[pair[1], pair[0]] != -1:
    #             p = ps.loc[pair[1], pair[0]]
    #             rho = rhos.loc[pair[1], pair[0]]
    #         else:
    #             a = root.joinpath(f'{pair[0]}_gp', 'grad.pt')
    #             b = root.joinpath(f'{pair[1]}_gp', 'grad.pt')
                
    #             a = raw_to_final_form(torch.load(a)).numpy()
    #             b = raw_to_final_form(torch.load(b)).numpy()

    #             a = np.expand_dims(a.flatten(), axis=1)
    #             b = np.expand_dims(b.flatten(), axis=1)

    #             rho, p = spearmanr(a, b)
        
    #     rhos.loc[pair[0], pair[1]] = rho
    #     ps.loc[pair[0], pair[1]] = p
    
    # rhos.to_csv(root.joinpath('rhos.csv'))
    # ps.to_csv(root.joinpath('ps.csv'))

    rhos = pd.read_csv(root.joinpath('rhos.csv'), index_col=0)
    ps = pd.read_csv(root.joinpath('ps.csv'), index_col=0)

    for data in [('rhos.pdf', rhos), ('ps.pdf', ps)]:
        plt.figure(figsize=(20, 20))
        annot_kws = {'fontsize': 20}
        ax = sns.heatmap(
            data[1],
            cbar=False,
            annot=True,
            annot_kws=annot_kws,
            fmt=".2f")

        ax.tick_params(axis='x', labelsize=20, labelrotation=90)
        ax.tick_params(axis='y', labelsize=20, labelrotation=0)

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