import torch
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
from argparse import ArgumentParser
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from pretrained_models import MODEL_CLASSES

def save_all_kqv(model_ckpt, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_ckpt == '':
        _, model_class, _ = MODEL_CLASSES['bert']
        bert = model_class.from_pretrained('bert-base-multilingual-cased', cache_dir='.cache')
        state_dict = {f'bert.{k}':v for k, v in bert.state_dict().items()}
        model_name = 'mBERT'
    else:   
        model_name = Path(model_ckpt).parent.name
        state_dict = torch.load(model_ckpt)['state']

    for layer_idx in range(12):
        K = state_dict[f'bert.encoder.layer.{layer_idx}.attention.self.key.weight']
        Q = state_dict[f'bert.encoder.layer.{layer_idx}.attention.self.query.weight']
        V = state_dict[f'bert.encoder.layer.{layer_idx}.attention.self.value.weight']

        n_heads = 12
        size_per_head = K.shape[-1] // n_heads

        for head_idx in range(n_heads):
            output_dst_for_head = output_dir.joinpath(model_name, str(layer_idx), str(head_idx))
            output_dst_for_head.mkdir(exist_ok=True, parents=True)

            ii = head_idx*size_per_head
            fi = (head_idx+1) * size_per_head

            # K
            output_dst = output_dst_for_head.joinpath('key.pt')
            torch.save(K[..., ii:fi], output_dst)

            # Q
            output_dst = output_dst_for_head.joinpath('query.pt')
            torch.save(Q[..., ii:fi], output_dst)

            # V
            output_dst = output_dst_for_head.joinpath('value.pt')
            torch.save(V[..., ii:fi], output_dst)

def compare_kqv(model_name, output_dir):
    output_dir = Path(output_dir)
    finetuned_kqv_dir = output_dir.joinpath(f'{model_name}')
    pretrained_kqv_dir = output_dir.joinpath('mBERT')
    output_diffs_file = output_dir.joinpath(f'results/{model_name}_diffs.npy')
    output_diffs_file.parent.mkdir(parents=True, exist_ok=True)

    if not output_diffs_file.is_file():
        diffs = np.zeros((12, 12))

        for hidden_layer_folder in pretrained_kqv_dir.iterdir():
            hidden_layer_idx = hidden_layer_folder.name
            for head_folder in hidden_layer_folder.iterdir():
                head_idx = head_folder.name

                pretrained_k = torch.load(head_folder.joinpath('key.pt'))
                pretrained_q = torch.load(head_folder.joinpath('query.pt'))
                pretrained_v = torch.load(head_folder.joinpath('value.pt'))

                ft_hl_head_folder = finetuned_kqv_dir.joinpath(hidden_layer_idx, head_idx)
                finetuned_k = torch.load(ft_hl_head_folder.joinpath('key.pt'))
                finetuned_q = torch.load(ft_hl_head_folder.joinpath('query.pt'))
                finetuned_v = torch.load(ft_hl_head_folder.joinpath('value.pt'))

                finetuned_k = finetuned_k.to(pretrained_k.device)
                finetuned_q = finetuned_q.to(pretrained_q.device)
                finetuned_v = finetuned_v.to(pretrained_v.device)

                # sum of abs diff
                k_diff = torch.sum(torch.abs(pretrained_k - finetuned_k))
                q_diff = torch.sum(torch.abs(pretrained_q - finetuned_q))
                v_diff = torch.sum(torch.abs(pretrained_v - finetuned_v))
                diff_for_head = (k_diff + q_diff + v_diff).cpu().detach().numpy()
                diffs[int(hidden_layer_idx), int(head_idx)] = diff_for_head

        np.save(output_diffs_file, diffs)
    
    create_ranked_heads_heatmap(output_diffs_file, output_dir.joinpath(model_name).with_suffix('.pdf'), model_name)

def create_ranked_heads_heatmap(diffs_file, out_file, normalize=True, diffs=None):    
    if diffs is None:
        heatmap = np.load(diffs_file)
    else:
        heatmap = diffs
    
    if normalize:
        max_ = np.amax(heatmap)
        heatmap /= max_
    
    font_size = 25
    plt.figure(figsize=(14, 14))
    annot_kws = {'fontsize': font_size}
    ax = sns.heatmap(
        heatmap,
        cbar=False,
        annot=False,
        annot_kws=annot_kws,
        xticklabels=list(range(1, 13)),
        yticklabels=list(range(1, 13)),
        fmt=".2f")

    ax.invert_yaxis()
    ax.set_xlabel('heads', fontsize=font_size)
    ax.set_ylabel('layers', fontsize=font_size)
    ax.tick_params(axis='x', labelsize=font_size)
    ax.tick_params(axis='y', labelsize=font_size)

    fig = ax.get_figure()
    fig.savefig(out_file, bbox_inches='tight')

def get_kqv_update_diff(model_name, output_dir):
    model_name = model_name.split("_")[0]
    diffs = [
        Path('kqv_outputs').joinpath('results', f'{model_name}_cross_diffs.npy'),
        Path('kqv_outputs').joinpath('results', f'{model_name}_multi_diffs.npy')
    ]
    out_name = Path(diffs[0]).with_suffix('').name + "-" + Path(diffs[1]).with_suffix('').name
    diffs_a, diffs_b = [np.load(d) for d in diffs]

    heatmap = diffs_a - diffs_b
    heatmap = (2 * ((heatmap - np.amin(heatmap)) / (np.amax(heatmap) - np.amin(heatmap)))) - 1

    font_size = 45
    plt.figure(figsize=(20, 16))
    annot_kws = {'fontsize': font_size}
    ax = sns.heatmap(
        heatmap,
        center=0, 
        cmap='bwr',
        cbar=False,
        annot=False,
        annot_kws=annot_kws,
        xticklabels=list(range(1, 13)),
        yticklabels=list(range(1, 13)),
        fmt=".2f")
    
    ax.invert_yaxis()
    ax.set_xticklabels(list(range(1, 13)))
    ax.set_yticklabels(list(range(1, 13)))
    ax.set_xlabel('heads', fontsize=font_size)
    ax.set_ylabel('layers', fontsize=font_size)
    ax.tick_params(axis='x', labelsize=font_size)
    ax.tick_params(axis='y', labelsize=font_size)

    fig = ax.get_figure()
    fig.savefig(Path(output_dir).joinpath(f'{out_name}.pdf'), bbox_inches='tight')

def get_spearmans_rho(diffs):
    diffs = [np.load(d) for d in diffs]
    diffs_a, diffs_b = [np.expand_dims(d.flatten(), axis=1) for d in diffs]
    rho, p = spearmanr(diffs_a, diffs_b)
    return rho, p

def get_spearmans_rho_between_seeds(task):
    # cross_diffs = get_mean_across_seeds(task, 'cross', True)
    # multi_diffs = get_mean_across_seeds(task, 'multi', True)

    # cross_diffs = np.load(f'kqv_outputs/results/{task}_cross_diffs.npy')
    # multi_diffs = np.load(f'kqv_outputs/results/{task}_multi_diffs.npy')

    def load_diff_for_seed(seed, task, setting):
        if seed == 0:
            diffs = np.load(f'kqv_outputs/results/{task}_{setting}_diffs.npy')
        else:
            diffs = np.load(f'kqv_outputs/results/{task}_{setting}_seed{seed}_diffs.npy')
        return diffs
    
    cross_data = []
    multi_data = []
    for seed_pair in [(0, 1), (1, 2), (0, 2)]:
        for setting in ['cross', 'multi']:
            diffs_1 = load_diff_for_seed(seed_pair[0], task, setting)
            diffs_2 = load_diff_for_seed(seed_pair[1], task, setting)

            diffs_1 = np.expand_dims(diffs_1.flatten(), axis=1)
            diffs_2 = np.expand_dims(diffs_2.flatten(), axis=1)

            rho, p = spearmanr(diffs_1, diffs_2)
            
            if setting == 'cross':
                cross_data.append([rho, p, seed_pair])
            else:
                multi_data.append([rho, p, seed_pair])

    return cross_data, multi_data

def plot_spearmans_rho(languages):
    def flatten_diffs(diffs):
        return np.expand_dims(diffs.flatten(), axis=1)
    
    heatmap = np.zeros((15, 15))
    diffs = []
    columns = []

    cross_diffs = np.load('/home/june/mt-dnn/kqv/cross_diffs.npy')
    cross_diffs = flatten_diffs(cross_diffs)
    diffs.append(cross_diffs)
    columns.append('cl')

    for language in languages:
        language_diff = np.load(f'/home/june/mt-dnn/kqv/multi-{language}_diffs.npy')
        language_diff = flatten_diffs(language_diff)
        diffs.append(language_diff)
        columns.append(language)
    
    for i, diff_a in enumerate(diffs):
        for j, diff_b in enumerate(diffs):
            rho, p = spearmanr(diff_a, diff_b)
            heatmap[i, j] = rho
    
    ax = sns.heatmap(heatmap, xticklabels=columns, yticklabels=columns)
    ax.set_title('spearman rho, head change after fine-tuning')
    fig = ax.get_figure()
    fig.savefig(f'/home/june/mt-dnn/kqv/alllang_ranked_heads_by_absdiff.png')

def get_mean_across_seeds(task, setting, return_df_only=True):
    root = Path('kqv_outputs/results')
    heatmap_out_file = root.joinpath(f'{task}_{setting}_mean.pdf')
    seed0 = np.load(root.joinpath(f'{task}_{setting}_diffs.npy'))
    seed1 = np.load(root.joinpath(f'{task}_{setting}_seed1_diffs.npy'))
    seed2 = np.load(root.joinpath(f'{task}_{setting}_seed2_diffs.npy'))

    print(task, setting)
    seeds = [seed0, seed1, seed2]
    for seed in seeds:
        max_ = np.amax(seed)
        seed /= max_

    data = np.stack([seed0, seed1, seed2], axis=0)
    mean = np.mean(data, axis=0)
    
    if return_df_only:
        return mean

    plt.figure(figsize=(14, 14))
    annot_kws = {'fontsize': 20}
    ax = sns.heatmap(
        mean,
        cbar=False,
        annot=True,
        annot_kws=annot_kws,
        fmt=".2f")

    ax.invert_yaxis()
    ax.set_xlabel('heads', fontsize=20)
    ax.set_ylabel('layers', fontsize=20)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

    fig = ax.get_figure()
    fig.savefig(heatmap_out_file, bbox_inches='tight')

def main_sequence(model_ckpt, model_name, output_dir):
    if model_name == '':
        model_name = Path(model_ckpt).parent.name
    if model_ckpt == '':
        model_ckpt = list(Path('checkpoint').joinpath(model_name).rglob("model_5*.pt"))[0]

    save_all_kqv(model_ckpt, output_dir)
    compare_kqv(model_name, output_dir)
    get_kqv_update_diff(model_name, output_dir)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_ckpt', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='kqv_outputs')
    parser.add_argument('--model_name', type=str, default='')

    parser.add_argument('--diffs_to_compare', nargs='+')
    args = parser.parse_args()

    main_sequence(args.model_ckpt, args.model_name, args.output_dir)