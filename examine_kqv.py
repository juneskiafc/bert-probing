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
    
    create_ranked_heads_heatmap(output_diffs_file, output_dir, model_name)

def create_ranked_heads_heatmap(diffs_file, out_file, normalize=True, diffs=None):    
    if diffs is None:
        heatmap = np.load(diffs_file)
    else:
        heatmap = diffs
    
    if normalize:
        max_ = np.amax(heatmap)
        heatmap /= max_
    
    font_size = 45
    plt.figure(figsize=(20, 16))
    annot_kws = {'fontsize': font_size}
    ax = sns.heatmap(
        heatmap,
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
    fig.savefig(out_file, bbox_inches='tight')

def compare_kqv_across_multiple(task_name_a, task_name_b):
    cross_diffs = np.load(f'kqv_outputs/results/{task_name_a}_diffs.npy')
    multi_diffs = np.load(f'kqv_outputs/results/{task_name_b}_diffs.npy')

    heatmap = cross_diffs - multi_diffs
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
        fmt=".2f")
    
    ax.invert_yaxis()
    ax.set_xticklabels(list(range(1, 13)))
    ax.set_yticklabels(list(range(1, 13)))
    ax.set_xlabel('heads', fontsize=font_size)
    ax.set_ylabel('layers', fontsize=font_size)
    ax.tick_params(axis='x', labelsize=font_size)
    ax.tick_params(axis='y', labelsize=font_size)

    fig = ax.get_figure()
    fig.savefig(f'kqv_outputs/{task_name_a}-{task_name_b}_comp_ranked_heads_by_absdiff.pdf', bbox_inches='tight')

def get_spearmans_rho(task):
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

def all_seeds_rho_p(args):
    crosses_data = []
    multis_data = []
    crosses_p = []
    multis_p = []

    tasks = ['NLI', 'POS', 'NER', 'PAWSX', 'MARC']
    for task in tasks:
        # rho, p = get_spearmans_rho(task)
        cross_data, multi_data = get_spearmans_rho(task)
        crosses_data.append([cross_data[i][0] for i in range(3)])
        multis_data.append([multi_data[i][0] for i in range(3)])
        crosses_p.append([cross_data[i][1] for i in range(3)])
        multis_p.append([multi_data[i][1] for i in range(3)])
        seeds = [multi_data[i][2] for i in range(3)]
    
    for i, (rhos, ps) in enumerate(zip([crosses_data, multis_data], [crosses_p, multis_p])):
        if i == 0:
            setting = 'cross'
        else:
            setting = 'multi'
        
        rho_file = Path(f'kqv_outputs/{setting}_3seeds_separate_rho.csv')
        df = pd.DataFrame(rhos)
        df.columns = seeds
        df.index = tasks
        df.to_csv(rho_file)

        ps = pd.DataFrame(ps)
        ps.columns = seeds
        ps.index = tasks
        ps.to_csv(f'kqv_outputs/{setting}_3seeds_separate_p.csv')

        plt.figure(figsize=(14, 14))
        annot_kws = {'fontsize': 20}
        ax = sns.heatmap(
            df,
            cbar=False,
            annot=True,
            annot_kws=annot_kws,
            fmt=".2f")

        # ax.set_xlabel('rho', fontsize=20)
        ax.set_ylabel('tasks', fontsize=20)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)

        fig = ax.get_figure()
        fig.savefig(f'kqv_outputs/{setting}_3seeds_separate_rho.pdf', bbox_inches='tight')

def create_final_ranked_heads_figure(args):
    for task in ['MARC', 'NER', 'NLI', 'PAWSX', 'POS']:
        for setting in ['cross', 'multi']:
            diffs_file = f'kqv_outputs/results/{task}_{setting}_diffs.npy'
            output_file = Path(args.output_dir).joinpath(f'{task}_{setting}.pdf')
            create_ranked_heads_heatmap(diffs_file, output_file, task)
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_ckpt', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='kqv_outputs')
    parser.add_argument('--model_name', type=str, default='')
    args = parser.parse_args()
    
    for task in ['MARC', 'NER', 'NLI', 'PAWSX', 'POS']:
        compare_kqv_across_multiple(f'{task}_cross', f'{task}_multi')
