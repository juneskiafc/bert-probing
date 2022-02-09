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

def create_ranked_heads_heatmap(diffs_file, output_dir, task, normalize=True, diffs=None):
    heatmap_out_file = output_dir.joinpath(f'results/{task}_ranked_heads_by_absdiff.pdf')
    
    if diffs is None:
        heatmap = np.load(diffs_file)
    else:
        heatmap = diffs
    
    if normalize:
        max_ = np.amax(heatmap)
        heatmap /= max_
    
    plt.figure(figsize=(14, 14))
    annot_kws = {'fontsize': 20}
    ax = sns.heatmap(
        heatmap,
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

def compare_kqv_across_multiple(task_name_a, task_name_b):
    cross_diffs = np.load(f'kqv_outputs/results/{task_name_a}_diffs.npy')
    multi_diffs = np.load(f'kqv_outputs/results/{task_name_b}_diffs.npy')

    heatmap = cross_diffs - multi_diffs
    heatmap = (2 * ((heatmap - np.amin(heatmap)) / (np.amax(heatmap) - np.amin(heatmap)))) - 1

    plt.figure(figsize=(14, 14))
    annot_kws = {'fontsize': 20}
    ax = sns.heatmap(
        heatmap,
        center=0, 
        cmap='bwr',
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
    fig.savefig(f'kqv_outputs/diff_results/{task_name_a}-{task_name_b}_comp_ranked_heads_by_absdiff.pdf', bbox_inches='tight')

def get_spearmans_rho(task):
    cross_diffs = np.load(f'kqv_outputs/results/{task}_cross_diffs.npy')
    multi_diffs = np.load(f'kqv_outputs/results/{task}_multi_diffs.npy')

    cross_diffs = np.expand_dims(cross_diffs.flatten(), axis=1)
    multi_diffs = np.expand_dims(multi_diffs.flatten(), axis=1)

    rho, p = spearmanr(cross_diffs, multi_diffs)

    return rho, p

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

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_ckpt', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='kqv_outputs')
    parser.add_argument('--model_name', type=str, default='')
    args = parser.parse_args()

    # for task in ['NER', 'POS']:
    #     for setting in ['cross', 'multi']:
    #         for seed in ['1', '2']:
    #             args.model_ckpt = list(Path(f'checkpoint/{task}_{setting}_seed{seed}').rglob("*.pt"))[0]
    #             save_all_kqv(args.model_ckpt, args.output_dir)

    for task in ['NER']:
        for setting in ['cross', 'multi']:
            for seed in range(3):
                model_name = f'{task}_{setting}'
                if seed > 0:
                    model_name += f'_seed{seed}'
                compare_kqv(model_name, args.output_dir)

        # compare_kqv_across_multiple(f'{task}_cross', f'{task}_multi')

    # rho_file = 'kqv_outputs/downstream_rho.csv'
    # data = []
    # for task in ['POS', 'NER', 'PAWSX', 'MARC']:
    #     rho, p = get_spearmans_rho(task)
    #     data.append([rho, p])
    # df = pd.DataFrame(data)
    # df.columns = ['rho', 'p']
    # df.index = ['POS', 'NER', 'PAWSX', 'MARC']
    # df.to_csv(rho_file)
    # plot_spearmans_rho(languages)