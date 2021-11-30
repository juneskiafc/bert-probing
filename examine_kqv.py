import torch
from pathlib import Path
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr

def save_all_kqv(model, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # index in to bert
    bert = model.network.bert
    _, encoder_module = list(bert.named_children())[1]
    _, all_hidden_layers = list(encoder_module.named_children())[0] # contains all encoder attention layers

    for hidden_layer_idx, hidden_layer in enumerate(all_hidden_layers):
        for (bert_sublayer_name, bert_sublayer) in hidden_layer.named_children():
            if bert_sublayer_name == 'attention':
                _, self_attention_layer = list(bert_sublayer.named_children())[0]
                assert self_attention_layer.__class__.__name__ == 'BertSelfAttention', self_attention_layer.__class__.__name__
                output_dir_for_layer = output_dir.joinpath(str(hidden_layer_idx))
                output_dir_for_layer.mkdir(parents=True, exist_ok=True)

                _, K = list(self_attention_layer.key.named_parameters())[0]
                _, Q = list(self_attention_layer.query.named_parameters())[0]
                _, V = list(self_attention_layer.value.named_parameters())[0]

                n_heads = 12
                size_per_head = K.shape[-1] // n_heads

                for head_idx in range(n_heads):
                    output_dst_for_head = output_dir_for_layer.joinpath(str(head_idx))
                    output_dst_for_head.mkdir(exist_ok=True)

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

def compare_kqv(task, output_dir):
    output_dir = Path(output_dir)
    finetuned_kqv_dir = Path(f'/home/june/mt-dnn/kqv/finetuned/{task}')
    pretrained_kqv_dir = Path(f'/home/june/mt-dnn/kqv/pretrained')
    output_diffs_file = output_dir.joinpath(f'{task}_diffs.npy')

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

        output_diffs_file = output_dir.joinpath(f'{task}_diffs.npy')
        np.save(output_diffs_file, diffs)
    
    create_ranked_heads_heatmap(output_diffs_file, output_dir, task)

def create_ranked_heads_heatmap(diffs_file, output_dir, task, normalize=True, diffs=None):
    heatmap_out_file = output_dir.joinpath(f'{task}_ranked_heads_by_absdiff.png')
    if heatmap_out_file.is_file():
        return
    
    if diffs is None:
        heatmap = np.load(diffs_file)
    else:
        heatmap = diffs
    
    if normalize:
        max_ = np.amax(heatmap)
        heatmap /= max_
    
    ax = sns.heatmap(heatmap)
    ax.invert_yaxis()
    ax.set_xlabel('heads')
    ax.set_ylabel('layers')
    ax.set_title(f'{task}-lingual \n absolute summed difference of KQV \n before/after fine-tuning')
    fig = ax.get_figure()
    fig.savefig(output_dir.joinpath(f'{task}_ranked_heads_by_absdiff.png'))

def compare_kqv_across_multiple(task_name_a, task_name_b):
    cross_diffs = np.load(f'/home/june/mt-dnn/kqv/{task_name_a}_diffs.npy')
    multi_diffs = np.load(f'/home/june/mt-dnn/kqv/{task_name_b}_diffs.npy')

    heatmap = cross_diffs - multi_diffs
    heatmap = (2 * ((heatmap - np.amin(heatmap)) / (np.amax(heatmap) - np.amin(heatmap)))) - 1

    ax = sns.heatmap(heatmap, center=0, cmap='bwr')
    ax.invert_yaxis()
    ax.set_xlabel('heads')
    ax.set_ylabel('layers')
    ax.set_title(f'{task_name_a}-{task_name_b}-comp \n absolute summed difference of KQV after fine-tuning')
    fig = ax.get_figure()
    fig.savefig(f'/home/june/mt-dnn/kqv/{task_name_a}-{task_name_b}_comp_ranked_heads_by_absdiff.png')

def get_spearmans_rho():
    cross_diffs = np.load('/home/june/mt-dnn/kqv/cross_diffs.npy')
    multi_diffs = np.load('/home/june/mt-dnn/kqv/multi_diffs.npy')

    cross_diffs = np.expand_dims(cross_diffs.flatten(), axis=1)
    multi_diffs = np.expand_dims(multi_diffs.flatten(), axis=1)

    rho, p = spearmanr(cross_diffs, multi_diffs)

    return rho

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
    languages = ['ar', 'bg', 'de', 'el', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']
    # for l in languages:
    #     task = f'multi-{l}'
    #     compare_kqv(task=task, output_dir='/home/june/mt-dnn/kqv')
    # compare_kqv_across_multiple('multi-ar', 'multi-bg')
    # get_spearmans_rho()
    plot_spearmans_rho(languages)