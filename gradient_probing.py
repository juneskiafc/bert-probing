""" Fine-tuning the library models for named entity recognition on CoNLL-2003. """
from argparse import ArgumentParser
import torch
import einops
from data_utils.task_def import EncoderModelType
from mt_dnn.batcher import Collater
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import spearmanr, pearsonr
from itertools import combinations
from experiments.exp_def import Experiment, LingualSetting, TaskDefs

from mt_dnn.model import MTDNNModel
from utils import create_heatmap, base_construct_model, build_dataset

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

def prediction_gradient(
    finetuned_task,
    finetuned_setting,
    downstream_task,
    downstream_setting,
    device_id,
    save_dir,
    batch_size=8
):  
    task_def_path_for_data = Path('experiments').joinpath(downstream_task, 'task_def.yaml')
    task_def_for_data = TaskDefs(task_def_path_for_data).get_task_def(downstream_task.lower())
    task_def_path_for_model = Path('experiments').joinpath(finetuned_task, 'task_def.yaml')

    data_path = Path('experiments').joinpath(
        downstream_task,
        downstream_setting,
        'bert-base-multilingual-cased',
        f'{downstream_task.lower()}_train.json'
    )
    print(f'building data from {data_path}')
    dataloader = build_dataset(
        data_path,
        EncoderModelType.BERT,
        batch_size=8,
        max_seq_len=512,
        task_def=task_def_for_data,
        is_train=True)
    
    model_ckpt = list(Path('checkpoint').joinpath(f'{finetuned_task}_{finetuned_setting}').rglob("model_5*.pt"))[0]
    print(f'loading model from {model_ckpt}')
    config, state_dict, _ = base_construct_model(
        model_ckpt,
        Experiment[finetuned_task.upper()],
        task_def_path_for_model,
        device_id
    )
    config['gradient_probe'] = True
    config['gradient_probe_n_classes'] = task_def_for_data.n_class
    model = MTDNNModel(config, devices=[f'cuda:{device_id}'])

    state_dict['state']['scoring_list.0.weight'] = model.network.state_dict()['scoring_list.0.weight']
    state_dict['state']['scoring_list.0.bias'] = model.network.state_dict()['scoring_list.0.bias']
    model.load_state_dict(state_dict)

    save_path = save_dir.joinpath(
        finetuned_task,
        downstream_task,
        finetuned_setting
    )

    if not save_path.joinpath('grad.pt').is_file():
        save_path.mkdir(parents=True, exist_ok=True)

        n_batches = len(dataloader)
        attention_gradients = torch.zeros((len(dataloader) * batch_size, 12, 12))

        for i, (batch_meta, batch_data) in enumerate(dataloader):
            batch_meta, batch_data = Collater.patch_data(
                torch.device(f'cuda:{device_id}'),
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
    attention_gradients = attention_gradients.detach().cpu().numpy()
    attention_gradients = pd.DataFrame(attention_gradients)
    attention_gradients.to_csv(save_path.joinpath('grad.csv'))

    create_heatmap(
        data_df=attention_gradients,
        row_labels=list(range(1, 13)),
        column_labels=list(range(1, 13)),
        xaxlabel='heads',
        yaxlabel='layers',
        figsize=(20, 16),
        fontsize=45,
        invert_y=True,
        out_file=save_path.joinpath(f'grad.pdf')
    )

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--finetuned_task", default='')
    parser.add_argument('--finetuned_setting', default='')
    parser.add_argument('--downstream_task', default='')
    parser.add_argument('--downstream_setting', default='cross')
    parser.add_argument('--save_dir', default='gradient_probing_outputs')
    parser.add_argument('--device_id', default=0)
    args = parser.parse_args()
    
    prediction_gradient(
        args.finetuned_task,
        args.finetuned_setting,
        args.downstream_task,
        args.downstream_setting,
        args.device_id,
        Path(args.save_dir)
    )

    # for method in ['pearson', 'spearman']:
    #     compute_correlation(method)
