from pathlib import Path
import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from experiments.exp_def import TaskDefs, TaskDef, LingualSetting, Experiment
from data_utils.task_def import EncoderModelType
from mt_dnn.model import MTDNNModel
from mt_dnn.batcher import SingleTaskDataset, Collater
from torch.utils.data import DataLoader
import tasks
import numpy as np
import einops
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from time import time

def compute_joint_attention(att_mat, add_residual=True):
    if add_residual:
        residual_att = np.eye(att_mat.shape[1])[None,...]
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(axis=-1)[...,None]
    else:
       aug_att_mat =  att_mat
    
    joint_attentions = np.zeros(aug_att_mat.shape)

    layers = joint_attentions.shape[0]
    joint_attentions[0] = aug_att_mat[0]
    for i in np.arange(1,layers):
        joint_attentions[i] = aug_att_mat[i].dot(joint_attentions[i-1])
        
    return joint_attentions

def save_acc_matrix_as_heatmap(file_df=None, out_file=''):
    fontsize = 20
    row_labels = file_df.index
    column_labels = file_df.columns
    data = file_df.to_numpy()

    plt.figure(figsize=(14, 14))
    annot_kws = {
        "fontsize":fontsize,
    }
    heatmap = sns.heatmap(
        data,
        cbar=False,
        annot=True,
        annot_kws=annot_kws,
        fmt=".2f",
        cmap='RdYlGn')

    heatmap.set_yticklabels(row_labels, rotation=0, fontsize=fontsize)
    heatmap.set_xticklabels(column_labels, rotation=0, fontsize=fontsize)
    heatmap.set_xlabel('method', fontsize=fontsize)
    heatmap.set_ylabel('languages', fontsize=fontsize)

    fig = heatmap.get_figure()
    fig.savefig(Path(out_file).with_suffix('.pdf'), bbox_inches='tight')

def create_model(finetuned_task: Experiment, setting: LingualSetting, device_id: int):
    """
    Create the MT-DNN model, finetuend on finetuned_task in the {cross, multi}-lingual setting.
    """
    checkpoint_dir = Path('checkpoint').joinpath(f'{finetuned_task.name}_{setting.name.lower()}')
    checkpoint_file = list(checkpoint_dir.rglob('*.pt'))[0]
    state_dict = torch.load(checkpoint_file)
    del state_dict['optimizer']

    task_def_path = Path(f'experiments').joinpath(
        finetuned_task.name,
        setting.name.lower(),
        'task_def.yaml')
    task_defs = TaskDefs(task_def_path)

    config = state_dict['config']
    task_def = task_defs.get_task_def(setting.name.lower())
    task_def_list = [task_def]
    config['task_def_list'] = task_def_list

    return MTDNNModel(config, state_dict=state_dict, devices=[device_id])

def load_data(data_file, task_def_path, language):
    """
    Create the test dataloader to use.

    Args:
    data_file: The JSON test data file
    task_def: Path to the task_def.
    language: Specific language to use.
    """
    test_data_set = SingleTaskDataset(
        data_file,
        is_train=False,
        task_def=TaskDefs(task_def_path).get_task_def(language),
    )

    collater = Collater(is_train=False, encoder_type=EncoderModelType.BERT)
    test_data = DataLoader(
        test_data_set,
        batch_size=8,
        collate_fn=collater.collate_fn,
        pin_memory=True)

    return test_data

def get_attention(model, batch_meta, batch_data):
    batch_meta, batch_data = Collater.patch_data(device_id, batch_meta, batch_data)

    # fwd pass and get CLS attention
    # prepare input
    task_id = batch_meta['task_id']
    task_def = TaskDef.from_dict(batch_meta['task_def'])
    inputs = batch_data[:batch_meta['input_len']]
    if len(inputs) == 3:
        inputs.append(None)
        inputs.append(None)
    inputs.append(task_id)

    input_ids = inputs[0]
    token_type_ids = inputs[1]
    attention_mask = inputs[2]

    outputs = model.network.bert(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        inputs_embeds=None,
        output_hidden_states=True,
        output_attentions=True
    )
    attentions = outputs.attentions # length 12 tuple (one for each layer), b, num_heads, seq_len, seq_len
    return attentions, input_ids

if __name__ == '__main__':
    device_id = 0
    languages = [
        # 'ar',
        # 'bg',
        # 'de',
        # 'el',
        'en',
        # 'es',
        # 'fr',
        # 'hi',
        # 'ru',
        # 'sw',
        # 'th',
        # 'tr',
        # 'ur',
        # 'vi',
        # 'zh'
    ]

    cross_model = create_model('cross', device_id)
    multi_model = create_model('multi', device_id)

    cross_model.network.eval()
    multi_model.network.eval()
    models = {'cross': cross_model, 'multi': multi_model}

    results_mean = []
    results_std = []

    for language in languages:
        data_file = Path(f'experiments/NLI/').joinpath(language, 'bert-base-multilingual-cased', f'{language}_test.json')
        task_def = data_file.parent.parent.joinpath('task_def.yaml')

        corrs_out_file = f'attention_flow/{language}/'
        Path(corrs_out_file).mkdir(parents=True, exist_ok=True)

        if len(list(Path(corrs_out_file).rglob("*.npy"))) == 3:
            corrs_saved = True
            rollout_corrs = np.load(corrs_out_file + "rollout_rho.npy")
            last3_corrs = np.load(corrs_out_file + "last3_rho.npy")
            last_corrs = np.load(corrs_out_file + "last_rho.npy")
        
        else:
            corrs_saved = False
            test_data = load_data(data_file, task_def, language)
            with torch.no_grad():
                rollout_corrs = []
                last3_corrs = []
                last_corrs = []

                ti_rs = []
                ti_l3s = []
                ti_ls = []

                saved_rollouts = []
                saved_last3s = []
                saved_lasts = []

                n_total_examples = len(test_data)
                logging_freq = 50

                for batch_idx, (batch_meta, batch_data) in enumerate(test_data):
                    if (batch_idx + 1) % logging_freq == 0:
                        print(f'[{language}] {batch_idx+1}/{n_total_examples}')

                    rollouts = []
                    last3s = []
                    lasts = []
                    iids = []

                    for setting, model in models.items():
                        attentions, input_ids = get_attention(model, batch_meta, batch_data)

                        # stack and rearrange to b, n_layers, n_heads, seq_len, seq_len
                        attentions = torch.stack(attentions, dim=0)
                        attentions = einops.rearrange(attentions, 'l b h x y -> b l h x y')
                        
                        # now it's batch operations
                        # average over heads
                        res_att_mat = torch.sum(attentions, dim=2)/attentions.shape[2] # b l s s

                        # last 3 layer attention
                        last_three_attn = torch.mean(res_att_mat[:, -3:, 0, :], dim=1)
                        last_layer_attn = res_att_mat[:, -1, 0, :]

                        # then add identity (s, s) ... weird average
                        res_att_mat = res_att_mat + torch.eye(res_att_mat.shape[2])[None, None, ...].type_as(res_att_mat) # b l s s
                        res_att_mat = res_att_mat / torch.sum(res_att_mat, dim=-1)[..., None]

                        # convert to np here
                        # don't wanna touch original implementations of complex attention flow stuff
                        res_att_mat = res_att_mat.detach().cpu().numpy()

                        rollout_start = time()
                        all_joint_attentions = []
                        for b in range(attentions.shape[0]):
                            joint_attentions = compute_joint_attention(res_att_mat[b, ...], add_residual=False)
                            all_joint_attentions.append(joint_attentions)
                        all_joint_attentions = np.stack(all_joint_attentions, axis=0)

                        # all batches, last layer, [cls], all target tokens. shape: (seq_len,)
                        cls_attention_rollout = all_joint_attentions[:, -1, 0, :]
                        rollout_elapsed = time() - rollout_start
                        # print(f'rollout: {rollout_elapsed:.2f}s')

                        # # select which nodes will be input, which nodes will be output
                        batch_size = attentions.shape[0]
                        seq_len = attentions.shape[-1]
                        n_layers = attentions.shape[1]

                        rollouts.append(cls_attention_rollout)
                        last3s.append(last_three_attn.cpu())
                        lasts.append(last_layer_attn.cpu())
                        iids.append(input_ids.cpu().numpy())

                    # compute correlation.
                    batch_size = rollouts[0].shape[0]
                    for b in range(batch_size):
                        ti = (batch_idx * batch_size) + b

                        rollout_correlation, rollout_p = spearmanr(rollouts[0][b, :], rollouts[1][b, :])
                        rollout_corrs.append(rollout_correlation)
                        ti_rs.append(ti)

                        last3_corr, last3_p = spearmanr(last3s[0][b, :], last3s[1][b, :])
                        last3_corrs.append(last3_corr)
                        ti_l3s.append(ti)

                        last_corr, last_p = spearmanr(lasts[0][b, :], lasts[1][b, :])
                        last_corrs.append(last_corr)
                        ti_ls.append(ti)

                        save_example = False
                        if ti in [538, 961, 700]:
                            save_example = True
                            cross_weights, multi_weights = rollouts[0][b, :], rollouts[1][b, :]
                            method = 'roll'
                        elif ti in [344, 448, 4393]:
                            save_example = True
                            cross_weights, multi_weights = last3s[0][b, :], last3s[1][b, :]
                            method = 'l3'
                        elif ti in [2900, 1368, 2233]:
                            save_example = True
                            cross_weights, multi_weights = lasts[0][b, :], lasts[1][b, :]
                            method = 'l'
                        
                        if save_example:
                            corr = spearmanr(cross_weights, multi_weights)[0]
                            cross_iid, multi_iid = iids[0][b, :], iids[1][b, :]
                            tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
                            cross_tokens = tokenizer.convert_ids_to_tokens(cross_iid)
                            multi_tokens = tokenizer.convert_ids_to_tokens(multi_iid)

                            heatmap = pd.DataFrame(np.stack([cross_weights, multi_weights], axis=0))
                            annot = np.array([cross_tokens, multi_tokens])

                            plt.figure(figsize=(40, 10))
                            heatmap = sns.heatmap(heatmap.iloc[:, 1:], cmap='RdBu', annot=annot[:, 1:], fmt='')
                            heatmap.set_yticklabels(['cross', 'multi'])

                            fig = heatmap.get_figure()
                            out_file = f'attention_flow/en_{ti}_{method}'
                            fig.savefig(Path(out_file).with_suffix('.png'), bbox_inches='tight')

        rollout_corr_mean, rollout_corr_std = np.mean(rollout_corrs), np.std(rollout_corrs)
        last3_corr_mean, last3_corr_std = np.mean(last3_corrs), np.std(last3_corrs)
        last_corr_mean, last_corr_std = np.mean(last_corrs), np.std(last_corrs)

        # save.
        # if not corrs_saved:
        #     np.save(corrs_out_file+"rollout_rho.npy", rollout_corrs)
        #     np.save(corrs_out_file+"last3_rho.npy", last3_corrs)
        #     np.save(corrs_out_file+"last_rho.npy", last_corrs)
        
        # average correlations.
        tis = [ti_rs, ti_l3s, ti_ls]
        for i, corrs in enumerate([rollout_corrs, last3_corrs, last_corrs]):
            argsorted = np.argsort(corrs)[:3]
            print(f'{language}: {[tis[i][a] for a in argsorted]}, {[corrs[a] for a in argsorted]}')
            print("\n")

        results_mean.append([rollout_corr_mean, last3_corr_mean, last_corr_mean])
        results_std.append([rollout_corr_std, last3_corr_std, last_corr_std])

    # make the final plot
    columns = ['rollout', 'last_3', 'last']

    if Path(f'attention_flow/xnli-attention.csv').is_file():
        mean_df = pd.read_csv(f'attention_flow/xnli-attention.csv', index_col=0)
    else:
        mean_df = pd.DataFrame(results_mean, index=languages, columns=columns)
        mean_df.to_csv(f'attention_flow/xnli-attention.csv')
    
    if Path(f'attention_flow/xnli-attention_std.csv').is_file():
        std_df = pd.read_csv(f'attention_flow/xnli-attention_std.csv', index_col=0)
    else:
        std_df = pd.DataFrame(results_std, index=languages, columns=columns)
        std_df.to_csv(f'attention_flow/xnli-attention_std.csv')

    save_acc_matrix_as_heatmap(mean_df, f'attention_flow/xnli-attention.pdf')
















