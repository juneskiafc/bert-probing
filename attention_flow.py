from pathlib import Path
import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from experiments.exp_def import TaskDefs, EncoderModelType, TaskDef
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
import networkx as nx

def get_adjmat(mat, input_tokens=None):
    n_layers, length, _ = mat.shape
    adj_mat = np.zeros(((n_layers+1)*length, (n_layers+1)*length))

    if input_tokens is not None:
        labels_to_index = {}
        for k in np.arange(length):
            labels_to_index['C_'+str(k)] = k

    for i in np.arange(1,n_layers+1):
        for k_f in np.arange(length):
            index_from = (i)*length+k_f

            label = "L"+str(i)+"_"+str(k_f)
            labels_to_index[label] = index_from

            for k_t in np.arange(length):
                index_to = (i-1)*length+k_t
                adj_mat[index_from][index_to] = mat[i-1][k_f][k_t]
                
    return adj_mat, labels_to_index 

def get_graph(adjmat):
    A = adjmat
    G = nx.from_numpy_matrix(A, create_using=nx.DiGraph())
    for i in np.arange(A.shape[0]):
        for j in np.arange(A.shape[1]):
            nx.set_edge_attributes(G, {(i,j): A[i,j]}, 'capacity')
    
    return G

def draw_attention_graph(adjmat, labels_to_index, n_layers, length):
    A = adjmat
    G = nx.from_numpy_matrix(A, create_using=nx.DiGraph())
    for i in np.arange(A.shape[0]):
        for j in np.arange(A.shape[1]):
            nx.set_edge_attributes(G, {(i,j): A[i,j]}, 'capacity')

    pos = {}
    label_pos = {}
    for i in np.arange(n_layers+1):
        for k_f in np.arange(length):
            pos[i*length+k_f] = ((i+0.4)*2, length - k_f)
            label_pos[i*length+k_f] = (i*2, length - k_f)

    index_to_labels = {}
    for key in labels_to_index:
        index_to_labels[labels_to_index[key]] = key.split("_")[-1]
        if labels_to_index[key] >= length:
            index_to_labels[labels_to_index[key]] = ''

    #plt.figure(1,figsize=(20,12))

    nx.draw_networkx_nodes(G,pos,node_color='green', labels=index_to_labels, node_size=50)
    nx.draw_networkx_labels(G,pos=label_pos, labels=index_to_labels, font_size=18)

    all_weights = []
    #4 a. Iterate through the graph nodes to gather all the weights
    for (node1,node2,data) in G.edges(data=True):
        all_weights.append(data['weight']) #we'll use this when determining edge thickness

    #4 b. Get unique weights
    unique_weights = list(set(all_weights))

    #4 c. Plot the edges - one by one!
    for weight in unique_weights:
        #4 d. Form a filtered list with just the weight you want to draw
        weighted_edges = [(node1,node2) for (node1,node2,edge_attr) in G.edges(data=True) if edge_attr['weight']==weight]
        #4 e. I think multiplying by [num_nodes/sum(all_weights)] makes the graphs edges look cleaner
        
        w = weight #(weight - min(all_weights))/(max(all_weights) - min(all_weights))
        width = w
        nx.draw_networkx_edges(G,pos,edgelist=weighted_edges,width=width, edge_color='darkblue')
    
    return G

def compute_flows(G, labels_to_index, input_nodes, output_nodes, length):
    flow_values = np.zeros((len(input_nodes), len(output_nodes)))

    # from target to source
    for oi, oup_node_key in enumerate(output_nodes):
        if oup_node_key not in input_nodes:
            u = labels_to_index[oup_node_key]

            for ii, inp_node_key in enumerate(input_nodes):
                v = labels_to_index[inp_node_key]
                flow_value = nx.maximum_flow_value(G,u,v, flow_func=nx.algorithms.flow.edmonds_karp)

                flow_values[ii, oi] = flow_value
            flow_values[oi] /= flow_values[oi].sum()
            
    return flow_values

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

def create_model(setting, device_id):
    checkpoint_file = Path('checkpoint').joinpath(setting, 'model_5.pt')
    dummy_task_def_path = Path(f'experiments/NLI/').joinpath(setting, 'task_def.yaml')

    state_dict = torch.load(checkpoint_file, map_location=torch.device(device_id))
    task_defs = TaskDefs(dummy_task_def_path)

    config = state_dict['config']
    config["cuda"] = True
    task_def = task_defs.get_task_def(setting)
    task_def_list = [task_def]
    config['task_def_list'] = task_def_list

    ## temp fix
    config['fp16'] = False
    config['answer_opt'] = 0
    config['adv_train'] = False
    del state_dict['optimizer']

    model = MTDNNModel(config, state_dict=state_dict, device=device_id)
    return model

def load_data(data_file, task_def, language):
    task_defs = TaskDefs(task_def)
    test_data_set = SingleTaskDataset(
        data_file,
        False,
        maxlen=512,
        task_id=0,
        task_def=task_defs.get_task_def(language),
        bert_model='bert-base-multilingual-cased'
    )

    encoder_type = EncoderModelType.BERT
    collater = Collater(is_train=False, encoder_type=encoder_type)
    test_data = DataLoader(test_data_set, batch_size=8, collate_fn=collater.collate_fn, pin_memory=True)

    return test_data

def get_attention(model, batch_meta, batch_data):
    batch_meta, batch_data = Collater.patch_data(device_id, batch_meta, batch_data)

    # fwd pass and get CLS attention
    # prepare input
    task_id = batch_meta['task_id']
    task_def = TaskDef.from_dict(batch_meta['task_def'])
    task_type = task_def.task_type
    task_obj = tasks.get_task_obj(task_def)
    inputs = batch_data[:batch_meta['input_len']]
    if len(inputs) == 3:
        inputs.append(None)
        inputs.append(None)
    inputs.append(task_id)

    input_ids = inputs[0]
    token_type_ids = inputs[1]
    attention_mask = inputs[2]
    y_input_ids = None

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

                        # adjacency matrix
                        # adj_mats = []
                        # all_labels_to_index = []
                        # for b in range(attentions.shape[0]):
                        #     input_tokens = input_ids[b]
                        #     res_adj_mat, labels_to_index = get_adjmat(res_att_mat[b, ...], input_tokens)
                        #     adj_mats.append(res_adj_mat)
                        #     all_labels_to_index.append(labels_to_index)

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

                        # graphs for max flow
                        # flow_start = time()
                        # graphs = [get_graph(adjmat) for adjmat in adj_mats]
                        # print(f'flow (creating graphs): {time()-flow_start:.2f}s')

                        # # select which nodes will be input, which nodes will be output
                        batch_size = attentions.shape[0]
                        seq_len = attentions.shape[-1]
                        n_layers = attentions.shape[1]

                        # input_nodes = [f'C_{i}' for i in range(seq_len)]
                        # output_nodes = [f'L{n_layers}_0']

                        # cls_attention_flow = []
                        # for i, graph in enumerate(graphs):
                        #     flow_start_single_graph = time()
                        #     flow_values = compute_flows(graph, all_labels_to_index[i], input_nodes, output_nodes, length=seq_len)
                        #     raise ValueError(f'flow (one graph): {time()-flow_start_single_graph:.2f}s')
                        #     cls_attention_flow.append(flow_values)
                        
                        # cls_attention_flow = np.stack(cls_attention_flow, axis=0)

                        # flow_elapsed = time() - flow_start
                        # print(f'flow: {flow_elapsed:.2f}s')
                        
                        # collect attention flow and attention rollout
                        # output_root = f'attention_flow/{setting}/{language}/{batch_idx*batch_size}_{(batch_idx*batch_size)+batch_size-1}'

                        # # flow_file = output_root + '_flow.npy'
                        # rollout_file = output_root + '_roll.npy'
                        # last3_file = output_root + '_l3.npy'
                        # last_file = output_root + '_last.npy'

                        # np.save(flow_file, cls_attention_flow)
                        # np.save(rollout_file, cls_attention_rollout)
                        # np.save(last3_file, last_three_attn.cpu())
                        # np.save(last_file, last_layer_attn.cpu())

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
















