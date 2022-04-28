import argparse
from pathlib import Path
import os
import torch
from torch import nn
from torch.utils.data import DataLoader

from experiments.exp_def import TaskDefs, EncoderModelType
from data_utils.task_def import TaskType
from torch.utils.data import DataLoader
from mt_dnn.batcher import SingleTaskDataset, Collater
from mt_dnn.model import MTDNNModel
from mt_dnn.inference import eval_model
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(),
                    help='whether to use GPU acceleration.')
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--task', type=str)
parser.add_argument('--subtask', type=str, default='head_probe')

args = parser.parse_args()
TASK = args.task
SUBTASK = args.subtask
MAX_SEQ_LEN = 512
BATCH_SIZE = 8

if TASK == 'NLI':
    task_type = TaskType.Classification
elif TASK == 'POS':
    task_type = TaskType.SequenceLabeling
elif TASK == 'NER':
    task_type = TaskType.Classification
else:
    raise NotImplementedError

def save_acc_matrix_as_heatmap(input_csv='', file_df=None, invert_y=False, axlabels=None, out_file=''):
    if file_df is None:
        file_df = pd.read_csv(input_csv, index_col=0)
        assert len(out_file) > 0
    
    row_labels = file_df.index
    column_labels = file_df.columns
    data = file_df.to_numpy()

    plt.figure(figsize=(14, 14))
    annot_kws = {
        "fontsize":14,
    }
    heatmap = sns.heatmap(
        data,
        cbar=False,
        annot=True,
        annot_kws=annot_kws,
        fmt=".2f",
        # xticklabels=column_labels,
        # yticklabels=row_labels,
        cmap='RdYlGn')

    if invert_y:
        heatmap.invert_yaxis()
    if axlabels is not None:
        heatmap.set_ylabel(axlabels[1], fontsize=14)
        heatmap.set_xlabel(axlabels[0], fontsize=14)

    heatmap.set_yticklabels(row_labels, rotation=0, fontsize=14)
    heatmap.set_xticklabels(column_labels, rotation=0, fontsize=14)
    fig = heatmap.get_figure()

    if len(input_csv) > 0: 
        fig.savefig(Path(input_csv).with_suffix('.png'))
    else:
        fig.savefig(Path(out_file).with_suffix('.pdf'), bbox_inches='tight')

def build_dataset(data_path, encoder_type, batch_size, max_seq_len, task_def, device_id):
    test_data_set = SingleTaskDataset(
        path=data_path,
        is_train=False,
        maxlen=max_seq_len,
        task_id=0,
        task_def=task_def
    )

    collater = Collater(is_train=False, encoder_type=encoder_type)

    test_data = DataLoader(
        test_data_set,
        batch_size=batch_size,
        collate_fn=collater.collate_fn,
        pin_memory=device_id > 0
    )

    return test_data

def get_acc(model, test_data, metric_meta, device_id, head_probe):
    with torch.no_grad():
        model.network.eval()
        model.network.to(device_id)
        
        results = eval_model(
            model,
            test_data,
            task_type=task_type,
            metric_meta=metric_meta,
            device=device_id,
            with_label=True,
            head_probe=head_probe
        )
    metrics = results[0]
    predictions = results[1]
    golds = results[3]
    ids = results[4]

    preds_df = pd.Series(predictions)
    golds_df = pd.Series(golds)
    id_df = pd.Series(ids)
    return metrics['ACC'], preds_df, golds_df, id_df

def evaluate_model_against_multiple_datasets(model,
                                            datasets,
                                            task_def,
                                            encoder_type,
                                            batch_size,
                                            max_seq_len,
                                            device_id):
    assert TASK == 'NLI'
    accs = []
    
    for dataset in datasets:
        print(f'\tevaluating on {dataset}')
        data_path = f'experiments/attention-probing/{dataset}/bert-base-multilingual-cased/{dataset}_test.json'
        test_data = build_dataset(data_path, encoder_type, batch_size, max_seq_len, task_def, device_id)
        acc = get_acc(model, test_data, metric_meta, device_id, head_probe=False)
        accs.append(acc)
    
    return accs

def evaluate_head_probe(hlhis, device_id, task_def, encoder_type, batch_size, max_seq_len, root_ckpt_path, setting, data_path, task_name, return_base_model):
    assert setting in ['multi', 'cross', 'base']
    
    if setting == 'cross':
        shorthand = 'cl'
    elif setting == 'multi':
        shorthand = 'ml'
    elif setting == 'base':
        shorthand = 'base'
    
    test_data = build_dataset(
        data_path,
        encoder_type,
        batch_size,
        max_seq_len,
        TaskDefs(task_def).get_task_def(task_name),
        device_id)
    
    if not return_base_model:
        base_model_ckpt = root_ckpt_path.joinpath(f'{setting}/model_5.pt')
    else:
        # just for config stuff to set up MTDNN.
        base_model_ckpt = root_ckpt_path.joinpath(f'cross/model_5.pt')
    
    model, metric_meta = construct_model(base_model_ckpt, task_name, task_def, return_base_model=return_base_model)

    if TASK == 'NLI':
        n_classes = 3
    elif TASK == 'POS':
        n_classes = 17
    elif TASK == 'NER':
        n_classes = 7

    output_dir = Path(f'score_outputs/{TASK}/{SUBTASK}/{setting}')
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, (hl, hi) in enumerate(hlhis):
        print(f'\n[{TASK}, {task_name}] computing metrics for {hl}_{hi}...')
        output_file_for_head = output_dir.joinpath(f'{hl}_{hi}.csv')
        if output_file_for_head.is_file():
            return

        # load state dict for the attention head.
        state_dict_for_head = root_ckpt_path.joinpath(f'head_probing/{TASK}/{SUBTASK}{setting}/{shorthand}_{hl}/{shorthand}_{hl}_{hi}/')
        state_dict_for_head = list(state_dict_for_head.rglob("*.pt"))[0]
        state_dict_for_head = torch.load(state_dict_for_head)
        if TASK != 'NLI':
            state_dict_for_head = state_dict_for_head.state_dict()

        # then attach the probing layer
        model.attach_head_probe(hl, hi, n_classes=n_classes)

        # get the layer and check
        layer = model.network.get_attention_layer(hl)
        assert hasattr(layer, 'head_probe_dense_layer')

        # and load (put it on same device)
        if 'head_probe_dense_layer.weight' in state_dict_for_head.keys():
            weight = state_dict_for_head['head_probe_dense_layer.weight']
            bias = state_dict_for_head['head_probe_dense_layer.bias']
        else:
            weight = state_dict_for_head[f'bert.encoder.layer.{hl}.attention.self.head_probe_dense_layer.weight']
            bias = state_dict_for_head[f'bert.encoder.layer.{hl}.attention.self.head_probe_dense_layer.bias']
        
        layer.head_probe_dense_layer.weight = nn.Parameter(weight.to(device_id))
        layer.head_probe_dense_layer.bias = nn.Parameter(bias.to(device_id))

        # compute acc and save
        _, preds_for_layer, golds, ids = get_acc(model, test_data, metric_meta, device_id, head_probe=True)
        pd.Series(preds_for_layer).to_csv(output_file_for_head)

        # save labels and ids
        if not output_dir.joinpath(f'labels.csv').is_file():
            pd.Series(golds).to_csv(output_dir.joinpath(f'labels.csv'))
        
        if not output_dir.joinpath(f'ids.csv').is_file():
            pd.Series(ids).to_csv(output_dir.joinpath(f'ids.csv'))
        
        # detach
        model.detach_head_probe(hl)

def construct_model(checkpoint, task, task_def, return_base_model=False):
    task_defs = TaskDefs(task_def)
    assert task in task_defs._task_type_map
    assert task in task_defs._data_type_map
    assert task in task_defs._metric_meta_map

    prefix = task.split('_')[0]
    metric_meta = task_defs._metric_meta_map[task]

    # load model
    assert os.path.exists(checkpoint)

    if args.device_id > 0:
        state_dict = torch.load(checkpoint)
    else:
        state_dict = torch.load(checkpoint, map_location="cpu")
    
    config = state_dict['config']

    config['fp16'] = False
    config['answer_opt'] = 0
    config['adv_train'] = False
    
    task_def = task_defs.get_task_def(prefix)
    task_def_list = [task_def]
    config['task_def_list'] = task_def_list
    config["cuda"] = args.device_id >= 0
    config['device'] = args.device_id
    del state_dict['optimizer']

    model = MTDNNModel(config, device=config['device'])

    # scoring_list classification head doesn't matter because we're just taking
    # the head probe outputs.
    state_dict['state']['scoring_list.0.weight'] = model.network.state_dict()['scoring_list.0.weight']
    state_dict['state']['scoring_list.0.bias'] = model.network.state_dict()['scoring_list.0.bias']

    if not return_base_model:
        model.load_state_dict(state_dict)
    
    return model, metric_meta

def distribute_heads_to_gpus(available_devices):
    hlhis = []
    devices = []
    n_per_gpu = 144 // len(available_devices)
    gpu_group = []
    curr_device_idx = 0
    n_heads_cumul = 0

    for hl in range(12):
        for hi in range(12):
            n_heads_cumul += 1
            gpu_group.append((hl, hi))

            if ((n_heads_cumul % n_per_gpu == 0 and n_heads_cumul != 0) or n_heads_cumul == 144):
                hlhis.append(gpu_group)
                devices.append(available_devices[curr_device_idx])
                print(f'adding {len(gpu_group)} to device {available_devices[curr_device_idx]}')
                gpu_group = []
                curr_device_idx += 1
    
    return hlhis, devices

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(),
                        help='whether to use GPU acceleration.')
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--task', type=str)
    parser.add_argument('--subtask', type=str, default='head_probe')

    if False:
        models = [
            # 'ml_ltr'
            'base',
            'cross',
            'multi',
            # 'lang_spec/ml_ar',
            # 'lang_spec/ml_bg',
            # 'lang_spec/ml_de',
            # 'lang_spec/ml_el',
            # 'lang_spec/ml_es',
            # 'lang_spec/ml_fr',
            # 'lang_spec/ml_hi',
            # 'lang_spec/ml_ru',
            # 'lang_spec/ml_sw',
            # 'lang_spec/ml_th',
            # 'lang_spec/ml_tr',
            # 'lang_spec/ml_ur',
            # 'lang_spec/ml_vi',
            # 'lang_spec/ml_zh',
        ]

        if not SUBTASK == 'head_probe':
            single_lang_datasets = [
                'multi-ar',
                'multi-bg',
                'multi-de',
                'multi-el',
                'multi-es',
                'multi-fr',
                'multi-hi',
                'multi-ru',
                'multi-sw',
                'multi-th',
                'multi-tr',
                'multi-ur',
                'multi-vi',
                'multi-zh',
                'multi-en' 
            ]

    else:
        models = [
            'cross',
            # 'multi',
            # 'base'
            ]
        assert SUBTASK == 'head_probe'

    root_ckpt_path = Path('checkpoint/')
    encoder_type = EncoderModelType.BERT

    for model_idx, model in enumerate(models):    
        if TASK == 'NLI':
            if model in ['cross', 'multi', 'base']:
                task_def = f'experiments/{TASK}/{model}/task_def.yaml'
            else:
                lang = model.split("/")[1].split("_")[1]
                task_def = f'experiments/{TASK}/{f"multi-{lang}"}/task_def.yaml'

            if SUBTASK == 'cross_evaluate':
                assert model == 'multi' or model == 'multi-ltr'
                result_matrix = np.zeros((len(models), len(single_lang_datasets)))
                checkpoint = list(root_ckpt_path.joinpath(model).rglob("model_5_*.pt"))[0]

                model = construct_model(checkpoint, task_def, head_probe=False)
                accs = evaluate_model_against_multiple_datasets(
                    model,
                    single_lang_datasets,
                    task_def,
                    encoder_type,
                    args.batch_size,
                    args.max_seq_len,
                    device_id=args.device_id
                )
                for j, acc in enumerate(accs):
                    result_matrix[model_idx, j] = acc
            
            elif SUBTASK == 'head_probe':
                out_csv_file = f'score_outputs/NLI/head_probe/{model}/acc_per_head.csv'
                if not Path(out_csv_file).is_file():
                    data_path = f'experiments/NLI/cross/bert-base-multilingual-cased/cross_test.json'
                    result_matrix, result_df = evaluate_head_probe(
                        task_def,
                        encoder_type,
                        BATCH_SIZE,
                        MAX_SEQ_LEN,
                        device_id=args.device_id,
                        root_ckpt_path=root_ckpt_path,
                        setting=model,
                        data_path=data_path,
                        task_name=model,
                        return_base_model=model=='base'
                    )
                else:
                    result_df_all = pd.read_csv(out_csv_file, index_col=0)

        else:
            task_def = f'experiments/{TASK}/task_def.yaml'
            data_path = f'experiments/{TASK}/bert-base-multilingual-cased/{TASK.lower()}_test.json'
            out_pdf_file = f'score_outputs/{TASK}/head_probe/{model}/xnli_{model}-{TASK.lower()}'

            if not Path(out_pdf_file).is_file():
                available_devices = [0, 1, 2, 3]
                hlhis, devices = distribute_heads_to_gpus(available_devices)
                args = []

                for i, hlhi_set in enumerate(hlhis):
                    args_ = [
                        hlhi_set,
                        devices[i],
                        task_def,
                        encoder_type,
                        BATCH_SIZE,
                        MAX_SEQ_LEN,
                        root_ckpt_path,
                        model,
                        data_path,
                        TASK.lower(),
                        model=='base'
                    ]
                    args.append(tuple(args_))
                
                with torch.multiprocessing.get_context('spawn').Pool(len(available_devices)) as p:
                    p.starmap(evaluate_head_probe, args)
                
                # collect.
                output_dir = Path(f'score_outputs/{TASK}/head_probe/{model}')
                full_layer_results = []
                for hl in range(12):
                    for hi in range(12):
                        full_layer_results.append(pd.read_csv(output_dir.joinpath(f'{hl}_{hi}.csv'), index_col=0))
                
                full_ids = pd.read_csv(output_dir.joinpath(f'ids.csv'), index_col=0)
                full_golds = pd.read_csv(output_dir.joinpath(f'labels.csv'), index_col=0)

                full_results = pd.concat([full_ids, full_golds] + full_layer_results, axis=1)
                full_results.to_csv(output_dir.joinpath(f'results.csv'))
