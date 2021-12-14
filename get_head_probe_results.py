from typing import List
import argparse
from pathlib import Path
import os
import torch
from torch import nn
from torch.utils.data import DataLoader

from torch.utils.data import DataLoader
from mt_dnn.batcher import SingleTaskDataset, Collater
from mt_dnn.model import MTDNNModel
from mt_dnn.inference import eval_model
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import argparse
from data_utils.task_def import EncoderModelType, TaskType
from experiments.exp_def import (
    Experiment,
    LingualSetting,
    TaskDefs,
)

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
            task_type=TaskType.Classification,
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

def evaluate_head_probe(
    hlhis: List,
    finetuned_task: Experiment,
    downstream_task: Experiment,
    setting: LingualSetting,
    batch_size: int,
    max_seq_len: int,
    device_id: int):

    """
    Evaluate head probe for task_finetuned model on a downstream subtask.
    """
    output_dir = Path('head_probe_outputs').joinpath(
        finetuned_task.name,
        downstream_task.name,
        setting.name.lower()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    task_def_path = Path('experiments').joinpath(
        finetuned_task.name,
        'cross',
        'task_def.yaml'
    )
    task_def = TaskDefs(task_def_path).get_task_def(finetuned_task.name.lower())
    data_path = Path('experiments').joinpath(
        downstream_task.name,
        'multi',
        'bert-base-multilingual-cased',
        f'{downstream_task.name.lower()}_test.json'
    )

    test_data = build_dataset(
        data_path,
        EncoderModelType.BERT,
        batch_size,
        max_seq_len,
        task_def,
        device_id)
    
    model, metric_meta = construct_model(
        finetuned_task,
        setting,
        device_id)

    for (hl, hi) in hlhis:
        print(f'\n[{finetuned_task.name}/{downstream_task.name}, {setting.name}]: {hl}_{hi}...')
        output_file_for_head = output_dir.joinpath(f'{hl}_{hi}.csv')
        if output_file_for_head.is_file():
            return

        # load state dict for the attention head.
        state_dict_for_head = Path('checkpoint').joinpath(
            'head_probing',
            finetuned_task.name,
            downstream_task.name,
            setting.name.lower(),
            str(hl),
            str(hi))

        state_dict_for_head = list(state_dict_for_head.rglob("*.pt"))[0]
        state_dict_for_head = torch.load(state_dict_for_head)['state']

        # then attach the probing layer
        model.attach_head_probe(hl, hi, task_def.n_class)

        # get the layer and check
        layer = model.network.get_attention_layer(hl)
        assert hasattr(layer, 'head_probe_dense_layer')

        # and load (put it on same device)
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

def construct_model(finetuned_task: Experiment, setting: LingualSetting, device_id: int):
    if setting is LingualSetting.BASE:
        task_def_path = Path('experiments').joinpath(
            finetuned_task.name,
            'cross',
            'task_def.yaml'
        )
    else:
        task_def_path = Path('experiments').joinpath(
            finetuned_task.name,
            setting.name.lower(),
            'task_def.yaml'
        )

    task_defs = TaskDefs(task_def_path)
    assert finetuned_task.name.lower() in task_defs._task_type_map
    assert finetuned_task.name.lower() in task_defs._data_type_map
    assert finetuned_task.name.lower() in task_defs._metric_meta_map

    metric_meta = task_defs._metric_meta_map[finetuned_task.name.lower()]

    # load model
    if setting is not LingualSetting.BASE:
        checkpoint_dir = Path('checkpoint').joinpath(f'{finetuned_task.name}_{setting.name.lower()}')
        checkpoint = list(checkpoint_dir.rglob('model_5*.pt'))[0]
        assert os.path.exists(checkpoint), checkpoint
    else:
        checkpoint_dir = Path('checkpoint').joinpath(f'{finetuned_task.name}_cross')
        checkpoint = list(checkpoint_dir.rglob('model_5*.pt'))[0]

    state_dict = torch.load(checkpoint)
    config = state_dict['config']

    config['fp16'] = False
    config['answer_opt'] = 0
    config['adv_train'] = False
    
    task_def = task_defs.get_task_def(finetuned_task.name.lower())
    task_def_list = [task_def]
    config['task_def_list'] = task_def_list
    config["cuda"] = True
    config['device'] = device_id

    if 'optimizer' in state_dict:
        del state_dict['optimizer']

    model = MTDNNModel(config, devices=[device_id])

    # scoring_list classification head doesn't matter because we're just taking
    # the head probe outputs.
    if 'scoring_list.0.weight' in state_dict['state']:
        state_dict['state']['scoring_list.0.weight'] = model.network.state_dict()['scoring_list.0.weight']
        state_dict['state']['scoring_list.0.bias'] = model.network.state_dict()['scoring_list.0.bias']

    if setting is not LingualSetting.BASE:
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
    parser.add_argument('--devices', nargs='+', default='')
    parser.add_argument('--finetuned_task', type=str)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_seq_len', type=int, default=512)

    args = parser.parse_args()

    if args.devices == '':
        args.devices = list(range(torch.cuda.device_count()))
    
    finetuned_task = Experiment[args.finetuned_task.upper()]
    root_ckpt_path = Path('checkpoint/')
    root_out_path = Path('head_probe_outputs')
    encoder_type = EncoderModelType.BERT

    for downstream_task in list(Experiment):
        if downstream_task is Experiment.NLI:
            continue
        
        for setting in list(LingualSetting):
            out_pdf_file = root_out_path.joinpath(
                finetuned_task.name,
                downstream_task.name,
                setting.name.lower(),
                f'xnli_{setting.name.lower()}-{finetuned_task.name.lower()}')

            if not Path(out_pdf_file).is_file():
                available_devices = [int(d) for d in args.devices]
                hlhis, devices = distribute_heads_to_gpus(available_devices)
                eval_head_probe_args = []

                for i, hlhi_set in enumerate(hlhis):
                    args_ = [
                        hlhi_set,
                        finetuned_task,
                        downstream_task,
                        setting,
                        args.batch_size,
                        args.max_seq_len,
                        devices[i]
                    ]
                    eval_head_probe_args.append(tuple(args_))
                
                with torch.multiprocessing.get_context('spawn').Pool(len(available_devices)) as p:
                    p.starmap(evaluate_head_probe, eval_head_probe_args)
                
                # collect.
                output_dir = Path('head_probe_outputs').joinpath(
                    finetuned_task.name,
                    downstream_task.name,
                    setting.name
                )

                full_layer_results = []
                for hl in range(12):
                    for hi in range(12):
                        result_for_layer = pd.read_csv(output_dir.joinpath(f'{hl}_{hi}.csv'), index_col=0)
                        full_layer_results.append(result_for_layer)
                
                full_ids = pd.read_csv(output_dir.joinpath(f'ids.csv'), index_col=0)
                full_golds = pd.read_csv(output_dir.joinpath(f'labels.csv'), index_col=0)

                full_results = pd.concat([full_ids, full_golds] + full_layer_results, axis=1)
                full_results.to_csv(output_dir.joinpath(f'results.csv'))
