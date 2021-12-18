from typing import List, Union, Tuple

import argparse
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader

from experiments.exp_def import (
    TaskDefs,
    Experiment
)
from data_utils.task_def import TaskType, EncoderModelType
from torch.utils.data import DataLoader
from mt_dnn.batcher import SingleTaskDataset, Collater
from mt_dnn.model import MTDNNModel
from mt_dnn.inference import eval_model
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import argparse

def create_heatmap(
    data_csv_path: str = '',
    data_df: Union[pd.DataFrame, None] = None,
    row_labels: List[str] = None,
    column_labels: List[str] = None,
    xaxlabel: str = None,
    yaxlabel: str = None,
    invert_y: bool = False,
    figsize: Tuple[int, int] = (14, 14),
    fontsize: int = 14,
    cmap: str = 'RdYlGn',
    out_file: str= ''):

    # read data if dataframe not directly supplied.
    if data_df is None:
        data_df = pd.read_csv(data_csv_path, index_col=0)
        assert len(out_file) > 0, f'invalid csv: {data_csv_path}'
    
    plt.figure(figsize=figsize)
    annot_kws = {
        "fontsize":fontsize,
    }
    heatmap = sns.heatmap(
        data_df.to_numpy(),
        cbar=False,
        annot=True,
        annot_kws=annot_kws,
        fmt=".2f",
        cmap=cmap)

    if invert_y:
        heatmap.invert_yaxis()

    heatmap.set_ylabel(xaxlabel, fontsize=fontsize)
    heatmap.set_xlabel(yaxlabel, fontsize=fontsize)

    heatmap.set_yticklabels(row_labels, rotation=0, fontsize=fontsize)
    heatmap.set_xticklabels(column_labels, rotation=0, fontsize=fontsize)

    fig = heatmap.get_figure()
    fig.savefig(Path(out_file).with_suffix('.pdf'), bbox_inches='tight')

def build_dataset(data_path, batch_size, max_seq_len, task_def, device_id):
    test_data_set = SingleTaskDataset(
        path=data_path,
        is_train=False,
        maxlen=max_seq_len,
        task_id=0,
        task_def=task_def
    )

    collater = Collater(is_train=False, encoder_type=EncoderModelType.BERT)

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

def evaluate_model_against_multiple_datasets(
    model: MTDNNModel,
    task: Experiment,
    metric_meta,
    datasets: List[str],
    task_def_path: str,
    device_id: int):

    accs = []
    
    for dataset in datasets:
        print(f'Evaluating on {dataset}')
        data_path = f'experiments/{task.name}/{dataset}/bert-base-multilingual-cased/{dataset}_test.json'
        test_data = build_dataset(
            data_path,
            task_def=TaskDefs(task_def_path).get_task_def(task.name.lower()),
            device_id=device_id,
            batch_size=8,
            max_seq_len=512)

        acc = get_acc(model, test_data, metric_meta, device_id, head_probe=False)
        accs.append(acc)
    
    return accs

def construct_model(checkpoint: str, task: Experiment, task_def_path: str, device_id: int):
    task_defs = TaskDefs(task_def_path)
    task_name = task.name.lower()
    assert task_name in task_defs._task_type_map
    assert task_name in task_defs._data_type_map
    assert task_name in task_defs._metric_meta_map
    metric_meta = task_defs._metric_meta_map[task_name]

    state_dict = torch.load(checkpoint, map_location=f'cuda:{device_id}')
    
    config = state_dict['config']
    config['fp16'] = False
    config['answer_opt'] = 0
    config['adv_train'] = False
    
    task_def = task_defs.get_task_def(task_name)
    task_def_list = [task_def]
    config['task_def_list'] = task_def_list
    config["cuda"] = True
    config['device'] = device_id
    del state_dict['optimizer']

    model = MTDNNModel(config, devices=[device_id], state_dict=state_dict)
    return model, metric_meta

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--model_ckpt', type=str)
    parser.add_argument('--task', type=str)
    args = parser.parse_args()

    task = Experiment[args.task.upper()]

    root_ckpt_path = Path('checkpoint/')
    encoder_type = EncoderModelType.BERT

    if task is Experiment.NLI:
        datasets = [
            'ar',
            'bg',
            'de',
            'el',
            'es',
            'fr',
            'hi',
            'ru',
            'sw',
            'th',
            'tr',
            'ur',
            'vi',
            'zh',
            'en' 
        ]
    else:
        datasets = ['en', 'fr', 'de', 'es']
    
    task_def_path = f'experiments/{task.name}/multi/task_def.yaml'
    result_matrix = np.zeros(len(datasets),)

    model, metric_meta = construct_model(args.model_ckpt, task, task_def_path, args.device_id)
    accs = evaluate_model_against_multiple_datasets(
        model,
        task,
        metric_meta,
        datasets,
        task_def_path,
        device_id=args.device_id
    )

    results = {datasets[i]: accs[i] for i in range(len(datasets))}
    results = pd.DataFrame(results)

    create_heatmap(
        data_df=results,
        row_labels=[d.upper() for d in datasets],
        column_labels=[task.name],
        xaxlabel='',
        yaxlabel='languages',
        out_file=f'evaluation_results/{task.name}'
    )
