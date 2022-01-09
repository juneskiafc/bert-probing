from typing import List, Union, Tuple, Dict
import argparse
from pathlib import Path
import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader

from mt_dnn.batcher import SingleTaskDataset, Collater
from mt_dnn.model import MTDNNModel
from mt_dnn.inference import eval_model

from data_utils.task_def import EncoderModelType, TaskType
from experiments.exp_def import (
    Experiment,
    LingualSetting,
    TaskDefs,
)

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
    out_file: str= ''
    ):
    """
    General heatmap from data.
    """
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

    heatmap.set_ylabel(yaxlabel, fontsize=fontsize)
    heatmap.set_xlabel(xaxlabel, fontsize=fontsize)

    heatmap.set_yticklabels(row_labels, rotation=0, fontsize=fontsize)
    heatmap.set_xticklabels(column_labels, rotation=0, fontsize=fontsize)

    fig = heatmap.get_figure()
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

def construct_model(task: Experiment, setting: LingualSetting, device_id: int):
    task_def_path = Path('experiments').joinpath(
        task.name,
        'task_def.yaml'
    )

    task_defs = TaskDefs(task_def_path)
    assert task.name.lower() in task_defs._task_type_map
    assert task.name.lower() in task_defs._data_type_map
    assert task.name.lower() in task_defs._metric_meta_map
    metric_meta = task_defs._metric_meta_map[task.name.lower()]

    # load model
    if setting is not LingualSetting.BASE:
        checkpoint_dir = Path('checkpoint').joinpath(f'{task.name}_{setting.name.lower()}')
        checkpoint = list(checkpoint_dir.rglob('model_5*.pt'))[0]
        assert os.path.exists(checkpoint), checkpoint
    else:
        checkpoint_dir = Path('checkpoint').joinpath(f'{task.name}_cross')
        checkpoint = list(checkpoint_dir.rglob('model_5*.pt'))[0]

    state_dict = torch.load(checkpoint)
    config = state_dict['config']

    config['fp16'] = False
    config['answer_opt'] = 0
    config['adv_train'] = False
    
    task_def = task_defs.get_task_def(task.name.lower())
    task_def_list = [task_def]
    config['task_def_list'] = task_def_list
    config["cuda"] = True
    config['device'] = device_id
    config['head_probe'] = False

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

def get_acc(model, test_data, metric_meta, device_id, model_probe):
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
            model_probe=model_probe
        )
    metrics = results[0]
    predictions = results[1]
    golds = results[3]
    ids = results[4]

    preds_df = pd.Series(predictions)
    golds_df = pd.Series(golds)
    id_df = pd.Series(ids)
    return metrics['F1MAC'], preds_df, golds_df, id_df

def evaluate_model_probe(
    task: Experiment,
    finetuned_task: Union[Experiment, None],
    setting: LingualSetting,
    results: pd.DataFrame,
    batch_size: int=8,
    max_seq_len: int=512,
    device_id: int=0):

    """
    Evaluate model probe for a NLI model on a task.
    """
    task_def_path = Path('experiments').joinpath(
        task.name,
        'task_def.yaml'
    )
    task_def = TaskDefs(task_def_path).get_task_def(task.name.lower())
    data_path = Path('experiments').joinpath(
        task.name,
        'multi',
        'bert-base-multilingual-cased',
        f'{task.name.lower()}_test.json'
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
    
    if finetuned_task is not None:
        print(f'\n{finetuned_task.name}/{setting.name.lower()} -> {task.name}')
    else:
        print(f'\nmBERT -> {task.name}')
        assert setting is LingualSetting.BASE
    
    # load state dict for the attention head.
    if finetuned_task is not None:
        state_dict_for_head = Path('checkpoint').joinpath(
            'full_model_probe',
            finetuned_task.name,
            task.name,
            setting.name.lower()
        )
    else:
        state_dict_for_head = Path('checkpoint').joinpath(
            'full_model_probe',
            'mBERT',
            task.name,
        )

    state_dict_for_head = list(state_dict_for_head.rglob("*.pt"))[0]
    state_dict_for_head = torch.load(state_dict_for_head)['state']

    # then attach the probing layer
    model.attach_model_probe(task_def.n_class)

    # get the layer and check
    layer = model.network.get_pooler_layer()
    assert hasattr(layer, 'model_probe_head')

    # and load (put it on same device)
    weight = state_dict_for_head[f'bert.pooler.model_probe_head.weight']
    bias = state_dict_for_head[f'bert.pooler.model_probe_head.bias']
    
    layer.model_probe_head.weight = nn.Parameter(weight.to(device_id))
    layer.model_probe_head.bias = nn.Parameter(bias.to(device_id))

    # compute acc and save
    acc, preds_for_layer, golds, ids = get_acc(model, test_data, metric_meta, device_id, model_probe=True)

    if finetuned_task is None:
        results.loc[f'mBERT', task.name] = acc
    else:
        results.loc[f'{finetuned_task.name}_{setting.name.lower()}', task.name] = acc
        
    return results

def combine_all_model_probe_scores():
    combined_results = None

    for task in list(Experiment):        
        result_for_task = f'model_probe_outputs/{task.name}/final_results.csv'
        result_for_task = pd.read_csv(result_for_task, index_col=0)

        if combined_results is None:
            combined_results = result_for_task
        else:
            combined_results = pd.concat([combined_results, result_for_task], axis=0)
    
    combined_results.to_csv('model_probe_outputs/final_result.csv')
    create_heatmap(
        data_df=combined_results,
        row_labels=list(combined_results.index),
        column_labels=list(combined_results.columns),
        xaxlabel='task',
        yaxlabel='model',
        out_file=f'model_probe_outputs/final_result'
    )

def get_model_probe_scores(args):
    tasks = list(Experiment)
    tasks.remove(Experiment.NLI)

    if args.finetuned_task != '':
        finetuned_task = Experiment[args.finetuned_task.upper()]
    else:
        finetuned_task = None
    
    final_results_out_file = Path(f'model_probe_outputs/{finetuned_task.name}/final_results.csv')
    if final_results_out_file.is_file():
        final_results = pd.read_csv(final_results_out_file, index_col=0)
    else:
        final_results_out_file.parent.mkdir(parents=True, exist_ok=True)

        results_out_file = final_results_out_file.parent.joinpath('results.csv')
        if results_out_file.is_file():
            results = pd.read_csv(results_out_file, index_col=0)
        else:
            results = pd.DataFrame(np.zeros((2, 4)))
            results.index = [
                f'{finetuned_task.name}_cross',
                f'{finetuned_task.name}_multi',
            ]
            results.columns = ['POS', 'NER', 'PAWSX', 'MARC']

            tasks = list(Experiment)
            tasks.remove(Experiment.NLI)
            for task in tasks:
                for setting in [LingualSetting.CROSS, LingualSetting.MULTI]:
                    evaluate_model_probe(task, finetuned_task, setting, results, device_id=args.device_id)
            results.to_csv(results_out_file)
        
        base_results_out_file = Path(f'model_probe_outputs/base/results.csv')
        if not base_results_out_file.is_file():
            base_results_out_file.parent.mkdir(parents=True, exist_ok=True)
            base_results = pd.DataFrame(np.zeros((1, 4)))
            base_results.index = ['mBERT']
            base_results.columns = ['POS', 'NER', 'PAWSX', 'MARC']

            for task in tasks:
                evaluate_model_probe(task, None, LingualSetting.BASE, base_results, device_id=args.device_id)
            base_results.to_csv(base_results_out_file)
        else:
            base_results = pd.read_csv(base_results_out_file, index_col=0)

        print(results)
        print(base_results)
        final_results = pd.DataFrame(results.values - base_results.values)
        final_results.index = [f'{finetuned_task.name}_cross', f'{finetuned_task.name}_multi']
        final_results.columns = ['POS', 'NER', 'PAWSX', 'MARC']
        final_results.to_csv(final_results_out_file)

    create_heatmap(
        data_df=final_results,
        row_labels=[f'{finetuned_task.name}_cross',f'{finetuned_task.name}_multi'],
        column_labels=['POS', 'NER', 'PAWSX', 'MARC'],
        xaxlabel='task',
        yaxlabel='model',
        out_file=f'model_probe_outputs/{finetuned_task.name}/final_results'
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--finetuned_task', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_seq_len', type=int, default=512)
    args = parser.parse_args()
    
    # get_model_probe_scores(args)
    combine_all_model_probe_scores()
