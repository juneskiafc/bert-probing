from typing import List, Union, Tuple
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
from data_utils.metrics import Metric
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

def build_dataset(data_path, encoder_type, batch_size, max_seq_len, task_def):
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
        pin_memory=True
    )

    return test_data

def construct_model(task: Experiment, setting: LingualSetting, device_id: int):
    task_def_path = Path('experiments').joinpath(task.name, 'task_def.yaml')
    task_name = task.name.lower()

    task_defs = TaskDefs(task_def_path)
    task_def = task_defs.get_task_def(task_name)

    # load model
    if setting is LingualSetting.BASE:
        try:
            checkpoint_dir = Path('checkpoint').joinpath(f'{task.name}_cross')
            checkpoint = list(checkpoint_dir.rglob('model_5*.pt'))[0]
        except:
            checkpoint_dir = Path('checkpoint').joinpath(f'{task.name}_multi')
            checkpoint = list(checkpoint_dir.rglob('model_5*.pt'))[0]
        
    checkpoint = list(checkpoint_dir.rglob('model_5*.pt'))[0]
    assert os.path.exists(checkpoint), checkpoint

    state_dict = torch.load(checkpoint, map_location=f'cuda:{device_id}')
    config = state_dict['config']

    config['fp16'] = False
    config['answer_opt'] = 0
    config['adv_train'] = False
    
    task_def_list = [task_def]
    config['task_def_list'] = task_def_list
    config["cuda"] = True
    config['device'] = device_id
    config['head_probe'] = False

    if 'optimizer' in state_dict:
        del state_dict['optimizer']

    model = MTDNNModel(config, devices=[device_id])
    if setting is LingualSetting.BASE:
        return model

    # scoring_list classification head doesn't matter because we're just taking
    # the model probe outputs.
    if 'scoring_list.0.weight' in state_dict['state']:
        state_dict['state']['scoring_list.0.weight'] = model.network.state_dict()['scoring_list.0.weight']
        state_dict['state']['scoring_list.0.bias'] = model.network.state_dict()['scoring_list.0.bias']

    model.load_state_dict(state_dict)
    return model

def get_acc(model, test_data, metric_meta, task_type, device_id, label_mapper, model_probe):
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
            model_probe=model_probe,
            label_mapper=label_mapper
        )
    metrics = results[0]
    metric_name = metric_meta[0].name
    return metrics[metric_name]

def evaluate_model_probe(
    downstream_task: Experiment,
    finetuned_task: Union[Experiment, None],
    finetuned_setting: LingualSetting,
    probe_setting: LingualSetting,
    model_ckpt: str,
    metric: str,
    batch_size: int=8,
    max_seq_len: int=512,
    device_id: int=0,
    lang: str='multi'):

    """
    Evaluate model probe for a model finetuned on finetuned_task on a downstream_task.
    """
    task_def_path = Path('experiments').joinpath(
        downstream_task.name,
        'task_def.yaml'
    )
    task_def = TaskDefs(task_def_path).get_task_def(downstream_task.name.lower())
    if task_def.metric_meta[0] is Metric.SeqEvalList:
        sequence = True
    else:
        sequence = False
    
    data_path = Path('experiments').joinpath(
        downstream_task.name,
        lang,
        'bert-base-multilingual-cased',
        f'{downstream_task.name.lower()}_test.json'
    )
    print(f'data from {data_path}')

    test_data = build_dataset(
        data_path,
        EncoderModelType.BERT,
        batch_size,
        max_seq_len,
        task_def)

    model = construct_model(
        finetuned_task,
        finetuned_setting,
        device_id)
    
    if finetuned_setting is not LingualSetting.BASE:
        print(f'\n{finetuned_task.name}_{finetuned_setting.name.lower()} model probed on {downstream_task.name} [{lang}], model probe setting: {probe_setting.name.lower()}')
    else:
        print(f'\nmBERT -> {downstream_task.name} [{lang}], probe setting: {probe_setting.name.lower()}')
    
    # load state dict for the attention head
    if model_ckpt is None: 
        if finetuned_setting is not LingualSetting.BASE:
            state_dict_for_head = Path('checkpoint').joinpath(
                f'{finetuned_task.name}_{finetuned_setting.name.lower()}:{downstream_task.name}'
            )
        else:
            state_dict_for_head = Path('checkpoint').joinpath(f'mBERT:{downstream_task.name}')
        state_dict_for_head = list(state_dict_for_head.rglob("*.pt"))[0]
    else:
        state_dict_for_head = Path(model_ckpt)

    print(f'loading from {state_dict_for_head}')
    state_dict_for_head = torch.load(state_dict_for_head, map_location=f'cuda:{device_id}')['state']

    # then attach the probing layer
    model.attach_model_probe(task_def.n_class, sequence=sequence)

    # get the layer and check
    layer = model.network.get_pooler_layer()
    assert hasattr(layer, 'model_probe_head')

    # and load (put it on same device)
    weight = state_dict_for_head[f'bert.pooler.model_probe_head.weight']
    bias = state_dict_for_head[f'bert.pooler.model_probe_head.bias']

    layer.model_probe_head.weight = nn.Parameter(weight)
    layer.model_probe_head.bias = nn.Parameter(bias)

    # compute acc and save
    acc = get_acc(
        model,
        test_data,
        task_def.metric_meta,
        task_def.task_type,
        device_id,
        task_def.label_vocab.ind2tok,
        model_probe=True)
        
    return acc

def combine_all_model_probe_scores(mean=True, std=True):
    combined_results = []
    combined_std = []

    for task in list(Experiment):
        for setting in [LingualSetting.CROSS, LingualSetting.MULTI]:  
            result_for_task = f'model_probe_outputs/{task.name}_{setting.name.lower()}/evaluation_results.csv'
            result_for_task = pd.read_csv(result_for_task, index_col=0)
            model_name = f'{task.name}_{setting.name.lower()}'

            if mean:
                result_for_task_mean_across_seeds = pd.DataFrame(result_for_task.mean(axis=0)).T
                result_for_task_mean_across_seeds.index = [model_name]
                combined_results.append(result_for_task_mean_across_seeds)
            else:
                combined_results.append(result_for_task)
            
            if std:
                result_for_task_std_across_seeds = pd.DataFrame(result_for_task.std(axis=0)).T
                result_for_task_std_across_seeds.index = [model_name]
                combined_std.append(result_for_task_std_across_seeds)

    for i, df in enumerate([combined_results, combined_std]):
        if len(df) > 0:
            combined_df = pd.concat(df, axis=0)
            if i == 0:
                if mean:
                    out_file_name = 'model_probe_outputs/final_result.csv'
                else:
                    out_file_name = 'model_probe_outputs/final_result_mean.csv'
            else:
                if std:
                    out_file_name = 'model_probe_outputs/final_result_std.csv'
            
            combined_df.to_csv(out_file_name)
            create_heatmap(
                data_df=combined_df,
                row_labels=list(combined_df.index),
                column_labels=list(combined_df.columns),
                xaxlabel='task',
                yaxlabel='model',
                out_file=Path(out_file_name).with_suffix('')
            )

def get_model_probe_final_score(
    finetuned_task: Experiment,
    finetuned_setting: LingualSetting):

    final_results_out_file = Path(f'model_probe_outputs').joinpath(
        f'{finetuned_task.name}_{finetuned_setting.name.lower()}',
        'evaluation_results.csv')

    result_path_for_finetuned_model = final_results_out_file.parent.joinpath('results.csv')
    
    result_path_for_mBERT = Path(f'model_probe_outputs').joinpath(
            f'mBERT',
            'results.csv')
    
    finetuned_results = pd.read_csv(result_path_for_finetuned_model, index_col=0)
    mBERT_results = pd.read_csv(result_path_for_mBERT, index_col=0)

    final_results = pd.DataFrame(finetuned_results.values - mBERT_results.values)
    final_results.index = finetuned_results.index
    final_results.columns = finetuned_results.columns
    final_results.to_csv(final_results_out_file)

def get_model_probe_scores(
    finetuned_task: Experiment,
    finetuned_setting: LingualSetting,
    probe_setting: LingualSetting,
    probe_task: Experiment,
    model_ckpt: str,
    out_file_name: str,
    metric: str,
    device_id: int,
    lang: str,
    batch_size: int = 8,
    max_seq_len: int = 512):
    
    if finetuned_setting is LingualSetting.BASE:
        model_name = 'mBERT'
    else:
        model_name = f'{finetuned_task.name}_{finetuned_setting.name.lower()}'

    results_out_file = Path(f'model_probe_outputs').joinpath(
        model_name,
        f'{out_file_name}.csv')

    if results_out_file.is_file():
        print(f'{results_out_file} already exists.')
        return
    else:
        print(results_out_file.parent)
        results_out_file.parent.mkdir(parents=True, exist_ok=True)
    
    tasks = [probe_task]
    results = pd.DataFrame(np.zeros((1, len(tasks))))
    results.index = [model_name]
    results.columns = [task.name for task in tasks]
    
    for downstream_task in tasks:
        acc = evaluate_model_probe(
            downstream_task,
            finetuned_task,
            finetuned_setting,
            probe_setting,
            model_ckpt,
            metric,
            batch_size,
            max_seq_len,
            device_id,
            lang)

        if finetuned_setting is LingualSetting.BASE:
            results.loc[f'mBERT', downstream_task.name] = acc
        else:
            results.loc[f'{finetuned_task.name}_{finetuned_setting.name.lower()}', downstream_task.name] = acc
        
    results.to_csv(results_out_file)

def create_perlang_results(target_task, langs):
    def get_data(root, version=0):
        data = pd.DataFrame(np.zeros((1, len(langs))))
        data.columns = langs

        for results_file in root.rglob("*.csv"):
            name_ = results_file.with_suffix('').name
            if version == 0:
                try:
                    task = name_.split("-")[1]
                except:
                    raise ValueError(name_)
                lang = name_.split("-")[-1].split('.')[0]
            else:
                task = name_.split("_")[0]
                lang = name_.split("_")[1]
            if task == target_task:
                results = pd.read_csv(results_file, index_col=0)
                data.loc[:, lang] = results.iloc[0,0]
        
        return data

    root = Path('model_probe_outputs/NLI_multi')
    model_data = get_data(root, 0)

    base = root.parent.joinpath('mBERT')
    base_data = get_data(base, 1)

    data = model_data - base_data
    data.index = ['XNLI-4lang']

    output_path = root.parent.joinpath(f'results/{target_task}-results.pdf')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path.with_suffix(".csv"))

def combine_and_heatmap(tasks, index=None):
    data = []
    for task in tasks:
        data_for_task = pd.read_csv(f'model_probe_outputs/results/{task}-results.csv', index_col=0)
        data.append(data_for_task)
    
    data = pd.concat(data, axis=0)
    data.index = index

    font_size = 30
    plt.figure(figsize=(14, 14))
    annot_kws = {'fontsize': font_size}
    ax = sns.heatmap(
        data,
        cbar=False,
        annot=True,
        annot_kws=annot_kws,
        fmt=".2f",
        square=True)

    ax.tick_params(axis='x', labelsize=font_size)
    ax.tick_params(axis='y', labelsize=font_size, labelrotation=0)

    fig = ax.get_figure()
    fig.savefig(f'model_probe_outputs/results/combined_result.pdf', bbox_inches='tight')

def get_scores_main(args):
    if args.model_ckpt == '':
        args.model_ckpt = None
    
    if args.probe_task == '':
        tasks = ['MARC', 'POS', 'NER', 'NLI', 'PAWSX']
    else:
        tasks = [args.probe_task.upper()]
    
    for task in tasks:
        if task == 'NLI':
            langs = [
                'ar',
                'bg',
                'de',
                'el',
                'en',
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
                'multi'
            ]
        else:
            langs = ['en', 'fr', 'de', 'es', 'multi']
        
        for lang in langs:            
            get_model_probe_scores(
                Experiment[args.finetuned_task.upper()],
                LingualSetting[args.finetuned_setting.upper()],
                LingualSetting[args.probe_setting.upper()],
                Experiment[task.upper()],
                args.model_ckpt,
                f'{task}_{lang}',
                args.metric,
                args.device_id,
                lang,
                args.batch_size,
                args.max_seq_len
            )

def create_perlang_heatmap(args):
    tasks = ['NLI', 'POS', 'NER', 'PAWSX', 'MARC']
    for task in tasks:
        langs = ['foreign', 'en']
        create_perlang_results(task, langs)

    combine_and_heatmap(['NLI', 'POS', 'NER', 'PAWSX', 'MARC'], ['XNLI', 'POS', 'NER', 'PI', 'SA'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--finetuned_task', type=str, default='')
    parser.add_argument('--finetuned_setting', type=str, default='base')

    parser.add_argument('--probe_setting', type=str, default='cross')
    parser.add_argument('--probe_task', type=str, default='')

    parser.add_argument('--model_ckpt', type=str, default='', help='checkpoint of model probe head')
    parser.add_argument('--metric', type=str, default='F1MAC')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_seq_len', type=int, default=512)
    args = parser.parse_args()

    get_scores_main(args)
    create_perlang_heatmap(args)