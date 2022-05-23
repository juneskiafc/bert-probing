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
from mt_dnn.model import MTDNNModel

from data_utils.task_def import EncoderModelType
from data_utils.metrics import Metric
from experiments.exp_def import (
    Experiment,
    LingualSetting,
    TaskDefs,
)
from utils import create_heatmap, build_dataset, get_metric, base_construct_model
import pickle


def construct_model(task: Experiment, setting: LingualSetting, device_id: int):
    if task is Experiment.BERT:
        with open('checkpoint/dummy_config.pkl', 'rb') as fr:
            config = pickle.load(fr)
            model = MTDNNModel(config, devices=[device_id])
            return model
    
    checkpoint = list(Path(f'checkpoint/{task.name}_{setting.name.lower()}').rglob('*.pt'))[0]
    task_def_path = f'experiments/{task.name}/task_def.yaml'
    config, state_dict, metric_meta = base_construct_model(checkpoint, task, task_def_path, device_id)

    if state_dict is not None and 'optimizer' in state_dict:
        del state_dict['optimizer']

    model = MTDNNModel(config, devices=[device_id])

    # scoring_list classification head doesn't matter because we're just taking
    # the model probe outputs.
    if 'scoring_list.0.weight' in state_dict['state']:
        state_dict['state']['scoring_list.0.weight'] = model.network.state_dict()['scoring_list.0.weight']
        state_dict['state']['scoring_list.0.bias'] = model.network.state_dict()['scoring_list.0.bias']

    model.load_state_dict(state_dict)
    return model

def evaluate_model_probe(
    downstream_task: Experiment,
    finetuned_task: Union[Experiment, None],
    finetuned_setting: LingualSetting,
    probe_setting: LingualSetting,
    model_ckpt: str,
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
        print(f'\nBERT -> {downstream_task.name} [{lang}], probe setting: {probe_setting.name.lower()}')
    
    # load state dict for the attention head
    if model_ckpt is None: 
        if finetuned_setting is not LingualSetting.BASE:
            state_dict_for_head = Path('checkpoint').joinpath(
                'model_probing',
                finetuned_task.name,
                finetuned_setting.name.lower(),
                downstream_task.name,
            )
        else:
            state_dict_for_head = Path('checkpoint').joinpath(
                'model_probing',
                finetuned_task.name,
                downstream_task.name,
            )
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
    acc = get_metric(
        model,
        test_data,
        task_def.metric_meta,
        task_def.task_type,
        device_id,
        task_def.label_vocab.ind2tok,
        model_probe=True)[0]
        
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
    
    result_path_for_BERT = Path(f'model_probe_outputs').joinpath(
            f'BERT',
            'results.csv')
    
    finetuned_results = pd.read_csv(result_path_for_finetuned_model, index_col=0)
    BERT_results = pd.read_csv(result_path_for_BERT, index_col=0)

    final_results = pd.DataFrame(finetuned_results.values - BERT_results.values)
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
    device_id: int,
    lang: str,
    batch_size: int = 8,
    max_seq_len: int = 512):
    
    if finetuned_setting is LingualSetting.BASE:
        model_name = 'BERT'
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
    results = np.zeros((1, len(tasks)))
    
    for i, downstream_task in enumerate(tasks):
        acc = evaluate_model_probe(
            downstream_task,
            finetuned_task,
            finetuned_setting,
            probe_setting,
            model_ckpt,
            batch_size,
            max_seq_len,
            device_id,
            lang)

        results[0, i] = acc
    
    results = pd.DataFrame(results)
    results.index = [model_name]
    results.columns = [task.name for task in tasks]

    results.to_csv(results_out_file)

def create_perlang_results(finetuned_task, finetuned_setting, downstream_task, langs):
    def get_data(root):
        data = pd.DataFrame(np.zeros((1, len(langs))))
        data.columns = langs

        for lang in langs:
            results_file = root.joinpath(f'{downstream_task}_{lang}.csv')
            results = pd.read_csv(results_file, index_col=0)
            data.loc[0, lang] = results.iloc[0, 0]
        
        return data
    
    if finetuned_task is Experiment.BERT:
        root = Path(f'model_probe_outputs/{finetuned_task}')
    else:
        root = Path(f'model_probe_outputs/{finetuned_task}_{finetuned_setting}')
    
    output_path = root.parent.joinpath(f'results/{finetuned_task}_{finetuned_setting.lower()}-{downstream_task}.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.is_file():
        data = pd.read_csv(output_path, index_col=0)
        return data
    
    model_data = get_data(root)

    base = root.parent.joinpath('BERT')
    base_data = get_data(base)

    data = model_data - base_data
    data.index = [f'{finetuned_task}_{finetuned_setting.lower()}']

    data.to_csv(output_path)
    return data

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
                args.device_id,
                lang,
                args.batch_size,
                args.max_seq_len
            )

def create_perlang_heatmap(args):
    # once for en, fr, de, es
    out_file = Path(f'model_probe_outputs/results/{args.finetuned_task}-{args.finetuned_setting}.csv')
    out_file.parent.mkdir(parents=True, exist_ok=True)

    exp_name_map = {
        'NLI': 'XNLI',
        'PAWSX': 'PI',
        'MARC': 'SA'
    }

    if not out_file.is_file():
        if args.probe_task == '':
            tasks = ['NLI', 'POS', 'NER', 'PAWSX', 'MARC']
        else:
            tasks = [args.probe_task.upper()]
        
        indicies = [exp_name_map[t] if t in exp_name_map else t for t in tasks]
        langs = ['en', 'fr', 'de', 'es', 'multi']
        data = np.zeros((len(tasks), len(langs)))

        for i, task in enumerate(tasks):
            data_for_task = create_perlang_results(
                args.finetuned_task,
                args.finetuned_setting,
                task,
                langs
            )
            data[i, :] = data_for_task.values

        # data = pd.DataFrame(data)
        # data.index = indicies
        # data.columns = langs
        # data.to_csv(out_file)
    
    else:
        data = pd.read_csv(out_file, index_col=0)
    
    # create_heatmap(
    #     data_df=data,
    #     row_labels=data.index,
    #     column_labels=data.columns,
    #     xaxlabel='language',
    #     yaxlabel='task',
    #     fontsize=20,
    #     figsize=(10, 10),
    #     out_file=out_file.with_suffix('.pdf'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--finetuned_task', type=str, default='bert')
    parser.add_argument('--finetuned_setting', type=str, default='base')

    parser.add_argument('--probe_setting', type=str, default='cross')
    parser.add_argument('--probe_task', type=str, default='')

    parser.add_argument('--model_ckpt', type=str, default='', help='checkpoint of model probe head')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_seq_len', type=int, default=512)
    args = parser.parse_args()

    get_scores_main(args)
    create_perlang_heatmap(args)