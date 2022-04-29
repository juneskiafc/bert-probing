from typing import List

import argparse
from pathlib import Path

from experiments.exp_def import (
    TaskDefs,
    Experiment
)
from data_utils.task_def import EncoderModelType
from mt_dnn.model import MTDNNModel
import pandas as pd
import argparse
from utils import get_metric, create_heatmap, build_dataset, base_construct_model

def construct_model(checkpoint: str, task: Experiment, task_def_path: str, device_id: int, task_id: int = 0):
    config, state_dict, metric_meta = base_construct_model(checkpoint, task, task_def_path, device_id)

    if state_dict is not None:
        del state_dict['optimizer']

        # we load the task we want to evaluate on the first scoring list, then delete the rest
        if task_id != 0:
            state_dict['state']['scoring_list.0.weight'] = state_dict['state'][f'scoring_list.{task_id}.weight']
            state_dict['state']['scoring_list.0.bias'] = state_dict['state'][f'scoring_list.{task_id}.bias']

        i = 1
        while f'scoring_list.{i}.weight' in state_dict['state']:
            del state_dict['state'][f'scoring_list.{i}.weight']
            del state_dict['state'][f'scoring_list.{i}.bias']
            i += 1

    model = MTDNNModel(config, devices=[device_id], state_dict=state_dict)
    return model, metric_meta

def evaluate_model_against_multiple_datasets(
    model: MTDNNModel,
    model_type: str,
    task: Experiment,
    metric_meta,
    datasets: List[str],
    task_def_path: str,
    device_id: int):

    metrics = []
    
    for dataset in datasets:
        print(f'Evaluating on {dataset}')
        if model_type == 'bert':
            model_full_name = 'bert-base-multilingual-cased'
        else:
            model_full_name = 'xlm-roberta-base'
        
        data_path = f'experiments/{task.name}/{dataset}/{model_full_name}/{task.name.lower()}_test.json'
        task_def = TaskDefs(task_def_path).get_task_def(task.name.lower())

        test_data = build_dataset(
            data_path,
            task_def=task_def,
            device_id=device_id,
            batch_size=8,
            max_seq_len=512)

        metric = get_metric(
            model,
            test_data,
            metric_meta,
            task_def.task_type,
            device_id,
            label_mapper=task_def.label_vocab.ind2tok)[0]
        
        metrics.append(metric)
    
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--model_ckpt', type=str, default='')
    parser.add_argument('--out_file', type=str, default='')
    parser.add_argument('--task', type=str)
    parser.add_argument('--model_type', type=str, default='bert')
    parser.add_argument('--task_id', type=int, default=0)
    args = parser.parse_args()

    task = Experiment[args.task.upper()]

    if args.out_file == '':
        results_out_file = Path(f'evaluation_results/{task.name}.csv')
    else:
        results_out_file = Path(f'evaluation_results/{args.out_file}.csv')
    results_out_file.parent.mkdir(parents=True, exist_ok=True)

    if task is Experiment.NLI:
        datasets = [
            # 'ar',
            # 'bg',
            # 'de',
            # 'el',
            # 'es',
            # 'fr',
            # 'hi',
            # 'ru',
            # 'sw',
            # 'th',
            # 'tr',
            # 'ur',
            # 'vi',
            # 'zh',
            # 'en',
            # 'combined'
            'multi'
        ]
    else:
        datasets = ['en', 'fr', 'de', 'es', 'multi']
        
    if not results_out_file.is_file():
        root_ckpt_path = Path('checkpoint/')
        encoder_type = EncoderModelType[args.model_type.upper()]
        
        task_def_path = f'experiments/{task.name}/task_def.yaml'

        model, metric_meta = construct_model(
            args.model_ckpt,
            task,
            task_def_path,
            args.device_id,
            args.task_id)
        
        accs = evaluate_model_against_multiple_datasets(
            model,
            args.model_type,
            task,
            metric_meta,
            datasets,
            task_def_path,
            device_id=args.device_id
        )

        results = pd.DataFrame(accs,index=datasets)
        results.to_csv(results_out_file)
    
    else:
        results = pd.read_csv(results_out_file, index_col=0)

    create_heatmap(
        data_df=results,
        row_labels=[d.upper() for d in datasets],
        column_labels=[task.name],
        xaxlabel='',
        yaxlabel='languages',
        out_file=results_out_file.with_suffix('.pdf'),
        figsize=(5, 14)
    )
