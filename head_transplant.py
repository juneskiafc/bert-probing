from pathlib import Path
import torch
from evaluate import evaluate_model_against_multiple_datasets
from experiments.exp_def import Experiment, TaskDefs
from argparse import ArgumentParser
from utils import base_construct_model

from mt_dnn.model import MTDNNModel
import pandas as pd

def transplant_eval(task, device_id):
    cross_checkpoint_dir = Path(f'checkpoint/{task.upper()}_cross')
    multi_checkpoint_dir = Path(f'checkpoint/{task.upper()}_multi')

    if not cross_checkpoint_dir.is_dir():
        raise ValueError(f'make sure you have checkpoint/{task.upper()}_cross')
    
    if not multi_checkpoint_dir.is_dir():
        raise ValueError(f'make sure you have checkpoint/{task.upper()}_multi')

    print(f'transplant evaluation [{task}]')
    base_model_ckpt = list(cross_checkpoint_dir.rglob("*.pt"))[0]
    multilingual_model_ckpt = list(multi_checkpoint_dir.rglob("*.pt"))[0]

    base_model = torch.load(base_model_ckpt)
    multilingual_model = torch.load(multilingual_model_ckpt)

    base_model['state']['scoring_list.0.weight'] = multilingual_model['state']['scoring_list.0.weight']
    base_model['state']['scoring_list.0.bias'] = multilingual_model['state']['scoring_list.0.bias']

    task_def_path = f'experiments/{task}/task_def.yaml'
    config, _, metric_meta = base_construct_model('', Experiment[task.upper()], task_def_path, device_id)
    model = MTDNNModel(config, devices=[device_id], state_dict=base_model)

    if task == 'NLI':
        datasets = [
            # 'ar',
            # 'bg',
            # 'el',
            # 'hi',
            # 'ru',
            # 'sw',
            # 'th',
            # 'tr',
            # 'ur',
            # 'vi',
            # 'zh',
            'en',
            'es',
            'fr', 
            'de',
            'multi'
        ]
    else:
        datasets = ['en', 'fr', 'de', 'es', 'multi']

    metric = evaluate_model_against_multiple_datasets(
        model,
        'BERT',
        Experiment[task],
        metric_meta,
        datasets,
        task_def_path,
        device_id=device_id
    )

    results_out_file = Path(f'evaluation_results/transplant/{task}_transplant.csv')
    results_out_file.parent.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(metric,index=datasets)
    results.to_csv(results_out_file)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--task', type=str)
    parser.add_argument('--device_id', type=int, default=0)
    args = parser.parse_args()

    transplant_eval(args.task, args.device_id)