from pathlib import Path
import torch
import subprocess
from evaluate import evaluate_model_against_multiple_datasets
from experiments.exp_def import Experiment, TaskDefs

for task in ['MARC', 'NLI', 'POS', 'NER', 'PAWSX']:
    cross_checkpoint_dir = Path(f'checkpoint/{task}_cross')
    multi_checkpoint_dir = Path(f'checkpoint/{task}_multi')
    if cross_checkpoint_dir.is_dir() and multi_checkpoint_dir.is_dir():
        print(f'transplanting {task}')
        base_model_ckpt = list(cross_checkpoint_dir.rglob("*.pt"))[0]
        multilingual_model_ckpt = list(multi_checkpoint_dir.rglob("*.pt"))[0]

        base_model = torch.load(base_model_ckpt)
        multilingual_model = torch.load(multilingual_model_ckpt)

        base_model['state']['scoring_list.0.weight'] = multilingual_model['state']['scoring_list.0.weight']
        base_model['state']['scoring_list.0.bias'] = multilingual_model['state']['scoring_list.0.bias']

        task_def_path = f'experiments/{task}/task_def.yaml'
        task_def = TaskDefs(task_def_path).get_task_def(task)
        metric_meta = task_def.metric_meta

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

        evaluate_model_against_multiple_datasets(
            base_model,
            'BERT',
            Experiment[task],
            metric_meta,
            datasets,
            task_def_path,
            device_id=0
        )
        