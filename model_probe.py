from pathlib import Path
import argparse
import subprocess
from experiments.exp_def import (
    Experiment,
    TaskDefs,
)

parser = argparse.ArgumentParser()
parser.add_argument('--finetuned_task', type=str, default='')
parser.add_argument('--finetuned_setting', type=str, default='')
parser.add_argument('--task', type=str, default='')
parser.add_argument('--devices', nargs='+')
args = parser.parse_args()

model_ckpt_dir = Path('checkpoint').joinpath(f'{args.finetuned_task.upper()}_{args.finetuned_setting}')
model_ckpt = list(model_ckpt_dir.rglob("model_5*.pt"))[0]
devices = [int(d) for d in args.devices]

processes = []
if args.task == '':
    tasks = list(Experiment)
else:
    tasks = [Experiment[args.task.upper()]]

for downstream_task in tasks:
    exp_name = f'model_probing/{args.finetuned_task}/{args.finetuned_setting}/{downstream_task.name}'
    dataset = f'{downstream_task.name}/cross'

    task_def_path = Path('experiments').joinpath(downstream_task.name, 'task_def.yaml')
    task_def = TaskDefs(task_def_path).get_task_def(downstream_task.name.lower())

    cmd = 'python train.py'
    cmd += f' --devices {devices[len(processes)]}'
    cmd += f' --model_probe --model_probe_n_classes {task_def.n_class}'
    cmd += f' --exp_name {exp_name}'
    cmd += f' --dataset_name {dataset}'
    cmd += ' --epochs 2'
    cmd += f' --model_ckpt {model_ckpt} --resume'

    if downstream_task in [Experiment.NER, Experiment.POS]:
        cmd += ' --model_probe_sequence'

    process = subprocess.run(cmd.split(' '))


