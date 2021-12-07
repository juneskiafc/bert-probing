from typing import List
import subprocess
import itertools
from argparse import ArgumentParser
from pathlib import Path
import torch
from experiments.exp_def import (
    Experiment,
    LingualSetting,
    TaskDefs
)

def probe_heads(setting: LingualSetting,
                finetuned_task: Experiment,
                task: Experiment,
                models_per_gpu: int = 2,
                devices: List = list(range(torch.cuda.device_count()))):
    """
    Probe heads for a model.

    Args:
    setting: cross, multi, or base, which finetuned model to probe. If base, just pretrained BERT.
    finetuned_task: the task that the model was finetuned on.
    task: the task to probe heads on.
    models_per_gpu: how many models should each gpu process?
    devices: devices to use.
    """
    # where all the data and task_def are stored.
    task_root = Path(f'experiments/{task.name}')

    # programmatically get n_classes for task
    task_def_path = task_root.joinpath('task_def.yaml')
    task_def = TaskDefs(task_def_path).get_task_def(task.name.lower())
    n_classes = task_def.n_class
    checkpoint_dir = Path(f'checkpoint/head_probing/{finetuned_task.name}').joinpath(task.name) # where the probed checkpoints will be

    # only probe heads that we haven't already probed.
    heads_to_probe = []
    for hl, hi in itertools.product(range(12), repeat=2):
        dir_for_head = checkpoint_dir.joinpath(setting.name.lower(), str(hl), str(hi))
        if len(list(dir_for_head.rglob('model_1_*.pt'))) == 0:
            heads_to_probe.append((hl, hi))
    
    # distribute heads to probe to different gpus.
    device_ids = []
    for i, _ in enumerate(heads_to_probe):
        device_ids.append(devices[i % len(devices)])
    
    print('heads to probe:')
    for i, hp in enumerate(heads_to_probe):
        print(hp, f'GPU: {device_ids[i]}')
    print("\n")

    # Run commands in parallel
    processes = []
    for i, (hl, hi) in enumerate(heads_to_probe):
        did = device_ids[i]
        checkpoint_dir_for_head = checkpoint_dir.joinpath(setting.name.lower(), str(hl), str(hi))

        template = f'python train.py --local_rank -1 '
        template += f'--dataset_name {task.name}/cross ' # always train head probes using cross-ling setting
        
        if setting is not LingualSetting.BASE:
            finetuned_checkpoint_dir = Path(f'checkpoint/{finetuned_task.name}_{setting.name.lower()}')
            finetuned_checkpoint = list(finetuned_checkpoint_dir.rglob('model_5*.pt'))[0]
            template += f"--resume --model_ckpt {finetuned_checkpoint} "
        
        template += f"--epochs 2 --output_dir {checkpoint_dir_for_head} "
        template += f"--init_checkpoint bert-base-multilingual-cased --devices {did} "
        template += f'--head_probe --head_probe_layer {hl} --head_probe_idx {hi} --head_probe_n_classes {n_classes}'
    
        process = subprocess.Popen(template, shell=True, stdout=None)
        processes.append(process)

        # wait if filled
        if len(processes) == len(devices) * models_per_gpu:
            _ = [p.wait() for p in processes]
            processes = []

    # Collect statuses
    _ = [p.wait() for p in processes]

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--task', type=str)
    parser.add_argument('--finetuned_task', type=str)
    parser.add_argument('--devices', nargs='+')
    parser.add_argument('--models_per_gpu', type=int, default=1)
    args = parser.parse_args()

    task = Experiment[args.task.upper()]
    finetuned_task = Experiment[args.finetuned_task.upper()]
    devices = [int(d) for d in args.devices]

    for setting in list(LingualSetting):
        probe_heads(setting=setting,            
                    finetuned_task=finetuned_task,
                    task=task,
                    devices=args.devices)
