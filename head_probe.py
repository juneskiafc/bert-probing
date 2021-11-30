from typing import List
import subprocess
import itertools
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
        if len(list(dir_for_head.rglob('*.pt'))) == 0:
            heads_to_probe.append((hl, hi))

    print('heads to probe:')
    for hp in heads_to_probe:
        print(hp)
    print("\n")
    
    # distribute heads to probe to different gpus.
    device_ids = []
    for i, _ in enumerate(heads_to_probe):
        device_ids.append(devices[i % len(devices)])

    # Run commands in parallel
    processes = []
    for i, (hl, hi) in enumerate(heads_to_probe):
        did = device_ids[i]
        checkpoint_dir_for_head = checkpoint_dir.joinpath(setting.name.lower(), str(hl), str(hi))

        template = f'python train.py --local_rank -1 '
        template += f'--dataset_name {task.name}/cross ' # always train head probes using cross-ling setting
        if setting is not LingualSetting.BASE:
            template += f"--resume --model_ckpt checkpoint/{finetuned_task.name}_{setting.name}/model_5.pt "
        
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

def compress_saved_heads(task):
    ckpt_root = Path(f'checkpoint/head_probing/{task}')
    for (setting, sh_setting) in [('cross', 'cl'), ('multi', 'ml')]:
        for hl in range(12):
            for hi in range(12):
                ckpt_dir = ckpt_root.joinpath(setting, f'{sh_setting}_{hl}', f'{sh_setting}_{hl}_{hi}')
                ckpt_file = list(ckpt_dir.rglob('*.pt'))[0]
                state_dict = torch.load(ckpt_file).state_dict()
                hp_weights = state_dict[f'bert.encoder.layer.{hl}.attention.self.head_probe_dense_layer.weight']
                hp_bias = state_dict[f'bert.encoder.layer.{hl}.attention.self.head_probe_dense_layer.bias']

                hp_state_dict = {
                    f'head_probe_dense_layer.weight': hp_weights,
                    f'.head_probe_dense_layer.bias': hp_bias
                }
                torch.save(hp_state_dict, ckpt_file)

if __name__ == '__main__':
    task = Experiment.POS
    finedtuned_task = Experiment.POS

    for setting in list(LingualSetting):
        probe_heads(setting=setting,            
                    finetuned_task=finedtuned_task,
                    task=task)
