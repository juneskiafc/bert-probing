import subprocess
from pathlib import Path
import torch

def probe_heads(setting, task, base=False, models_per_gpu=2):
    if setting == 'cross':
        shorthand_setting = 'cl'
    elif setting == 'multi':
        shorthand_setting = 'ml'    
    
    if base:
        shorthand_setting = 'base'

    task_root = Path(f'experiments/{task}')
    if task == 'NLI':
        task_root = task_root.joinpath(setting)
        train_dataset = setting
        n_classes = 3
        checkpoint_task = 'NLI'
    elif task == 'POS':
        train_dataset = 'pos'
        checkpoint_task = 'POS'
        n_classes = 17
    elif task == 'NER':
        checkpoint_task = 'NER'
        n_classes = 7
        train_dataset = 'ner'
    elif task == 'MARC':
        checkpoint_task = 'MARC'
        n_classes = 5
        train_dataset = 'marc'

    task_def = task_root.joinpath('task_def.yaml')
    data_dir = task_root.joinpath('bert-base-multilingual-cased')

    checkpoint_dir = Path(f'checkpoint/head_probing').joinpath(checkpoint_task)

    heads_to_probe = []
    for hl in range(12):
        for hi in range(12):
            dir_for_head = checkpoint_dir.joinpath(shorthand_setting, f'{shorthand_setting}_{hl}/{shorthand_setting}_{hl}_{hi}')
            if len(list(dir_for_head.rglob('*.pt'))) > 0:
                continue
            else:
                heads_to_probe.append((hl, hi))

    device_ids = []
    available_devices = [0, 1, 2, 3]
    for i, _ in enumerate(heads_to_probe):
        device_ids.append(available_devices[i % len(available_devices)])

    # Run commands in parallel
    processes = []
    template = f'python train.py --task_def {task_def} '
    template += f"--data_dir {data_dir} --train_datasets {train_dataset} --local_rank -1 "
    if not base: 
        template += f"--resume --model_ckpt checkpoint/{setting}/model_5.pt "

    for i, (hl, hi) in enumerate(heads_to_probe):
        did = device_ids[i]
        exp_name = f'{shorthand_setting}_{hl}_{hi}'
        checkpoint_dir_for_head = checkpoint_dir.joinpath(f'{shorthand_setting}_{hl}')
        template += f"--head_probe_n_classes {n_classes} --epochs 2 --output_dir {checkpoint_dir_for_head} "
        template += f"--init_checkpoint bert-base-multilingual-cased --device_id {did} "
        template += f"--exp_name {exp_name} --head_probe --head_probe_layer {hl} --head_probe_idx {hi} "
        process = subprocess.Popen(template, shell=True, stdout=None)
        processes.append(process)

        if len(processes) == len(available_devices) * models_per_gpu:
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
    setting = 'cross'
    task = 'MARC'
    probe_heads(setting, task, base=False)
    # compress_saved_heads(task)