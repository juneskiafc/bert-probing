from pathlib import Path
import subprocess
from experiments.exp_def import (
    Experiment,
    LingualSetting,
)

model = 'NLI'
task_to_n_classes = {
    'NLI': 3,
    'POS': 17,
    'PAWSX': 2,
    'MARC': 5,
    'NER': 7
}
seeds = 1
devices = [6, 7]

processes = []
for setting in [LingualSetting.CROSS]:
    for downstream_task in list(Experiment):
        # for seed in range(seeds):
        #     if downstream_task is not Experiment.NLI and seed == 0:
        #         continue
        seed = 0
        exp_name = f'{model}_{setting.name.lower()}-{downstream_task.name}-model_probe_seed{seed}'
        dataset = f'{downstream_task.name}/multi'
        model_ckpt = list(Path(f'checkpoint/{model}_{setting.name.lower()}').rglob('*.pt'))[0]
        cmd = 'python train.py'
        cmd += f' --devices {devices[len(processes)]}'
        cmd += f' --model_probe --model_probe_n_classes {task_to_n_classes[downstream_task.name]}'
        cmd += f' --exp_name {exp_name}'
        cmd += f' --dataset_name {dataset}'
        cmd += ' --epochs 2'
        cmd += f' --model_ckpt {model_ckpt} --resume'
        cmd += f' --seed {seed + 2}018'
        raise ValueError(cmd)
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)

        if len(processes) == 2:
            results = [p.wait() for p in processes]
            processes = []


