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
processes = []
for setting in [LingualSetting.CROSS, LingualSetting.MULTI]:
    for downstream_task in list(Experiment):
        for seed in range(3):
            if downstream_task is not Experiment.NLI and seed == 0:
                continue

            exp_name = f'{model}_{setting.name.lower()}-{downstream_task.name}-model_probe_seed{seed}'
            dataset = f'{downstream_task.name}/cross'
            model_ckpt = list(Path(f'checkpoint/{model}_{setting.name.lower()}').rglob('*.pt'))[0]
            cmd = 'python train.py'
            cmd += f' --devices {len(processes)}'
            cmd += f' --model_probe --model_probe_n_classes {task_to_n_classes[downstream_task.name]}'
            cmd += f' --exp_name {exp_name}'
            cmd += f' --dataset_name {dataset}'
            cmd += ' --epochs 2'
            cmd += f' --model_ckpt {model_ckpt} --resume'
            cmd += f' --seed {seed + 2}018'

            process = subprocess.Popen(cmd, shell=True)
            processes.append(process)

            if len(processes) == 4:
                results = [p.wait() for p in processes]
                processes = []


