import subprocess

for task in ['MARC', 'NER', 'NLI', 'PAWSX', 'POS']:
    for setting in ['cross', 'multi']:
        print(f"{task}_{setting}")
        command = f'python train.py --devices 3 --exp_name {task}_{setting}_gp --gradient_probe --dataset_name {task}/{setting}'
        subprocess.Popen(command, shell=True)