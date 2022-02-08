from pathlib import Path
import subprocess

finetuned_task = 'NLI'
finetuned_setting = 'multi'
root = Path('checkpoint')
procs = []

for downstream_task in ['MARC', 'NER', 'NLI', 'PAWSX', 'POS']:
    for seed in range(3):
        if downstream_task != 'NLI' and seed == 0:
            continue
        
        ckpt = root.joinpath(f'{finetuned_task}_{finetuned_setting}-{downstream_task}-model_probe_seed{seed}')
        if ckpt.is_dir():
            cmd = f'scp -r {ckpt} june@129.170.213.92:/data_big/june/mt-dnn/checkpoint/full_model_probe/cross_head_training/{finetuned_task}/{finetuned_setting}/{downstream_task}'
            process = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE)
            process.communicate(input=b'june\n')
            procs.append(process)
            print(f'{finetuned_task}_{finetuned_setting}-{downstream_task}, seed {seed}')

for p in procs:
    p.wait()


