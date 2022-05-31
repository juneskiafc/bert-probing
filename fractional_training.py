from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
from experiments.exp_def import (
    Experiment
)
from argparse import ArgumentParser

def evaluate_trained_models(task):
    models = [Experiment[task.upper()]]

    for seed in range(3):
        for frac in [0.2, 0.4, 0.6, 0.8]:
            for model in models:
                ckpt_dir = Path(f'checkpoint/{model.name}_{frac}_{seed}')
                if not Path(f'evaluation_results/{model.name}-{seed}-{frac}.csv').is_file() and ckpt_dir.is_dir():
                    model_ckpt = list(ckpt_dir.rglob("*.pt"))[0]
                    command = 'python evaluate.py --device_id 0 '
                    command += f'--model_ckpt {model_ckpt} '
                    command += f'--out_file {model.name}-{seed}-{frac} --task {model.name}'
                    subprocess.call(command, shell=True, stdout=subprocess.PIPE)
                
def make_graph(task):
    plt.rcParams.update({'font.size': 14})

    root = Path('evaluation_results')
    us_tasks = [Experiment[task.upper()]]

    for us_task in us_tasks:
        ax = plt.subplot(111)
        colors = ['black', 'green', 'blue']
        cross_result_path = root.joinpath(f'{us_task.name}_cross.csv')
        multi_result_path = root.joinpath(f'{us_task.name}_multi.csv')

        if cross_result_path.is_file():
            cross_result = pd.read_csv(root.joinpath(f'{us_task.name}_cross.csv'), index_col=0).iloc[-1, 0]
        else:
            raise ValueError(f'please evalute {us_task}_cross')

        if multi_result_path.is_file():
            multi_result = pd.read_csv(root.joinpath(f'{us_task.name}_multi.csv'), index_col=0).iloc[-1, 0]
        else:
            raise ValueError(f'please evalute {us_task.name}_multi')

        seeds = [500, 900, 1300]
        for seed in range(3):
            y = []
            y.append(cross_result)

            x = [0, 0.2, 0.4, 0.6, 0.8, 1]
            for fraction in [0.2, 0.4, 0.6, 0.8]:
                result_path = root.joinpath(f'{us_task.name}-{seed}-{fraction}.csv')
                if result_path.is_file():
                    result = pd.read_csv(result_path, index_col=0)
                else:
                    raise ValueError(f'please evaluate {result_path}')
                
                y.append(result.iloc[0, 0])
            
            y.append(multi_result)

            ax.plot(x, y, c=colors[seed], label=f'seed: {seeds[seed]}')

        if us_task.name != 'NLI':
            ax.set_ylabel('Macro average F1')
        else:
            ax.set_ylabel('Accuracy')

        ax.set_xlabel('Fraction of target language training data')
        ax.legend()

        plt.savefig(root.joinpath(f'{us_task.name}_frac_training.pdf'), bbox_inches='tight')
        plt.clf()
                

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--task', default='', type=str)
    args = parser.parse_args()

    evaluate_trained_models(args.task)
    make_graph(args.task)


        

