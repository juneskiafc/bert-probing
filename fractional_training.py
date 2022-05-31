from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import subprocess

def evaluate_trained_models():
    for seed in range(3):
        for frac in [0.2, 0.4, 0.6, 0.8]:
            for model in ['NER']:
                if not Path(f'evaluation_results/{model}-{seed}-{frac}.csv').is_file():
                    ckpt_dir = Path(f'checkpoint/{model}_{frac}_{seed}')
                    if ckpt_dir.is_dir():
                        model_ckpt = list(ckpt_dir.rglob("*.pt"))[0]
                        command = 'python evaluate.py --device_id 0 '
                        command += f'--model_ckpt {model_ckpt} '
                        command += f'--out_file {model}-{seed}-{frac} --task {model}'
                        subprocess.call(command, shell=True, stdout=subprocess.PIPE)
                
def make_graph():
    plt.rcParams.update({'font.size': 14})
    root = Path('evaluation_results')
    for us_task in ['NER']:
        ax = plt.subplot(111)
        colors = ['black', 'green', 'blue']
        cross_result = pd.read_csv(root.joinpath(f'{us_task}_cross.csv'), index_col=0).iloc[-1, 0]
        multi_result = pd.read_csv(root.joinpath(f'{us_task}_multi.csv'), index_col=0).iloc[-1, 0]

        seeds = [500, 900, 1300]
        for seed in range(3):
            y = []
            y.append(cross_result)

            x = [0, 0.2, 0.4, 0.6, 0.8, 1]
            for fraction in [0.2, 0.4, 0.6, 0.8]:
                result = root.joinpath(f'{us_task}-{seed}-{fraction}.csv')
                result = pd.read_csv(result, index_col=0)
                y.append(result.iloc[0, 0])
            
            y.append(multi_result)

            ax.plot(x, y, c=colors[seed], label=f'seed: {seeds[seed]}')

        if us_task != 'NLI':
            ax.set_ylabel('Macro average F1')
        else:
            ax.set_ylabel('Accuracy')
        ax.set_xlabel('Fraction of target language training data')
        ax.legend()

        plt.savefig(root.joinpath(f'{us_task}.pdf'), bbox_inches='tight')
        plt.clf()
                

if __name__ == '__main__':
    evaluate_trained_models()
    make_graph()


        

