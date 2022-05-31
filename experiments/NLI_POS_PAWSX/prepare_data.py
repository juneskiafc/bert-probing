import shutil
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from experiments.utils import merge_yaml

ROOT = Path('experiments/NLI_POS_PAWSX')
NLI_TASK_DEF = ROOT.joinpath('task_def_nli.yaml')
POS_TASK_DEF =  ROOT.joinpath('task_def_pos.yaml')
PAWSX_TASK_DEF =  ROOT.joinpath('task_def_pawsx.yaml')
ROOT_TASK_DEF =  ROOT.joinpath('task_def.yaml')

task_defs = []
for task in ['NLI', 'POS', 'PAWSX']:
    shutil.copytree(f'experiments/{task}/cross', ROOT, dirs_exist_ok=True)
    if ROOT_TASK_DEF.is_file():
        ROOT_TASK_DEF.unlink()

    task_def = ROOT.joinpath(f'task_def_{task.lower()}.yaml')
    shutil.copy(f'experiments/{task}/task_def.yaml', task_def)
    task_defs.append(task_def)

# merge task def
merge_yaml(task_defs, ROOT_TASK_DEF)
for td in task_defs:
    td.unlink()
