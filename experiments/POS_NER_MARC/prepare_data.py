import shutil
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from experiments.utils import merge_yaml

POS_TASK_DEF = Path('experiments/POS_NER_MARC/task_def_pos.yaml')
NER_TASK_DEF = Path('experiments/POS_NER_MARC/task_def_ner.yaml')
MARC_TASK_DEF = Path('experiments/POS_NER_MARC/task_def_marc.yaml')
ROOT_TASK_DEF = Path('experiments/POS_NER_MARC/task_def.yaml')

for task in ['POS', 'NER', 'MARC']:
    shutil.copytree(f'experiments/{task}/cross', 'experiments/POS_NER_MARC/', dirs_exist_ok=True)
    if task == 'POS':
        shutil.move(ROOT_TASK_DEF, POS_TASK_DEF)
    if task == 'NER':
        shutil.move(ROOT_TASK_DEF, NER_TASK_DEF)
    if task == 'MARC':
        shutil.move(ROOT_TASK_DEF, MARC_TASK_DEF)

# merge task def
merge_yaml([POS_TASK_DEF, NER_TASK_DEF, MARC_TASK_DEF], ROOT_TASK_DEF)
POS_TASK_DEF.unlink()
NER_TASK_DEF.unlink()
MARC_TASK_DEF.unlink()
