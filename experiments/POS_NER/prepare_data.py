import shutil
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from experiments.utils import merge_yaml

POS_TASK_DEF = Path('experiments/POS_NER/task_def_pos.yaml')
NER_TASK_DEF = Path('experiments/POS_NER/task_def_ner.yaml')
ROOT_TASK_DEF = Path('experiments/POS_NER/task_def.yaml')

shutil.copytree('experiments/POS/cross', 'experiments/POS_NER/', dirs_exist_ok=True)
shutil.move(ROOT_TASK_DEF, POS_TASK_DEF)

shutil.copytree('experiments/NER/cross', 'experiments/POS_NER/', dirs_exist_ok=True)
shutil.move(ROOT_TASK_DEF, NER_TASK_DEF)

# merge task def
merge_yaml([POS_TASK_DEF, NER_TASK_DEF], ROOT_TASK_DEF)
POS_TASK_DEF.unlink()
NER_TASK_DEF.unlink()
