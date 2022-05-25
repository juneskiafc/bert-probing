import shutil
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from experiments.utils import merge_yaml

PAWSX_TASK_DEF = Path('experiments/PAWSX_NER/task_def_pawsx.yaml')
NER_TASK_DEF = Path('experiments/PAWSX_NER/task_def_ner.yaml')
ROOT_TASK_DEF = Path('experiments/PAWSX_NER/task_def.yaml')

shutil.copytree('experiments/PAWSX/cross', 'experiments/PAWSX_NER/', dirs_exist_ok=True)
shutil.move(ROOT_TASK_DEF, PAWSX_TASK_DEF)

shutil.copytree('experiments/NER/cross', 'experiments/PAWSX_NER/', dirs_exist_ok=True)
shutil.move(ROOT_TASK_DEF, NER_TASK_DEF)

# merge task def
merge_yaml([PAWSX_TASK_DEF, NER_TASK_DEF], ROOT_TASK_DEF)
PAWSX_TASK_DEF.unlink()
NER_TASK_DEF.unlink()
