from pathlib import Path
import shutil

for f in Path('head_probe_ranking').joinpath('combined').rglob('*.np'):
    dst = f.with_suffix('.csv')
    shutil.move(f, dst)