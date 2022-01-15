"""
Extract the top-n heads amongst head-probe results of several downstream tasks.
Then create a head_mask matrix (12x12), where the intersection of the top-n heads between all tasks
have a value of 1, and others have a value of 0.5.
"""

from pathlib import Path
from experiments.exp_def import Experiment

def extract_top_n_heads(n=60, tasks=Experiment):
    pass