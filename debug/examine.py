from pathlib import Path
import torch

marc_cross = torch.load(Path('debug/cross_head_training/NLI/MARC.pt'))
marc_multi = torch.load(Path('debug/multi_head_training/NLI/MARC.pt'))

print(torch.equal(marc_cross, marc_multi))

