from pathlib import Path
import torch

for task in ['MARC', 'NLI', 'POS', 'NER', 'PAWSX']:
    cross_checkpoint_dir = Path(f'checkpoint/{task}_cross')
    multi_checkpoint_dir = Path(f'checkpoint/{task}_multi')
    if cross_checkpoint_dir.is_dir() and multi_checkpoint_dir.is_dir():
        print(f'transplanting {task}')
        base_model_ckpt = list(cross_checkpoint_dir.rglob("*.pt"))[0]
        multilingual_model_ckpt = list(multi_checkpoint_dir.rglob("*.pt"))[0]

        base_model = torch.load(base_model_ckpt)
        multilingual_model = torch.load(multilingual_model_ckpt)

        base_model['state']['scoring_list.0.weight'] = multilingual_model['state']['scoring_list.0.weight']
        base_model['state']['scoring_list.0.bias'] = multilingual_model['state']['scoring_list.0.bias']

        torch.save(base_model, Path(f'checkpoint/{task}_transplant.pt'))