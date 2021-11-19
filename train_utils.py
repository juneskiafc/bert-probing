import json
import os
from pathlib import Path
from torch.distributed import is_initialized, get_rank

def save_checkpoint(model, epoch, output_dir):
    model_file = Path(output_dir).joinpath(f'model_{epoch}_{model.updates}.pt')
    model_file.parent.mkdir(exist_ok=True, parents=True)

    # only keep last
    existing_checkpoints = list(Path(output_dir).iterdir())
    if len(existing_checkpoints) > 0:
        for checkpoint in existing_checkpoints:
            if Path(str(checkpoint)).is_file():
                os.remove(checkpoint)
    
    model.save(model_file)
    return model_file

def dump_opt(opt, output_dir):
    config_file = os.path.join(output_dir, 'config.json')
    with open(config_file, 'w', encoding='utf-8') as writer:
        writer.write('{}\n'.format(json.dumps(opt)))
    
def dump(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)

def print_message(logger, message, level=0):
    if is_initialized():
        if get_rank() == 0:
            do_logging = True
        else:
            do_logging = False
    else:
        do_logging = True
    
    if do_logging:
        if level == 1:
            logger.warning(message)
        else:
            logger.info(message)