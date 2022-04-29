from typing import Union, List, Tuple
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from pathlib import Path


from experiments.exp_def import (
    TaskDefs,
    Experiment
)

import pickle
import torch
from torch.utils.data import DataLoader
from mt_dnn.inference import eval_model
from mt_dnn.batcher import SingleTaskDataset, Collater

def create_heatmap(
    data_csv_path: str = '',
    data_df: Union[pd.DataFrame, None] = None,
    row_labels: List[str] = None,
    column_labels: List[str] = None,
    xaxlabel: str = None,
    yaxlabel: str = None,
    invert_y: bool = False,
    figsize: Tuple[int, int] = (14, 14),
    fontsize: int = 14,
    cmap: str = 'RdYlGn',
    out_file: str= ''
    ):
    """
    General heatmap from data.
    """
    # read data if dataframe not directly supplied.
    if data_df is None:
        data_df = pd.read_csv(data_csv_path, index_col=0)
        assert len(out_file) > 0, f'invalid csv: {data_csv_path}'
    
    plt.figure(figsize=figsize)
    annot_kws = {
        "fontsize":fontsize,
    }
    heatmap = sns.heatmap(
        data_df.to_numpy(),
        cbar=False,
        annot=True,
        annot_kws=annot_kws,
        fmt=".2f",
        cmap=cmap)

    if invert_y:
        heatmap.invert_yaxis()

    heatmap.set_ylabel(yaxlabel, fontsize=fontsize)
    heatmap.set_xlabel(xaxlabel, fontsize=fontsize)

    heatmap.set_yticklabels(row_labels, rotation=0, fontsize=fontsize)
    heatmap.set_xticklabels(column_labels, rotation=0, fontsize=fontsize)

    fig = heatmap.get_figure()
    fig.savefig(Path(out_file).with_suffix('.pdf'), bbox_inches='tight')

def build_dataset(data_path, encoder_type, batch_size, max_seq_len, task_def):
    test_data_set = SingleTaskDataset(
        path=data_path,
        is_train=False,
        maxlen=max_seq_len,
        task_id=0,
        task_def=task_def
    )

    collater = Collater(is_train=False, encoder_type=encoder_type)

    test_data = DataLoader(
        test_data_set,
        batch_size=batch_size,
        collate_fn=collater.collate_fn,
        pin_memory=True
    )

    return test_data

def get_metric(
    model,
    test_data,
    metric_meta,
    task_type,
    device_id,
    label_mapper,
    model_probe=False,
    head_probe=False):

    with torch.no_grad():
        model.network.eval()
        model.network.to(device_id)
        
        results = eval_model(
            model,
            test_data,
            task_type=task_type,
            metric_meta=metric_meta,
            device=device_id,
            with_label=True,
            model_probe=model_probe,
            head_probe=head_probe,
            label_mapper=label_mapper
        )
    
    metrics = results[0]
    predictions = results[1]
    golds = results[3]
    ids = results[4]

    preds_df = pd.Series(predictions)
    golds_df = pd.Series(golds)
    id_df = pd.Series(ids)

    metric_name = metric_meta[0].name

    return metrics[metric_name], preds_df, golds_df, id_df

def base_construct_model(checkpoint: str, task: Experiment, task_def_path: str, device_id: int):
    task_defs = TaskDefs(task_def_path)
    task_name = task.name.lower()
    metric_meta = task_defs._metric_meta_map[task_name]

    # base mBERT model
    if checkpoint == '':
        with open('checkpoint/dummy_config.pkl', 'rb') as fr:
            config = pickle.load(fr)
        state_dict = None
    else:
        state_dict = torch.load(checkpoint, map_location=f'cuda:{device_id}')
        config = state_dict['config']
    
    config['fp16'] = False
    config['answer_opt'] = 0
    config['adv_train'] = False
    
    task_def_list = [task_defs.get_task_def(task_name)]
    config['task_def_list'] = task_def_list
    config["cuda"] = True
    config['device'] = device_id

    return config, state_dict, metric_meta