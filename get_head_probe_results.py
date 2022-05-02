from typing import List, Union, Tuple, Dict
import argparse
from collections import defaultdict
from pathlib import Path
import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

from datasets import Dataset, load_dataset
from conllu import parse_incr

import torch
from torch import nn
from torch.utils.data import DataLoader

from mt_dnn.batcher import SingleTaskDataset, Collater
from mt_dnn.model import MTDNNModel
from mt_dnn.inference import eval_model

from data_utils.task_def import EncoderModelType, TaskType
from experiments.exp_def import (
    Experiment,
    LingualSetting,
    TaskDefs,
)

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

    heatmap.set_ylabel(xaxlabel, fontsize=fontsize)
    heatmap.set_xlabel(yaxlabel, fontsize=fontsize)

    heatmap.set_yticklabels(row_labels, rotation=0, fontsize=fontsize)
    heatmap.set_xticklabels(column_labels, rotation=0, fontsize=fontsize)

    fig = heatmap.get_figure()
    fig.savefig(Path(out_file).with_suffix('.pdf'), bbox_inches='tight')

def build_dataset(data_path, encoder_type, batch_size, max_seq_len, task_def, device_id):
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
        pin_memory=device_id > 0
    )

    return test_data

def construct_model(finetuned_task: Experiment, setting: LingualSetting, device_id: int):
    if setting is LingualSetting.BASE:
        task_def_path = Path('experiments').joinpath(
            finetuned_task.name,
            'cross',
            'task_def.yaml'
        )
    else:
        task_def_path = Path('experiments').joinpath(
            finetuned_task.name,
            setting.name.lower(),
            'task_def.yaml'
        )

    task_defs = TaskDefs(task_def_path)
    assert finetuned_task.name.lower() in task_defs._task_type_map
    assert finetuned_task.name.lower() in task_defs._data_type_map
    assert finetuned_task.name.lower() in task_defs._metric_meta_map

    metric_meta = task_defs._metric_meta_map[finetuned_task.name.lower()]

    # load model
    if setting is not LingualSetting.BASE:
        checkpoint_dir = Path('checkpoint').joinpath(f'{finetuned_task.name}_{setting.name.lower()}')
        checkpoint = list(checkpoint_dir.rglob('model_5*.pt'))[0]
        assert os.path.exists(checkpoint), checkpoint
    else:
        checkpoint_dir = Path('checkpoint').joinpath(f'{finetuned_task.name}_cross')
        checkpoint = list(checkpoint_dir.rglob('model_5*.pt'))[0]

    state_dict = torch.load(checkpoint)
    config = state_dict['config']

    config['fp16'] = False
    config['answer_opt'] = 0
    config['adv_train'] = False
    
    task_def = task_defs.get_task_def(finetuned_task.name.lower())
    task_def_list = [task_def]
    config['task_def_list'] = task_def_list
    config["cuda"] = True
    config['device'] = device_id

    if 'optimizer' in state_dict:
        del state_dict['optimizer']

    model = MTDNNModel(config, devices=[device_id])

    # scoring_list classification head doesn't matter because we're just taking
    # the head probe outputs.
    if 'scoring_list.0.weight' in state_dict['state']:
        state_dict['state']['scoring_list.0.weight'] = model.network.state_dict()['scoring_list.0.weight']
        state_dict['state']['scoring_list.0.bias'] = model.network.state_dict()['scoring_list.0.bias']

    if setting is not LingualSetting.BASE:
        model.load_state_dict(state_dict)
    
    return model, metric_meta

def get_acc(model, test_data, metric_meta, device_id, head_probe):
    with torch.no_grad():
        model.network.eval()
        model.network.to(device_id)
        
        results = eval_model(
            model,
            test_data,
            task_type=TaskType.Classification,
            metric_meta=metric_meta,
            device=device_id,
            with_label=True,
            head_probe=head_probe
        )
    metrics = results[0]
    predictions = results[1]
    golds = results[3]
    ids = results[4]

    preds_df = pd.Series(predictions)
    golds_df = pd.Series(golds)
    id_df = pd.Series(ids)
    return metrics['F1MAC'], preds_df, golds_df, id_df

def evaluate_head_probe(
    hlhis: List,
    finetuned_task: Experiment,
    downstream_task: Experiment,
    setting: LingualSetting,
    batch_size: int,
    max_seq_len: int,
    device_id: int):

    """
    Evaluate head probe for task_finetuned model on a downstream subtask.
    """
    output_dir = Path('head_probe_outputs').joinpath(
        finetuned_task.name,
        downstream_task.name,
        setting.name.lower()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    task_def_path = Path('experiments').joinpath(
        finetuned_task.name,
        'cross',
        'task_def.yaml'
    )
    task_def = TaskDefs(task_def_path).get_task_def(finetuned_task.name.lower())
    data_path = Path('experiments').joinpath(
        downstream_task.name,
        'multi',
        'bert-base-multilingual-cased',
        f'{downstream_task.name.lower()}_test.json'
    )

    # quick scan for done.
    hlhis_to_probe = []
    for (hl, hi) in hlhis:
        output_file_for_head = output_dir.joinpath(f'{hl}_{hi}.csv')
        if not output_file_for_head.is_file():
            hlhis_to_probe.append((hl, hi))
    
    if len(hlhis_to_probe) == 0:
        return 
    else:
        test_data = build_dataset(
            data_path,
            EncoderModelType.BERT,
            batch_size,
            max_seq_len,
            task_def,
            device_id)
    
        model, metric_meta = construct_model(
            finetuned_task,
            setting,
            device_id)
    
    for (hl, hi) in hlhis_to_probe:
        print(f'\n[{finetuned_task.name}/{downstream_task.name}, {setting.name}]: {hl}_{hi}...')
        output_file_for_head = output_dir.joinpath(f'{hl}_{hi}.csv')

        # load state dict for the attention head.
        state_dict_for_head = Path('checkpoint').joinpath(
            'head_probing',
            finetuned_task.name,
            downstream_task.name,
            setting.name.lower(),
            str(hl),
            str(hi))

        state_dict_for_head = list(state_dict_for_head.rglob("*.pt"))[0]
        state_dict_for_head = torch.load(str(state_dict_for_head))['state']

        # then attach the probing layer
        model.attach_head_probe(hl, hi, task_def.n_class)

        # get the layer and check
        layer = model.network.get_attention_layer(hl)
        assert hasattr(layer, 'head_probe_dense_layer')

        # and load (put it on same device)
        weight = state_dict_for_head[f'bert.encoder.layer.{hl}.attention.self.head_probe_dense_layer.weight']
        bias = state_dict_for_head[f'bert.encoder.layer.{hl}.attention.self.head_probe_dense_layer.bias']
        
        layer.head_probe_dense_layer.weight = nn.Parameter(weight.to(device_id))
        layer.head_probe_dense_layer.bias = nn.Parameter(bias.to(device_id))

        # compute acc and save
        _, preds_for_layer, golds, ids = get_acc(model, test_data, metric_meta, device_id, head_probe=True)
        pd.Series(preds_for_layer).to_csv(output_file_for_head)

        # save labels and ids
        if not output_dir.joinpath(f'labels.csv').is_file():
            pd.Series(golds).to_csv(output_dir.joinpath(f'labels.csv'))
        
        if not output_dir.joinpath(f'ids.csv').is_file():
            pd.Series(ids).to_csv(output_dir.joinpath(f'ids.csv'))
        
        # detach
        model.detach_head_probe(hl)

def distribute_heads_to_gpus(
    root_out_path,
    finetuned_task,
    downstream_task,
    setting,
    available_devices):

    # quick check for done
    heads_to_distribute = []
    for hl in range(0, 12):
        for hi in range(0, 12):
            result_csv_for_head = root_out_path.joinpath(
                finetuned_task.name,
                downstream_task.name,
                setting.name.lower(),
                f'{hl}_{hi}.csv')
            if not result_csv_for_head.is_file():
                heads_to_distribute.append((hl, hi))
            else:
                print(f'{(hl, hi)} exists.')
    
    hlhis = []
    devices = []
    n_per_gpu = len(heads_to_distribute) // len(available_devices)
    gpu_group = []
    curr_device_idx = 0
    n_heads_cumul = 0

    for (hl, hi) in heads_to_distribute:
        n_heads_cumul += 1
        gpu_group.append((hl, hi))

        if ((n_heads_cumul % n_per_gpu == 0 and n_heads_cumul != 0) or n_heads_cumul == 144):
            hlhis.append(gpu_group)
            devices.append(available_devices[curr_device_idx])
            print(f'adding {len(gpu_group)} to device {available_devices[curr_device_idx]}')
            gpu_group = []
            curr_device_idx += 1
    
    return hlhis, devices

def evaluate_head_probe_multi_gpu_wrapper(
    finetuned_task: Experiment,
    downstream_tasks: List,
    root_out_path: Path = Path('head_probe_outputs'),
    devices: List[int] = list(range(torch.cuda.device_count()))):
    
    print('Evaluating the accuracy of each head...')

    for downstream_task in downstream_tasks:
        for setting in [
            LingualSetting.CROSS,
            LingualSetting.MULTI
            ]:
            out_results_file = root_out_path.joinpath(
                finetuned_task.name,
                downstream_task.name,
                setting.name.lower(),
                'results.csv')

            if Path(out_results_file).is_file():
                print(f'\tDone and collected: {finetuned_task.name}_{setting.name} -> {downstream_task.name}.')
            else:
                print(f'\tEvaluating for {finetuned_task.name}_{setting.name} -> {downstream_task.name}.')

                available_devices = [int(d) for d in args.devices]
                hlhis, devices = distribute_heads_to_gpus(
                    root_out_path,
                    finetuned_task,
                    downstream_task,
                    setting,
                    available_devices
                )
                eval_head_probe_args = []

                for i, hlhi_set in enumerate(hlhis):
                    args_ = [
                        hlhi_set,
                        finetuned_task,
                        downstream_task,
                        setting,
                        args.batch_size,
                        args.max_seq_len,
                        devices[i]
                    ]
                    eval_head_probe_args.append(tuple(args_))
                
                with torch.multiprocessing.get_context('spawn').Pool(len(available_devices)) as p:
                    p.starmap(evaluate_head_probe, eval_head_probe_args)
                
                # collect.
                print(f'\n\t Done: {finetuned_task.name}_{setting.name} -> {downstream_task.name}. Collecting...')
                output_dir = Path('head_probe_outputs').joinpath(
                    finetuned_task.name,
                    downstream_task.name,
                    setting.name.lower()
                )

                full_layer_results = []
                for hl in range(12):
                    for hi in range(12):
                        result_for_layer = pd.read_csv(output_dir.joinpath(f'{hl}_{hi}.csv'), index_col=0)
                        full_layer_results.append(result_for_layer)
                
                full_ids = pd.read_csv(output_dir.joinpath(f'ids.csv'), index_col=0)
                full_golds = pd.read_csv(output_dir.joinpath(f'labels.csv'), index_col=0)

                full_results = pd.concat([full_ids, full_golds] + full_layer_results, axis=1)
                full_results.to_csv(output_dir.joinpath(f'results.csv'))

def get_lang_to_id(task: Experiment) -> Dict[str, List[int]]:
    def _pawsx():
        lang_to_id = defaultdict(list)

        i = 0
        for lang in ['en', 'fr', 'de', 'es']:
            dataset = load_dataset('paws-x', lang, split='test')
            lang_to_id[lang] = list(range(i, i + len(dataset)))
            i += len(dataset)
        
        return lang_to_id

    def _pos():
        data_root = Path('/home/june/mt-dnn/experiments/POS/data')

        dataset_dirs = [
            data_root.joinpath('en/UD_English-EWT'),
            data_root.joinpath('fr/UD_French-FTB'),
            data_root.joinpath('de/UD_German-GSD'),
            data_root.joinpath('es/UD_Spanish-AnCora')
        ]

        langs = ['en', 'fr', 'de', 'es']
        lang_to_id = defaultdict(list)
        cumulative_sent_idx = 0

        for i, data_dir in enumerate(dataset_dirs):
            lang = langs[i]
            with open(data_dir.joinpath(f'test.conllu'), 'r', encoding='utf-8') as f:
                for _, _ in enumerate(parse_incr(f)):
                    lang_to_id[lang].append(cumulative_sent_idx)
                    cumulative_sent_idx += 1
        
        return lang_to_id

    def _ner():
        root = 'experiments/NER/multi/ner_test_tmp.json'
        df = Dataset.from_json(root)
        lang_to_id = defaultdict(list)

        for i, row in enumerate(df):
            lang = row['langs'][0]
            lang_to_id[lang].append(i)
        
        return lang_to_id

    def _marc():
        root = 'experiments/MARC/multi/marc_test_tmp.json'
        df = Dataset.from_json(root)
        lang_to_id = defaultdict(list)

        for i, row in enumerate(df):
            lang = row['language']
            lang_to_id[lang].append(i)
        
        return lang_to_id
    
    if task is Experiment.POS:
        return _pos()
    elif task is Experiment.NER:
        return _ner()
    elif task is Experiment.PAWSX:
        return _pawsx()
    elif task is Experiment.MARC:
        return _marc()
    else:
        raise NotImplementedError(task.name)

def get_results_csvs(
    finetuned_task: Experiment,
    downstream_task: Experiment,
    setting: LingualSetting,
    languages: List[str], 
    do_individual=True,
    do_combined=True,
    combined_postfix='combined'):

    def _get_acc_for_heads(labels, preds):
        results = np.zeros((12, 12))
        
        for hl in range(12):
            for hi in range(12):
                preds_for_head = preds.iloc[:, hl*12+hi]
                acc = accuracy_score(labels, preds_for_head)
                results[hl, hi] = acc
        
        return results
    
    print(f'Getting result csvs for {finetuned_task.name}_{setting.name} -> {downstream_task.name}...')

    if setting is not LingualSetting.BASE:
        raw_results_path = Path('head_probe_outputs').joinpath(
            finetuned_task.name,
            downstream_task.name,
            setting.name.lower(),
            'results.csv'
        )
    else:
       raw_results_path = Path('head_probe_outputs').joinpath(
            'mBERT',
            downstream_task.name,
            'base',
            'results.csv'
        ) 

    raw_results = pd.read_csv(raw_results_path, index_col=0)
    
    if do_individual:
        language_to_ids = get_lang_to_id(downstream_task)

        # individual languages
        for lang in languages:
            out_file = Path('head_probe_outputs').joinpath(
                finetuned_task.name,
                downstream_task.name,
                setting.name.lower(),
                f'{finetuned_task.name.lower()}_{setting.name.lower()}-{downstream_task.name.lower()}-{lang}-pure.csv')
            
            if Path(out_file).is_file():
                print(f'\tDone: {finetuned_task.name}_{setting.name} -> {downstream_task.name} [{lang}]')
            else:  
                ids = language_to_ids[lang]
                data_for_lang = raw_results.iloc[ids, :]

                labels = data_for_lang.iloc[:, 1]
                preds = data_for_lang.iloc[:, 2:]
                results = _get_acc_for_heads(labels, preds)
                
                out_df = pd.DataFrame(results)
                out_df.to_csv(out_file)
    
    if do_combined:
        if setting is LingualSetting.BASE:
            combined_out_file = Path('head_probe_outputs').joinpath(
                'mBERT',
                downstream_task.name,
                setting.name.lower(),
                f'{setting.name.lower()}-{downstream_task.name.lower()}-{combined_postfix}.csv')
        else:
            combined_out_file = Path('head_probe_outputs').joinpath(
                finetuned_task.name,
                downstream_task.name,
                setting.name.lower(),
                f'{finetuned_task.name.lower()}_{setting.name.lower()}-{downstream_task.name.lower()}-{combined_postfix}.csv')

        if Path(combined_out_file).is_file():
            print(f'\tDone: {finetuned_task.name}_{setting.name} -> {downstream_task.name} [{combined_postfix}]')
        else:
            combined_labels = raw_results.iloc[:, 1]
            combined_preds = raw_results.iloc[:, 2:]

            combined_results = _get_acc_for_heads(combined_labels, combined_preds)
            out_df = pd.DataFrame(combined_results)
            out_df.to_csv(combined_out_file)
    
def get_final_probing_result(
    finetuned_task: Experiment,
    downstream_task: Experiment,
    languages: List[str],
    do_individual=True,
    do_combined=True,
    combined_postfix='combined'):

    def _get_individual_probing_result(
        finetuned_task: Experiment,
        downstream_task: Experiment,
        setting: LingualSetting,
        combined=False,
        lang=None):

        assert combined or (not combined and lang is not None)

        if not combined:
            print(f'Getting final pretty results for {finetuned_task.name}_{setting.name} -> {downstream_task.name} [{lang}]...', end='')
        else:
            print(f'Getting final pretty results for {finetuned_task.name}_{setting.name} -> {downstream_task.name} [{combined_postfix}]...', end='')

        base_prefix = Path(f'head_probe_outputs/mBERT/{downstream_task.name}')
        base_postfix = f'base/base-{downstream_task.name.lower()}'

        prefix = Path(f'head_probe_outputs/{finetuned_task.name}/{downstream_task.name}/{setting.name.lower()}')
        postfix = f'{finetuned_task.name.lower()}_{setting.name.lower()}-{downstream_task.name.lower()}'
        
        base_csv_path = base_prefix.joinpath(base_postfix)
        result_csv_path = prefix.joinpath(postfix)

        out_file_root = prefix.parent.joinpath('results', postfix)

        if not combined:
            base_csv_path = str(base_csv_path) + f'-{lang}-pure'
            result_csv_path = str(result_csv_path) + f'-{lang}-pure'
            out_file_root = str(out_file_root) + f'-{lang}'
        else:
            base_csv_path = str(base_csv_path) + f'-{combined_postfix}'
            result_csv_path = str(result_csv_path) + f'-{combined_postfix}'
            out_file_root = str(out_file_root) + f'-{combined_postfix}'
        
        out_file_root = Path(out_file_root)
        base_csv_path = Path(base_csv_path)
        result_csv_path = Path(result_csv_path)

        if not (out_file_root.with_suffix('.pdf').is_file() and out_file_root.with_suffix('.csv').is_file()):
            out_file_root.parent.mkdir(parents=True, exist_ok=True)

            base_df = pd.read_csv(base_csv_path.with_suffix('.csv'), index_col=0)
            setting_df = pd.read_csv(result_csv_path.with_suffix('.csv'), index_col=0)

            diff = setting_df - base_df
            diff.to_csv(out_file_root.with_suffix('.csv'))
            create_heatmap(
                data_df=diff,
                row_labels=list(range(1, 13)),
                column_labels=list(range(1, 13)),
                xaxlabel='heads',
                yaxlabel='layers',
                invert_y=True,
                fontsize=25,
                out_file=out_file_root.with_suffix('.pdf'))
        
        print('done.')

    for setting in [LingualSetting.CROSS, LingualSetting.MULTI]:
        if do_individual:
            for lang in languages:
                _get_individual_probing_result(
                    finetuned_task,
                    downstream_task,
                    setting,
                    combined=False,
                    lang=lang)

        if do_combined:
            _get_individual_probing_result(
                finetuned_task,
                downstream_task,
                setting,
                combined=True)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--devices', nargs='+', default='')
    parser.add_argument('--finetuned_task', type=str)
    parser.add_argument('--downstream_task', type=str, default='')
    parser.add_argument("--combined_postfix", type=str, default='combined')
    parser.add_argument('--do_individual', action='store_true')
    parser.add_argument('--do_combined', action='store_true')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_seq_len', type=int, default=512)

    args = parser.parse_args()

    if args.devices == '':
        args.devices = list(range(torch.cuda.device_count()))
    
    if args.downstream_task == '':
        downstream_tasks = list(Experiment)
        # downstream_tasks.remove(Experiment.NLI)
    else:
        downstream_tasks = [Experiment[args.downstream_task.upper()]]
    
    finetuned_task = Experiment[args.finetuned_task.upper()]
    evaluate_head_probe_multi_gpu_wrapper(finetuned_task, downstream_tasks, devices=args.devices)

    languages = [
        'en',
        'fr',
        'es',
        'de'
    ]
    for downstream_task in downstream_tasks:
        for setting in [LingualSetting.BASE, LingualSetting.CROSS, LingualSetting.MULTI]:
            get_results_csvs(
                finetuned_task,
                downstream_task,
                setting,
                languages,
                do_individual=args.do_individual,
                do_combined=args.do_combined,
                combined_postfix=args.combined_postfix
            )

        get_final_probing_result(
            finetuned_task,
            downstream_task,
            languages,
            do_individual=args.do_individual,
            do_combined=args.do_combined,
            combined_postfix=args.combined_postfix)
