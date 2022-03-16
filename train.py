# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import argparse
from pathlib import Path
import os

import torch
from torch.utils.data import DataLoader
from pretrained_models import *

from experiments.exp_def import TaskDefs
from data_utils.log_wrapper import create_logger
from data_utils.task_def import EncoderModelType
from data_utils.utils import set_environment

from mt_dnn.model import MTDNNModel
from mt_dnn.batcher import (
    SingleTaskDataset,
    MultiTaskDataset,
    Collater,
    MultiTaskBatchSampler
)
from train_utils import (
    dump_opt,
    print_message,
    save_checkpoint
)
from gradient_probing import prediction_gradient
import wandb

def model_config(parser):
    # SET THESE
    ##############
    parser.add_argument('--multi_gpu_on', action='store_true')
    parser.add_argument('--devices', nargs='+')

    # head probing
    parser.add_argument('--head_probe', action='store_true')
    parser.add_argument('--head_probe_layer', type=int)
    parser.add_argument('--head_probe_idx', type=int)
    parser.add_argument('--head_probe_n_classes', type=int)

    # model probing
    parser.add_argument('--model_probe', action='store_true')
    parser.add_argument('--model_probe_n_classes', type=int)

    # gradient probing
    parser.add_argument('--gradient_probe', action='store_true')

    # finetuning after joint modeling
    parser.add_argument('--jm_finetune', action='store_true')
    parser.add_argument('--jm_finetune_new', action='store_true') # if task does not exist in 1st stage finetuning
    parser.add_argument('--jm_task_id', type=int, default=0) # if task exists from 1st stage finetuning
    ##############

    # DON"T NEED THESE
    ##############
    parser.add_argument('--update_bert_opt', default=0, type=int)
    parser.add_argument('--mem_cum_type', type=str, default='simple',
                        help='bilinear/simple/defualt')
    parser.add_argument('--answer_num_turn', type=int, default=5)
    parser.add_argument('--answer_mem_drop_p', type=float, default=0.1)
    parser.add_argument('--answer_att_hidden_size', type=int, default=128)
    parser.add_argument('--answer_att_type', type=str, default='bilinear',
                        help='bilinear/simple/defualt')
    parser.add_argument('--answer_rnn_type', type=str, default='gru',
                        help='rnn/gru/lstm')
    parser.add_argument('--answer_sum_att_type', type=str, default='bilinear',
                        help='bilinear/simple/defualt')
    parser.add_argument('--answer_merge_opt', type=int, default=1)
    parser.add_argument('--answer_mem_type', type=int, default=1)
    parser.add_argument('--max_answer_len', type=int, default=10)
    parser.add_argument('--answer_dropout_p', type=float, default=0.1)
    parser.add_argument('--answer_weight_norm_on', action='store_true')
    parser.add_argument('--dump_state_on', action='store_true')
    parser.add_argument('--answer_opt', type=int, default=1, help='0,1')
    parser.add_argument('--pooler_actf', type=str, default='tanh',
                        help='tanh/relu/gelu')
    parser.add_argument('--mtl_opt', type=int, default=0)
    parser.add_argument('--ratio', type=float, default=0)
    parser.add_argument('--mix_opt', type=int, default=0)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--init_ratio', type=float, default=1)
    parser.add_argument('--encoder_type', type=int, default=EncoderModelType.BERT)
    parser.add_argument('--num_hidden_layers', type=int, default=-1)

    # BERT pre-training
    parser.add_argument('--bert_model_type', type=str, default='bert-base-multilingual-cased')
    parser.add_argument('--init_checkpoint', type=str, default='bert-base-multilingual-cased')
    parser.add_argument('--do_lower_case', action='store_true')
    parser.add_argument('--masked_lm_prob', type=float, default=0.15)
    parser.add_argument('--short_seq_prob', type=float, default=0.2)
    parser.add_argument('--max_predictions_per_seq', type=int, default=128)

    # bin samples
    parser.add_argument('--bin_on', action='store_true')
    parser.add_argument('--bin_size', type=int, default=64)
    parser.add_argument('--bin_grow_ratio', type=int, default=0.5)

    # dist training
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--world_size", type=int, default=1, help="For distributed training: world size")
    parser.add_argument("--master_addr", type=str, default="localhost")
    parser.add_argument("--master_port", type=str, default="6600")
    parser.add_argument("--backend", type=str, default="nccl")
    ##############

    return parser


def data_config(parser):
    # SET THESE
    ############
    parser.add_argument('--exp_name', default='', help='experiment name')
    parser.add_argument('--dataset_name', default='', help='dataset name')
    parser.add_argument('--wandb', action='store_true')
    ############

    # DON'T NEED THESE
    ############
    parser.add_argument('--log_file', default='mt-dnn.log', help='path for log file.')
    parser.add_argument('--data_sort_on', action='store_true')
    parser.add_argument('--mkd-opt', type=int, default=0, 
                        help=">0 to turn on knowledge distillation, requires 'softlabel' column in input data")
    parser.add_argument('--do_padding', action='store_true')
    ############
    return parser


def train_config(parser):
    # SET THESE
    ############
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=8)

    # loading
    parser.add_argument("--model_ckpt", default='', type=str)
    parser.add_argument("--resume", action='store_true')
    parser.add_argument('--huggingface_ckpt', action='store_true')
    ############

    # DON'T NEED THESE
    ############
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(),
                        help='whether to use GPU acceleration.')
    parser.add_argument('--log_per_updates', type=int, default=100)
    parser.add_argument('--save_per_updates', type=int, default=10000)
    parser.add_argument('--save_per_updates_on', action='store_true')
    parser.add_argument('--optimizer', default='adamax',
                        help='supported optimizer: adamax, sgd, adadelta, adam')
    parser.add_argument('--grad_clipping', type=float, default=0)
    parser.add_argument('--global_grad_clipping', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--warmup', type=float, default=0.1)
    parser.add_argument('--warmup_schedule', type=str, default='warmup_linear')
    parser.add_argument('--adam_eps', type=float, default=1e-6)

    parser.add_argument('--vb_dropout', action='store_false')
    parser.add_argument('--dropout_p', type=float, default=0.1)
    parser.add_argument('--dropout_w', type=float, default=0.000)
    parser.add_argument('--bert_dropout_p', type=float, default=0.1)

    # scheduler
    parser.add_argument('--have_lr_scheduler', dest='have_lr_scheduler', action='store_false')
    parser.add_argument('--multi_step_lr', type=str, default='10,20,30')
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--scheduler_type', type=str, default='ms', help='ms/rop/exp')
    parser.add_argument('--output_dir', default='checkpoint')
    parser.add_argument('--seed', type=int, default=2018,
                        help='random seed for data shuffling, embedding init, etc.')
    parser.add_argument('--grad_accumulation_step', type=int, default=1)

    #fp 16
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    # adv training
    parser.add_argument('--adv_train', action='store_true')

    # the current release only includes smart perturbation
    parser.add_argument('--adv_opt', default=0, type=int)
    parser.add_argument('--adv_norm_level', default=0, type=int)
    parser.add_argument('--adv_p_norm', default='inf', type=str)
    parser.add_argument('--adv_alpha', default=1, type=float)
    parser.add_argument('--adv_k', default=1, type=int)
    parser.add_argument('--adv_step_size', default=1e-5, type=float)
    parser.add_argument('--adv_noise_var', default=1e-5, type=float)
    parser.add_argument('--adv_epsilon', default=1e-6, type=float)
    parser.add_argument('--encode_mode', action='store_true', help="only encode test data")
    parser.add_argument('--debug', action='store_true', help="print debug info")

    # transformer cache
    parser.add_argument("--transformer_cache", default='.cache', type=str)
    ############

    return parser

# parse args
parser = argparse.ArgumentParser()
parser = data_config(parser)
parser = model_config(parser)
parser = train_config(parser)
args = parser.parse_args()

# some stuff in data can be automated
dataset_name = args.dataset_name
args.data_dir = f'experiments/{dataset_name}/{args.bert_model_type}'
args.task_def = f'experiments/{dataset_name}/task_def.yaml'
if "/" in dataset_name:
    args.train_datasets = dataset_name.split("/")[0].lower()
else:
    args.train_datasets = dataset_name.lower()

# set task name, root data dir, and output dir.
output_dir = args.output_dir
data_dir = args.data_dir

# multiple datasets are split by '_'
args.train_datasets = args.train_datasets.split('_')

# seed everything.
set_environment(args.seed, args.cuda)

# stores task: param_args for each TaskDef param
task_defs = TaskDefs(args.task_def)
encoder_type = args.encoder_type

exp_name = args.exp_name

# make log dir and set logger.
log_path = f'logs/{exp_name}/{args.log_file}'
Path(log_path).parent.mkdir(parents=True, exist_ok=True)
logger = create_logger(__name__, to_disk=True, log_file=log_path)

# make output dir and set to absolute path.
if not args.head_probe:
    output_dir = Path(output_dir).joinpath(exp_name)

output_dir = Path(os.path.abspath(output_dir))
output_dir.mkdir(exist_ok=True, parents=True)

def main():
    print_message(logger, 'Launching MT-DNN training.')
    opt = vars(args)
    args.devices = [int(g) for g in args.devices]

    tasks = {}
    task_def_list = []
    train_data_lists = []
    train_datasets = []

    # create training dataset.
    for dataset in args.train_datasets:
        prefix = dataset.split('_')[0]

        if prefix not in tasks:
            task_id = len(tasks)
            tasks[prefix] = task_id

            task_def = task_defs.get_task_def(prefix)
            task_def_list.append(task_def)

            train_path = os.path.join(data_dir, '{}_train.json'.format(dataset))
            print_message(logger, 'Loading {} as task {}'.format(train_path, task_id))

            train_data_set = SingleTaskDataset(
                path=train_path,
                is_train=True,
                maxlen=args.max_seq_len,
                task_id=task_id,
                task_def=task_def,
                printable=True)
            
            train_datasets.append(train_data_set)

    train_collater = Collater(
        dropout_w=args.dropout_w,
        encoder_type=encoder_type,
        soft_label=False,
        max_seq_len=args.max_seq_len,
        do_padding=args.do_padding)
    
    multi_task_train_dataset = MultiTaskDataset(train_datasets)
    multi_task_batch_sampler = MultiTaskBatchSampler(
                                train_datasets,
                                args.batch_size,
                                args.mix_opt,
                                args.ratio,
                                bin_on=args.bin_on,
                                bin_size=args.bin_size,
                                bin_grow_ratio=args.bin_grow_ratio)

    multi_task_train_dataloader = DataLoader(
        multi_task_train_dataset,
        batch_sampler=multi_task_batch_sampler,
        collate_fn=train_collater.collate_fn,
        pin_memory=len(args.devices)>0)

    train_data_lists.append(multi_task_train_dataloader)

    # div number of grad accumulation. 
    n_batch_per_epoch = len(multi_task_train_dataloader) // args.grad_accumulation_step
    num_all_batches = args.epochs * n_batch_per_epoch
    print_message(logger, '############# Gradient Accumulation Info #############')
    print_message(logger, 'number of step: {}'.format(args.epochs * len(multi_task_train_dataloader)))
    print_message(logger, 'number of grad grad_accumulation step: {}'.format(args.grad_accumulation_step))
    print_message(logger, 'adjusted number of step: {}'.format(num_all_batches))
    print_message(logger, '#######################################\n')

    if opt['encoder_type'] not in EncoderModelType._value2member_map_:
        raise ValueError("encoder_type is out of pre-defined types")

    literal_encoder_type = EncoderModelType(opt['encoder_type']).name.lower()
    config_class, _, _ = MODEL_CLASSES[literal_encoder_type]
    config = config_class.from_pretrained(args.bert_model_type).to_dict()

    config['attention_probs_dropout_prob'] = args.bert_dropout_p
    config['hidden_dropout_prob'] = args.bert_dropout_p
    config['multi_gpu_on'] = opt["multi_gpu_on"]

    if args.num_hidden_layers > 0:
        config['num_hidden_layers'] = args.num_hidden_layers

    opt['task_def_list'] = task_def_list
    opt['head_probe'] = args.head_probe
    opt.update(config)

    # if resuming, load state dict, and get init epoch and step.
    if args.resume:
        assert args.model_ckpt != '' and Path(args.model_ckpt).is_file(), args.model_ckpt
        print_message(logger, f'loading model from {args.model_ckpt}')           
        state_dict = torch.load(args.model_ckpt, map_location=f'cuda:{args.devices[0]}')

        if args.jm_finetune:
            if args.jm_finetune_new:
                i = 0
                while f'scoring_list.{i}.weight' in state_dict['state']:
                    del state_dict['state'][f'scoring_list.{i}.weight']
                    del state_dict['state'][f'scoring_list.{i}.bias']
                    i += 1
            else:
                i = 0
                while f'scoring_list.{i}.weight' in state_dict['state']:
                    if i != args.jm_task_id:
                        del state_dict['state'][f'scoring_list.{i}.weight']
                        del state_dict['state'][f'scoring_list.{i}.bias']
                    i += 1
                
                if args.jm_task_id != 0:
                    for param in ['weight', 'bias']:
                        state_dict['state'][f'scoring_list.0.{param}'] = state_dict['state'][f'scoring_list.{args.jm_task_id}.{param}']
                        del state_dict['state'][f'scoring_list.{args.jm_task_id}.{param}']

        if not args.huggingface_ckpt:
            split_model_name = args.model_ckpt.split("/")[-1].split("_")
            if len(split_model_name) > 2:
                init_epoch_idx = int(split_model_name[1]) + 1
                init_global_step = int(split_model_name[2].split(".")[0]) + 1
            else:
                init_epoch_idx = int(split_model_name[1].split(".")[0]) + 1
                init_global_step = 0
        else:
            init_epoch_idx = 0
            init_global_step = 0
    
    else:
        state_dict = None
        init_epoch_idx = 0
        init_global_step = 0

    model = MTDNNModel(
        opt,
        devices=args.devices,
        num_train_step=num_all_batches)
    
    if not (args.head_probe or args.model_probe):
        if state_dict is not None and args.resume:
            if args.huggingface_ckpt:
                if args.bert_model_type == 'bert-base-multilingual-cased':
                    params_to_remove = [
                        "cls.predictions.bias",
                        "cls.predictions.transform.dense.weight",
                        "cls.predictions.transform.dense.bias",
                        "cls.predictions.transform.LayerNorm.weight",
                        "cls.predictions.transform.LayerNorm.bias",
                        "cls.predictions.decoder.weight",
                        "cls.predictions.decoder.bias"
                    ]
                elif args.bert_model_type == 'xlm-roberta-base':
                    params_to_remove = []

                    # rename roberta -> bert
                    renamed_state_dict = {}
                    for n, p in state_dict.items():
                        new_name = n.split('.')
                        if new_name[0] == 'roberta':
                            new_name[0] = 'bert'
                            new_name = '.'.join(new_name)
                            renamed_state_dict[new_name] = p
                    state_dict = renamed_state_dict

                for param_name in params_to_remove:
                    if param_name in state_dict:
                        print(f'{param_name} in state_dict, removing')
                        del state_dict[param_name]
                    else:
                        print(f'{param_name} not in state_dict')
                
                _init_state_dict = model.network.state_dict()

                params_to_add = [
                        "bert.pooler.dense.weight",
                        "bert.pooler.dense.bias",
                        "pooler.dense.weight",
                        "pooler.dense.bias"
                    ]
                num_tasks = len(task_def_list)
                for i in range(num_tasks):
                    params_to_add.extend([f"scoring_list.{i}.weight", f"scoring_list.{i}.bias"])
                
                for param_name in params_to_add:
                    state_dict[param_name] = _init_state_dict[param_name]
                
                state_dict = {'state': state_dict}
            
            if args.jm_finetune and args.jm_finetune_new:
                for param in ['weight', 'bias']:
                    state_dict['state'][f'scoring_list.0.{param}'] = model.network.state_dict()[f'scoring_list.0.{param}']
            
            model.load_state_dict(state_dict)

    if args.head_probe:
        print_message(logger, f'attached head probe at layer #{args.head_probe_layer+1}, head #{args.head_probe_idx+1}')
        opt['head_idx_to_probe'] = (args.head_probe_layer, args.head_probe_idx)

        # freeze all params
        for p in model.network.parameters():
            p.requires_grad = False
        
        # load model, making sure to match scoring_list params
        if args.model_ckpt != '':
            state_dict = torch.load(args.model_ckpt, map_location=f'cuda:{args.devices[0]}')
            state_dict['state']['scoring_list.0.weight'] = model.network.state_dict()['scoring_list.0.weight']
            state_dict['state']['scoring_list.0.bias'] = model.network.state_dict()['scoring_list.0.bias']
            model.load_state_dict(state_dict)
        
        # then attach probing head
        model.attach_head_probe(
            args.head_probe_layer,
            args.head_probe_idx,
            n_classes=args.head_probe_n_classes)
        optimizer_parameters = model._get_param_groups()
        model._setup_optim(optimizer_parameters, None, num_all_batches)
    
        init_epoch_idx = 0
        init_global_step = 0
    
    if args.model_probe:
        print_message(logger, 'probing whole model. Setting n_epochs to 2.')
        args.epochs = 2

        # freeze all params
        for p in model.network.parameters():
            p.requires_grad = False
        
        # load model, making sure to match scoring_list params
        if args.model_ckpt != '':
            print(f'model_probe, loading model ckpt from {args.model_ckpt}')
            state_dict = torch.load(args.model_ckpt, map_location=f'cuda:{args.devices[0]}')
            state_dict['state']['scoring_list.0.weight'] = model.network.state_dict()['scoring_list.0.weight']
            state_dict['state']['scoring_list.0.bias'] = model.network.state_dict()['scoring_list.0.bias']
            model.load_state_dict(state_dict)
        
        # then attach probing head
        model.attach_model_probe(n_classes=args.model_probe_n_classes)
        optimizer_parameters = model._get_param_groups()
        model._setup_optim(optimizer_parameters, None, num_all_batches)

        init_epoch_idx = 0
        init_global_step = 0
    
    # dump config
    dump_opt(opt, output_dir)

    if args.gradient_probe:
        save_path = Path('gradient_probe_outputs').joinpath(exp_name)
        prediction_gradient(
            args,
            model, 
            multi_task_train_dataloader,
            save_path
        )
        return
    
    if args.wandb:
        wandb.init(project='soroush', name=exp_name)

    # main training loop
    for epoch in range(init_epoch_idx, init_epoch_idx+args.epochs):        
        print_message(logger, f'At epoch {epoch}', level=1)

        for (batch_meta, batch_data) in multi_task_train_dataloader:
            batch_meta, batch_data = Collater.patch_data(
                torch.device(args.devices[0]),
                batch_meta,
                batch_data)

            task_id = batch_meta['task_id']
            model.update(batch_meta, batch_data)

            if (model.updates - 1) % (args.log_per_updates) == 0:
                print_message(logger, f"[e{epoch}] [{model.updates % n_batch_per_epoch}/{n_batch_per_epoch}] train loss: {model.train_loss.avg:.5f}")

                if args.wandb:
                    wandb.log({
                        'train/loss': model.train_loss.avg,
                        'global_step': init_global_step + model.updates,
                        'epoch': epoch
                    })
        
        model_file = save_checkpoint(model, epoch, output_dir)
        print_message(logger, f'Saving mt-dnn model to {model_file}')

if __name__ == '__main__':
    main()
