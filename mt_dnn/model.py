# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
from typing import List
import copy
import torch
import tasks
import logging
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import *
from data_utils.utils import AverageMeter
from pytorch_pretrained_bert import BertAdam as Adam
from module.bert_optim import Adamax, RAdam
from mt_dnn.loss import LOSS_REGISTRY
from mt_dnn.matcher import SANBertNetwork
from mt_dnn.perturbation import SmartPerturbation
from mt_dnn.loss import *
from experiments.exp_def import TaskDef


logger = logging.getLogger(__name__)


class MTDNNModel(object):
    def __init__(self, opt, state_dict=None, num_train_step=-1):
        self.config = opt

        self.updates = state_dict['updates'] if state_dict and 'updates' in state_dict else 0
        self.local_updates = 0

        self.train_loss = AverageMeter()
        self.adv_loss = AverageMeter()
        self.emb_val =  AverageMeter()
        self.eff_perturb = AverageMeter()
        self.initial_from_local = True if state_dict else False

        model = SANBertNetwork(opt, initial_from_local=self.initial_from_local)
        self.network = model
        if state_dict:
            self.load_state_dict(state_dict)

        if len(opt['devices']) > 0:
            self.device = opt['devices'][0]
            self.network.to(self.device)
        else:
            self.device = None

        self.total_param = sum([p.nelement() for p in model.parameters() if p.requires_grad])        
        optimizer_parameters = self._get_param_groups()
        self._setup_optim(optimizer_parameters, state_dict, num_train_step)
        self.optimizer.zero_grad()

        self.head_probe = opt['head_probe']
        n_classes = opt['task_def_list'][0].n_class
        if self.head_probe:
            for hl in range(12):
                for hi in range(12):
                    self.attach_head_probe(hl, hi, n_classes)

        if len(opt['devices']) > 0:
            logger.info(f'Using data parallel: {opt["devices"]}')
            self.mnetwork = nn.DataParallel(self.network, device_ids=opt['devices'])
        else:
            self.mnetwork = self.network

        self._setup_lossmap(self.config)
        self._setup_kd_lossmap(self.config)
        self._setup_adv_lossmap(self.config)
        self._setup_adv_training(self.config)
        self._setup_tokenizer()

    def load_state_dict(self, state_dict):
        if self.config['head_probe']:
            state_dict['state']['scoring_list.0.weight'] = self.network.state_dict()['scoring_list.0.weight']
            state_dict['state']['scoring_list.0.bias'] = self.network.state_dict()['scoring_list.0.bias']
        
        self.network.load_state_dict(state_dict['state'], strict=True)
    
    def get_head_probe_layer(self, hl):
        return self.network.get_attention_layer(hl)
    
    def attach_head_probe(self, hl, hi, n_classes):
        self.network.attach_head_probe(hl, hi, n_classes)

    def detach_head_probe(self, hl):
        self.network.detach_head_probe(hl)

    def _setup_adv_training(self, config):
        self.adv_teacher = None
        if config.get('adv_train', False):
            self.adv_teacher = SmartPerturbation(config['adv_epsilon'],
                    config['multi_gpu_on'],
                    config['adv_step_size'],
                    config['adv_noise_var'],
                    config['adv_p_norm'],
                    config['adv_k'],
                    config['fp16'],
                    config['encoder_type'],
                    loss_map=self.adv_task_loss_criterion,
                    norm_level=config['adv_norm_level'])

    def _get_param_groups(self):
        no_decay = ['bias', 'gamma', 'beta', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in self.network.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.network.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        return optimizer_parameters

    def _setup_optim(self, optimizer_parameters, state_dict=None, num_train_step=-1):
        if self.config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(optimizer_parameters, self.config['learning_rate'],
                                       weight_decay=self.config['weight_decay'])

        elif self.config['optimizer'] == 'adamax':
            self.optimizer = Adamax(optimizer_parameters,
                                    self.config['learning_rate'],
                                    warmup=self.config['warmup'],
                                    t_total=num_train_step,
                                    max_grad_norm=self.config['grad_clipping'],
                                    schedule=self.config['warmup_schedule'],
                                    weight_decay=self.config['weight_decay'])
            if self.config.get('have_lr_scheduler', False): self.config['have_lr_scheduler'] = False
        elif self.config['optimizer'] == 'radam':
            self.optimizer = RAdam(optimizer_parameters,
                                    self.config['learning_rate'],
                                    warmup=self.config['warmup'],
                                    t_total=num_train_step,
                                    max_grad_norm=self.config['grad_clipping'],
                                    schedule=self.config['warmup_schedule'],
                                    eps=self.config['adam_eps'],
                                    weight_decay=self.config['weight_decay'])
            if self.config.get('have_lr_scheduler', False): self.config['have_lr_scheduler'] = False
            # The current radam does not support FP16.
            self.config['fp16'] = False
        elif self.config['optimizer'] == 'adam':
            self.optimizer = Adam(optimizer_parameters,
                                  lr=self.config['learning_rate'],
                                  warmup=self.config['warmup'],
                                  t_total=num_train_step,
                                  max_grad_norm=self.config['grad_clipping'],
                                  schedule=self.config['warmup_schedule'],
                                  weight_decay=self.config['weight_decay'])
            if self.config.get('have_lr_scheduler', False): self.config['have_lr_scheduler'] = False
        else:
            raise RuntimeError('Unsupported optimizer: %s' % self.config['optimizer'])

        if state_dict and 'optimizer' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])

        if self.config.get('have_lr_scheduler', False):
            if self.config.get('scheduler_type', 'rop') == 'rop':
                self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=self.config['lr_gamma'], patience=3)
            elif self.config.get('scheduler_type', 'rop') == 'exp':
                self.scheduler = ExponentialLR(self.optimizer, gamma=self.config.get('lr_gamma', 0.95))
            else:
                milestones = [int(step) for step in self.config.get('multi_step_lr', '10,20,30').split(',')]
                self.scheduler = MultiStepLR(self.optimizer, milestones=milestones, gamma=self.config.get('lr_gamma'))
        else:
            self.scheduler = None

    def _setup_lossmap(self, config):
        task_def_list: List[TaskDef] = config['task_def_list']
        self.task_loss_criterion = []
        for idx, task_def in enumerate(task_def_list):
            cs = task_def.loss
            lc = LOSS_REGISTRY[cs](name='Loss func of task {}: {}'.format(idx, cs))
            self.task_loss_criterion.append(lc)

    def _setup_kd_lossmap(self, config):
        task_def_list: List[TaskDef] = config['task_def_list']
        self.kd_task_loss_criterion = []
        if config.get('mkd_opt', 0) > 0:
            for idx, task_def in enumerate(task_def_list):
                cs = task_def.kd_loss
                assert cs is not None
                lc = LOSS_REGISTRY[cs](name='KD Loss func of task {}: {}'.format(idx, cs))
                self.kd_task_loss_criterion.append(lc)

    def _setup_adv_lossmap(self, config):
        task_def_list: List[TaskDef] = config['task_def_list']
        self.adv_task_loss_criterion = []
        if config.get('adv_train', False):
            for idx, task_def in enumerate(task_def_list):
                cs = task_def.adv_loss
                assert cs is not None
                lc = LOSS_REGISTRY[cs](name='Adv Loss func of task {}: {}'.format(idx, cs))
                self.adv_task_loss_criterion.append(lc)
    
    def _setup_tokenizer(self):
        try:
            from transformers import AutoTokenizer 
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config['init_checkpoint'],
                cache_dir=self.config['transformer_cache'])
        except:
            self.tokenizer = None
        
    def _to_cuda(self, tensor):
        if tensor is None: return tensor

        if isinstance(tensor, list) or isinstance(tensor, tuple):
            y = [e.to(self.device) for e in tensor]
            for e in y:
                e.requires_grad = False
        else:
            y = tensor.to(self.device)
            y.requires_grad = False
        return y
    
    def update(self, batch_meta, batch_data):
        self.network.train()

        # we want complete eval mode when head probing,
        # except the actual heads.
        # the heads are linear layers so it doesn't matter whether they are train() or eval().
        if self.head_probe:
            self.network.eval()

        y = batch_data[batch_meta['label']]
        if self.device is not None:
            y = self._to_cuda(y)

        task_id = batch_meta['task_id']
        inputs = batch_data[:batch_meta['input_len']]
        if len(inputs) == 3:
            inputs.append(None)
            inputs.append(None)
        inputs.append(task_id)

        if 'y_token_id' in batch_meta:
            inputs.append(batch_data[batch_meta['y_token_id']])

        weight = None
        if self.config.get('weighted_on', False):
            if self.config['cuda']:
                weight = batch_data[batch_meta['factor']].cuda(non_blocking=True)
            else:
                weight = batch_data[batch_meta['factor']]

        # fw to get logits
        logits, head_probe_outputs = self.mnetwork(*inputs)

        # compute loss
        loss_criterion = self.task_loss_criterion[task_id]
        if loss_criterion and (y is not None):
            if not self.head_probe:
                loss = [loss_criterion(logits, y, weight, ignore_index=-1)]
            else:
                # register separate loss for each head
                loss = []
                for layer in range(12):
                    for head in range(12):
                        logits_for_head = head_probe_outputs[layer][:, head, :]
                        loss_for_head = loss_criterion(logits_for_head, y, weight, ignore_index=-1)
                        loss.append(loss_for_head)

        batch_size = batch_data[batch_meta['token_id']].size(0)

        if self.config['bin_on']:
            for i in range(len(loss)):
                loss[i] = loss[i] * (1.0 * batch_size / self.config['batch_size'])
        
        if self.config['local_rank'] != -1:
            raise ValueError('local rank != -1 not supported')
            copied_loss = copy.deepcopy(loss.data)
            torch.distributed.all_reduce(copied_loss)
            copied_loss = copied_loss / self.config['world_size'] # ddp
            self.train_loss.update(copied_loss.item(), batch_size)
        else:
            # take the average of all head probe losses.
            for i in range(len(loss)):
                self.train_loss.update(loss[i].item(), batch_size)
            
        # scale loss
        for i in range(len(loss)):
            loss[i] = loss[i] / self.config.get('grad_accumulation_step', 1)

        if self.head_probe:
            for hl in range(12):
                for hi in range(12):
                    loss_for_head = loss[12*hl+hi]
                    loss_for_head.backward(retain_graph=True)

        self.local_updates += 1
        if self.local_updates % self.config.get('grad_accumulation_step', 1) == 0:
            if self.config['global_grad_clipping'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.config['global_grad_clipping'])
            self.updates += 1
            self.optimizer.step()
            self.optimizer.zero_grad()

    def encode(self, batch_meta, batch_data):
        self.network.eval()
        inputs = batch_data[:3]
        sequence_output = self.network.encode(*inputs)[0]
        return sequence_output

    def extract(self, batch_meta, batch_data):
        self.network.eval()
        # 'token_id': 0; 'segment_id': 1; 'mask': 2
        inputs = batch_data[:3]
        all_encoder_layers, pooled_output = self.mnetwork.bert(*inputs)
        return all_encoder_layers, pooled_output

    def predict(self, batch_meta, batch_data, head_probe=False):
        self.network.eval()

        task_id = batch_meta['task_id']
        task_def = TaskDef.from_dict(batch_meta['task_def'])
        task_type = task_def.task_type
        task_obj = tasks.get_task_obj(task_def)

        inputs = batch_data[:batch_meta['input_len']]
        if len(inputs) == 3:
            inputs.append(None)
            inputs.append(None)
        inputs.append(task_id)

        score, head_probe_logits = self.mnetwork(*inputs)
        if head_probe:
            score = head_probe_logits

        if task_obj is not None:
            score, predict = task_obj.test_predict(score)
        else:
            raise ValueError("Unknown task_type: %s" % task_type)

        return score, predict, batch_meta['label']

    def save(self, filename):
        model = self.network
        network_state = dict([(k, v.cpu()) for k, v in model.state_dict().items()])

        params = {
            'state': network_state,
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
        }
        torch.save(params, filename)
        logger.info('model saved to {}'.format(filename))

    def load(self, checkpoint, load_optimizer=True):
        model_state_dict = torch.load(checkpoint)
        if 'state' in model_state_dict:
            self.network.load_state_dict(model_state_dict['state'], strict=True)
        if 'optimizer' in model_state_dict and load_optimizer:
            self.optimizer.load_state_dict(model_state_dict['optimizer'])
        if 'config' in model_state_dict:
            self.config.update(model_state_dict['config'])

    def cuda(self):
        self.network.cuda()