# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
from typing import List
import torch
from data_utils.task_def import TaskType
import tasks
import numpy as np
import logging
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import *
from data_utils.utils import AverageMeter
from pytorch_pretrained_bert import BertAdam as Adam
from module.bert_optim import Adamax, RAdam
from mt_dnn.loss import LOSS_REGISTRY
from mt_dnn.matcher import SANBertNetwork
from mt_dnn.loss import *
from experiments.exp_def import TaskDef
import einops

logger = logging.getLogger(__name__)


class MTDNNModel(object):
    def __init__(self, opt, devices=None, state_dict=None, num_train_step=-1):
        self.config = opt
        self.devices = devices

        # global step
        self.updates = state_dict['updates'] if state_dict and 'updates' in state_dict else 0
        self.local_updates = 0

        # create blank slate BERT
        model = SANBertNetwork(opt)
        self.network = model
        if state_dict:
            self.load_state_dict(state_dict)

        # multi GPU, single, or CPU
        if devices is not None:
            self.device = devices[0]
            self.network.to(self.device)
        
        if self.config['multi_gpu_on']:
            self.mnetwork = nn.DataParallel(self.network, device_ids=devices)
        else:
            self.mnetwork = self.network
        
        # setup optimizer
        optimizer_parameters = self._get_param_groups()
        self._setup_optim(optimizer_parameters, state_dict, num_train_step)

        self._setup_lossmap(self.config)
        self._setup_tokenizer()

        # stats and misc
        self.total_param = sum([p.nelement() for p in model.parameters() if p.requires_grad])
        self.train_loss = AverageMeter()

        self.model_probe = opt.get('model_probe', False)
        self.head_probe = opt.get('head_probe', False)

    def load_state_dict(self, state_dict):
        self.network.load_state_dict(state_dict['state'], strict=True)
    
    def get_head_probe_layer(self, hl):
        return self.network.get_attention_layer(hl)
    
    def attach_head_probe(self, hl, hi, n_classes, sequence=False):
        self.head_probe = True
        self.network.attach_head_probe(hl, hi, n_classes, sequence, self.device)

    def detach_head_probe(self, hl):
        self.head_probe = False
        self.network.detach_head_probe(hl)
    
    def attach_model_probe(self, n_classes, sequence):
        self.model_probe = True
        self.network.attach_model_probe(n_classes, self.device, sequence=sequence)

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
        self.warmup_steps = self.config['warmup'] * num_train_step
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
        
        self.optimizer.zero_grad()

    def _setup_lossmap(self, config):
        task_def_list: List[TaskDef] = config['task_def_list']
        self.task_loss_criterion = []
        for idx, task_def in enumerate(task_def_list):
            cs = task_def.loss
            lc = LOSS_REGISTRY[cs](name='Loss func of task {}: {}'.format(idx, cs))
            self.task_loss_criterion.append(lc)
    
    def _setup_tokenizer(self):
        try:
            from transformers import AutoTokenizer 
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config['init_checkpoint'],
                cache_dir=self.config['transformer_cache'])
        except:
            self.tokenizer = None

    def __call__(self, inputs):
        logits, head_probe_logits = self.mnetwork(*inputs)
    
    def get_update_gradients(self, batch_meta, batch_data):
        self.network.train()
        y = batch_data[batch_meta['label']]

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

        logits = self.mnetwork(*inputs, model_probe=False, head_probe=False)

        # compute loss
        loss = 0
        loss_criterion = self.task_loss_criterion[task_id]
        if loss_criterion and (y is not None):
            y.to(logits.device)
            if len(logits.shape) > 2:
                # sequence output, combine seq w/ batch
                logits = einops.rearrange(logits, 'b s c -> (b s) c')
            loss = loss_criterion(logits, y, weight, ignore_index=-1)

        batch_size = batch_data[batch_meta['token_id']].size(0)
        self.train_loss.update(loss.item(), batch_size)
        
        loss = loss / self.config.get('grad_accumulation_step', 1)
        loss.backward()


    def update(self, batch_meta, batch_data):
        self.network.train()
        y = batch_data[batch_meta['label']]

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
        logits = self.mnetwork(*inputs, model_probe=self.model_probe, head_probe=self.head_probe)

        # compute loss
        loss = 0
        loss_criterion = self.task_loss_criterion[task_id]
        if loss_criterion and (y is not None):
            y.to(logits.device)
            loss = loss_criterion(logits, y, weight, ignore_index=-1)

        batch_size = batch_data[batch_meta['token_id']].size(0)
        self.train_loss.update(loss.item(), batch_size)
        
        loss = loss / self.config.get('grad_accumulation_step', 1)
        loss.backward()

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

    # TODO: similar as function extract, preserve since it is used by extractor.py
    # will remove after migrating to transformers package
    def extract(self, batch_meta, batch_data):
        self.network.eval()
        # 'token_id': 0; 'segment_id': 1; 'mask': 2
        inputs = batch_data[:3]
        all_encoder_layers, pooled_output = self.mnetwork.bert(*inputs)
        return all_encoder_layers, pooled_output

    def predict(self, batch_meta, batch_data, head_probe=False, model_probe=False):
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

        if model_probe:
            self.mnetwork.task_types[task_id] = task_def.task_type
                
        score = self.mnetwork(
            *inputs,
            model_probe=model_probe,
            head_probe=head_probe
        )
    
        if task_obj is not None:
            score, predict = task_obj.test_predict(score)
        elif task_type == TaskType.SequenceLabeling:
            mask = batch_data[batch_meta["mask"]]
            score = score.contiguous()
            score = score.data.cpu()
            score = score.numpy()
            predict = np.argmax(score, axis=1).reshape(mask.size()).tolist()
            valid_length = mask.sum(1).tolist()
            final_predict = []

            for idx, p in enumerate(predict):
                final_predict.append(p[: valid_length[idx]])

            score = score.reshape(-1).tolist()
            return score, final_predict, batch_meta["label"]
        else:
            raise ValueError("Unknown task_type: %s" % task_type)
        
        return score, predict, batch_meta['label']

    def save(self, filename):
        if isinstance(self.mnetwork, torch.nn.parallel.DistributedDataParallel):
            model = self.mnetwork.module
        else:
            model = self.network
        network_state = dict([(k, v.cpu()) for k, v in model.state_dict().items()])
        if self.head_probe:
            state_to_save = {}
            for k, v in network_state.items():
                if 'head_probe' in k:
                    state_to_save[k] = v
            params = {'state': state_to_save}
        else:
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
            self.network.load_state_dict(model_state_dict['state'], strict=False)
        if 'optimizer' in model_state_dict and load_optimizer:
            self.optimizer.load_state_dict(model_state_dict['optimizer'])
        if 'config' in model_state_dict:
            self.config.update(model_state_dict['config'])

    def cuda(self):
        self.network.cuda()