# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import torch
import torch.nn as nn
from pretrained_models import MODEL_CLASSES
from module.dropout_wrapper import DropoutWrapper
from module.san import SANClassifier, MaskLmHeader
from module.san_model import SanModel
from module.pooler import Pooler
from torch.nn.modules.normalization import LayerNorm
from data_utils.task_def import EncoderModelType, TaskType
import tasks
from experiments.exp_def import TaskDef

def generate_decoder_opt(enable_san, max_opt):
    opt_v = 0
    if enable_san and max_opt < 2:
        opt_v = max_opt
    return opt_v

class SANBertNetwork(nn.Module):
    def __init__(self, opt):
        super(SANBertNetwork, self).__init__()
        self.dropout_list = nn.ModuleList()

        if opt['encoder_type'] not in EncoderModelType._value2member_map_:
            raise ValueError("encoder_type is out of pre-defined types")
        self.encoder_type = opt['encoder_type']
        self.preloaded_config = None

        literal_encoder_type = EncoderModelType(self.encoder_type).name.lower()
        _, model_class, _ = MODEL_CLASSES[literal_encoder_type]
        self.bert = model_class.from_pretrained(opt['init_checkpoint'], cache_dir=opt['transformer_cache'])
                    
        hidden_size = self.bert.config.hidden_size

        if opt.get('dump_feature', False):
            self.config = opt
            return
        if opt['update_bert_opt'] > 0:
            for p in self.bert.parameters():
                p.requires_grad = False

        task_def_list = opt['task_def_list']
        self.task_def_list = task_def_list
        self.decoder_opt = []
        self.task_types = []
        for task_id, task_def in enumerate(task_def_list):
            self.decoder_opt.append(generate_decoder_opt(task_def.enable_san, opt['answer_opt']))
            self.task_types.append(task_def.task_type)

        # create output header
        self.scoring_list = nn.ModuleList()
        self.dropout_list = nn.ModuleList()

        for task_id in range(len(task_def_list)):
            task_def: TaskDef = task_def_list[task_id]
            lab = task_def.n_class
            decoder_opt = self.decoder_opt[task_id]
            task_type = self.task_types[task_id]

            # dropout
            task_dropout_p = opt['dropout_p'] if task_def.dropout_p is None else task_def.dropout_p
            dropout = DropoutWrapper(task_dropout_p, opt['vb_dropout'])
            self.dropout_list.append(dropout)

            task_obj = tasks.get_task_obj(task_def)
            if task_obj is not None: 
                self.pooler = Pooler(hidden_size, dropout_p= opt['dropout_p'], actf=opt['pooler_actf'])
                out_proj = task_obj.train_build_task_layer(decoder_opt, hidden_size, lab, opt, prefix='answer', dropout=dropout)
            elif task_type == TaskType.SequenceLabeling:
                out_proj = nn.Linear(hidden_size, lab)
            elif task_type == TaskType.MaskLM:
                self.mask_lm_header = MaskLmHeader(self.bert.embeddings.word_embeddings.weight)
                out_proj = MaskLmHeader(self.bert.embeddings.word_embeddings.weight)
            else:
                if decoder_opt == 1:
                    out_proj = SANClassifier(hidden_size, hidden_size, lab, opt, prefix='answer', dropout=dropout)
                else:
                    out_proj = nn.Linear(hidden_size, lab)
            
            # hack, make sure clean
            if task_type != TaskType.MaskLM:
                self.scoring_list.append(out_proj)
            else:
                delattr(self, 'scoring_list')
            
        self.config = opt

    def get_attention_layer(self, layer):
        # index in to encoder
        _, encoder_module = list(self.bert.named_children())[1]

        # attach the head probing linear layer
        _, all_hidden_layers = list(encoder_module.named_children())[0] # contains all encoder attention layers
        hidden_layer_to_probe = all_hidden_layers[layer]

        for (bert_sublayer_name, bert_sublayer) in hidden_layer_to_probe.named_children():
            if bert_sublayer_name == 'attention':
                _, self_attention_layer = list(bert_sublayer.named_children())[0]
                assert self_attention_layer.__class__.__name__ == 'BertSelfAttention', self_attention_layer.__class__.__name__
                return self_attention_layer
    
    def attach_head_probe(self, attention_layer_to_probe, head_idx_to_probe, n_classes, sequence, device):
        layer = self.get_attention_layer(attention_layer_to_probe)
        layer.attach_head_probe(head_idx_to_probe, n_classes, sequence, device)
    
    def detach_head_probe(self, hl):
        layer = self.get_attention_layer(hl)
        layer.detach_head_probe()

    def get_pooler_layer(self):
        _, pooler = list(self.bert.named_children())[2]
        return pooler
    
    def attach_model_probe(self, n_classes, device, sequence=False):
        pooler = self.get_pooler_layer()
        pooler.attach_model_probe(n_classes, device, sequence)

    def embed_encode(self, input_ids, token_type_ids=None, attention_mask=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        embedding_output = self.bert.embeddings(input_ids, token_type_ids)
        return embedding_output

    def encode(self, input_ids, token_type_ids, attention_mask, inputs_embeds=None, y_input_ids=None, output_hidden_states=False):
        if self.encoder_type == EncoderModelType.T5:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds)
            last_hidden_state = outputs.last_hidden_state
            all_hidden_states = outputs.hidden_states # num_layers + 1 (embeddings)
        elif self.encoder_type == EncoderModelType.T5G:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=y_input_ids)
            # return logits from LM header
            last_hidden_state = outputs.logits
            all_hidden_states = outputs.encoder_last_hidden_state # num_layers + 1 (embeddings)
        else:
            outputs = self.bert(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_hidden_states=output_hidden_states
            )

            last_hidden_state = outputs.last_hidden_state
            all_hidden_states = outputs.hidden_states 
            head_probe_output = outputs.head_probe_output
            model_probe_output = outputs.model_probe_output

        return last_hidden_state, all_hidden_states, head_probe_output, model_probe_output

    def forward(self, input_ids, token_type_ids, attention_mask, premise_mask=None, hyp_mask=None, task_id=0, y_input_ids=None, fwd_type=0, embed=None, model_probe=False, head_probe=False):        
        assert fwd_type == 0, fwd_type
        encode_outputs = self.encode(
                            input_ids,
                            token_type_ids,
                            attention_mask,
                            y_input_ids=y_input_ids,
                            output_hidden_states=True
                        )
        
        last_hidden_state, all_hidden_states, head_probe_output, model_probe_output = encode_outputs

        decoder_opt = self.decoder_opt[task_id]
        task_type = self.task_types[task_id]
        task_obj = tasks.get_task_obj(self.task_def_list[task_id])

        if model_probe:
            if task_type == TaskType.SequenceLabeling:
                model_probe_output = model_probe_output.contiguous().view(-1, model_probe_output.size(2))
            return model_probe_output

        elif head_probe:
            if task_type == TaskType.SequenceLabeling:
                head_probe_output = head_probe_output.contiguous().view(-1, head_probe_output.size(2))
            return head_probe_output
    
        if task_obj is not None: # Classification
            pooled_output = self.pooler(last_hidden_state)
            logits = task_obj.train_forward(last_hidden_state,
                                            pooled_output,
                                            premise_mask,
                                            hyp_mask,
                                            decoder_opt,
                                            self.dropout_list[task_id],
                                            self.scoring_list[task_id])
            return logits

        elif task_type == TaskType.Span:
            assert decoder_opt != 1
            last_hidden_state = self.dropout_list[task_id](last_hidden_state)
            logits = self.scoring_list[task_id](last_hidden_state)
            start_scores, end_scores = logits.split(1, dim=-1)
            start_scores = start_scores.squeeze(-1)
            end_scores = end_scores.squeeze(-1)
            return start_scores, end_scores
        elif task_type == TaskType.SpanYN:
            assert decoder_opt != 1
            last_hidden_state = self.dropout_list[task_id](last_hidden_state)
            logits = self.scoring_list[task_id](last_hidden_state)
            start_scores, end_scores = logits.split(1, dim=-1)
            start_scores = start_scores.squeeze(-1)
            end_scores = end_scores.squeeze(-1)
            return start_scores, end_scores
        elif task_type == TaskType.SequenceLabeling:
            pooled_output = last_hidden_state
            pooled_output = self.dropout_list[task_id](pooled_output)
            pooled_output = pooled_output.contiguous().view(-1, pooled_output.size(2))
            logits = self.scoring_list[task_id](pooled_output)
            return logits
        elif task_type == TaskType.MaskLM:
            last_hidden_state = self.dropout_list[task_id](last_hidden_state)
            logits = self.mask_lm_header(last_hidden_state)
            return logits, head_probe_output
        elif task_type == TaskType.SeqenceGeneration:
            logits = last_hidden_state.view(-1, last_hidden_state.size(-1))
            return logits
        else:
            if decoder_opt == 1:
                max_query = hyp_mask.size(1)
                assert max_query > 0
                assert premise_mask is not None
                assert hyp_mask is not None
                hyp_mem = last_hidden_state[:, :max_query, :]
                logits = self.scoring_list[task_id](last_hidden_state, hyp_mem, premise_mask, hyp_mask)
            else:
                raise ValueError
                pooled_output = self.dropout_list[task_id](pooled_output)
                logits = self.scoring_list[task_id](pooled_output)
            return logits