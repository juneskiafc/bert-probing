""" Fine-tuning the library models for named entity recognition on CoNLL-2003. """
import torch
import torch

from mt_dnn.batcher import Collater

def min_max_norm(matrix):
    min_ = torch.min(matrix)
    max_ = torch.max(matrix)
    norm_matrix = (matrix - min_)/(max_ - min_)
    return norm_matrix

def raw_to_final_form(raw_attention_gradients):
    # global norm
    attention_gradients = min_max_norm(raw_attention_gradients)

    # sum across training instances and layer norm
    attention_gradients = torch.sum(attention_gradients, dim=0)
    for layer in range(12):
        attention_gradients[layer, :] = min_max_norm(attention_gradients[layer, :])

    return attention_gradients

def prediction_gradient(args, model, dataloader, save_path):
    if not save_path.is_file():
        attention_gradients = torch.zeros((len(dataloader) * args.batch_size, 12, 12))

        for i, (batch_meta, batch_data) in enumerate(dataloader):
            batch_meta, batch_data = Collater.patch_data(
                torch.device(args.devices[0]),
                batch_meta,
                batch_data)
            model.get_update_gradients(batch_meta, batch_data)

            for layer in range(12):
                attention_layer = model.get_head_probe_layer(layer)
                k = attention_layer.__getattr__('self.key.weight').grad.detach()
                v = attention_layer.__getattr__('self.value.weight').grad.detach()
                raise ValueError(k.shape, v.shape)

            model.zero_grad()

        # save raw
        with open(save_path, 'w') as f:
            torch.save(attention_gradients, f)
    
    else:
        with open(save_path, 'r') as f:
            attention_gradients = torch.load(f)

    return raw_to_final_form(attention_gradients)
