import random

import torch

def pad_to_max_length(t1, t2, dim, side='right'):
    if t1.size(dim) < t2.size(dim):
        t1 = pad_to_length(t1, t2.size(dim), dim, side)
    elif t2.size(dim) < t1.size(dim):
        t2 = pad_to_length(t2, t1.size(dim), dim, side)
    return t1, t2


def pad_to_length(tensor, length, dim, side='right', value=0):
    assert side in ['left', 'right']
    assert tensor.size(dim) <= length
    if tensor.size(dim) == length:
        return tensor
    else:
        zeros_shape = list(tensor.shape)
        zeros_shape[dim] = length - tensor.size(dim)
        zeros_shape = tuple(zeros_shape)
        if side == 'right':
            return torch.cat([tensor, torch.zeros(zeros_shape).type(tensor.type()).to(tensor.device).fill_(value)], dim=dim)
        else:
            return torch.cat([torch.zeros(zeros_shape).type(tensor.type()).to(tensor.device).fill_(value), tensor], dim=dim)


def pad_mask(lengths: torch.LongTensor, device='cuda', max_seqlen=None) -> torch.ByteTensor:
    # lengths: bs. Ex: [2, 3, 1]
    if max_seqlen is None:
        max_seqlen = torch.max(lengths)
    expanded_lengths = lengths.unsqueeze(0).repeat((max_seqlen, 1))  # [[2, 3, 1], [2, 3, 1], [2, 3, 1]]
    indices = torch.arange(max_seqlen).unsqueeze(1).repeat((1, lengths.size(0))).to(device)  # [[0, 0, 0], [1, 1, 1], [2, 2, 2]]

    return expanded_lengths > indices  # pad locations are 0. #[[1, 1, 1], [1, 1, 0], [0, 1, 0]]. seqlen x bs