import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

def predict_greedy(sg, model, batch, max_len=100, bos_token=None):
    BOS_ID = sg.eos if bos_token is None else bos_token

    net_input = batch["net_input"]
    src_tokens = net_input["src_tokens"]
    batch_size = len(src_tokens)

    # encoder_output, encoder_mask, initial_hidden = model.encode(source)
    encoder_outs = sg.model.forward_encoder(net_input)
    encoder_outs = sg.model.reorder_encoder_out(encoder_outs, torch.arange(batch_size).to(src_tokens))
    decoded_indices = torch.zeros(1, batch_size).long().to(src_tokens) + BOS_ID
    master_idx = torch.arange(batch_size).to(src_tokens)

    preds = [[] for _ in range(batch_size)]

    decode_calls = 0
    for step in range(max_len + 1):
        decode_calls += 1
        lprobs, _ = sg.model.forward_decoder(
                decoded_indices.permute(1, 0)[:, :step+1], encoder_outs, sg.temperature
        )
        lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)
        lprobs[:, sg.pad] = -math.inf  # never select pad
        lprobs[:, sg.unk] -= sg.unk_penalty  # apply unk penalty
        if step >= max_len:
            lprobs[:, : sg.eos] = -math.inf
            lprobs[:, sg.eos + 1 :] = -math.inf
        if step < sg.min_len:
            # minimum length constraint (does not apply if using prefix_tokens)
            lprobs[:, sg.eos] = -math.inf

        # lprobs is batch*beam x vocab
        _, next_indices = lprobs.max(dim=1)
        decoded_indices = torch.cat([decoded_indices, next_indices.unsqueeze(0)], dim=0)
        
        end_found = (next_indices == sg.eos)
        for idx in (end_found.long()).nonzero().flatten():
            preds[master_idx[idx]].append({'tokens': decoded_indices[:, idx]})
        keep_idx = (1-end_found.long()).nonzero().flatten()
        master_idx = master_idx[keep_idx]
        decoded_indices = decoded_indices[:, keep_idx]
        encoder_outs = sg.model.reorder_encoder_out(encoder_outs, keep_idx)
        sg.model.reorder_incremental_state(keep_idx)

        if len(master_idx) == 0:
            break

    return preds, decode_calls, 0