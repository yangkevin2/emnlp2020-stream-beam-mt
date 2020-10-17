from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from fairseq import utils
from fairseq.models.fairseq_encoder import EncoderOut

from util import pad_mask, pad_to_length, pad_to_max_length

def select_source_indices(num_valid_beams, progress, index, max_indices, reverse=False, sort=False):
    # select source infos (starting from the least progress made) until we hit max allowed beams
    indices = torch.arange(len(index)).to(index.device)
    prog_min = progress.min()
    return indices[progress == prog_min], indices[progress != prog_min], prog_min


def reorder_incremental_state_no_model(state, reorder_idx):
    for i in range(len(state)):
        for key in state[i].keys():
            for key2 in state[i][key].keys():
                if state[i][key][key2] is not None:
                    state[i][key][key2] = state[i][key][key2].index_select(0, reorder_idx)
    return state


def reorder_incremental_state(model, reorder_idx): # because fairseq's is bugged
    for i in range(len(model.incremental_states)):
        for key in model.incremental_states[i].keys():
            for key2 in model.incremental_states[i][key].keys():
                if model.incremental_states[i][key][key2] is not None:
                    model.incremental_states[i][key][key2] = model.incremental_states[i][key][key2].index_select(0, reorder_idx)

# NOTE transformer.py:709 use self.embed_positions.weights[prev_output_tokens.shape[1] + 1 - (prev_output_tokens.cumsum(dim=1) == 0).sum(dim=1)].unsqueeze(1)
# NOTE benchmark without this change for baselines

# note this weird behavior is another reason to track our own state https://github.com/pytorch/fairseq/blob/1c8ab79ca59b466120e3df448673cab840f571ea/fairseq/modules/multihead_attention.py#L416
class IncrementalState:
    def __init__(self, initial_num_sources, k, ensemble_size, device): # it'll expand itself when it's initially none
        self.state = None
        self.master_indices = torch.LongTensor(0).to(device)
        self.k = k
        self.ensemble_size = ensemble_size
        self.num_sources = initial_num_sources
        self.decoder_keys = [set() for _ in range(ensemble_size)]
        self.encoder_keys = [set() for _ in range(ensemble_size)]
        self.device = device
    
    def append_new_incremental_state(self, num_new_sources, dummy_state, new_master_indices):
        if self.state is None:
            self.state = [{} for _ in range(self.ensemble_size)]
            for i in range(len(dummy_state)): # initialize decoder keys
                for key in dummy_state[i]:
                    self.state[i][key] = {}
                    self.state[i][key]['prev_key'] = torch.zeros(0, dummy_state[i][key]['prev_key'].shape[1], 0, dummy_state[i][key]['prev_key'].shape[3]).to(self.device)
                    self.state[i][key]['prev_value'] = torch.zeros(0, dummy_state[i][key]['prev_value'].shape[1], 0, dummy_state[i][key]['prev_value'].shape[3]).to(self.device)
                    self.state[i][key]['prev_key_padding_mask'] = torch.zeros(0, 0).to(self.device).bool()
                    if dummy_state[i][key]['prev_key'].shape[2] == 1:
                        self.decoder_keys[i].add(key) # others are encoder-side keys and we don't need to do any tricks for those
                    else:
                        self.encoder_keys[i].add(key)
                assert len(self.decoder_keys[i]) * 2 == len(dummy_state[i].keys())
                assert len(self.encoder_keys[i]) * 2 == len(dummy_state[i].keys())

        self.master_indices = torch.cat([self.master_indices, new_master_indices], dim=0)
        for i in range(len(self.state)):
            for key in self.state[i].keys():
                if key in self.decoder_keys[i]:
                    if self.state[i][key]['prev_key_padding_mask'] is None:
                        self.state[i][key]['prev_key_padding_mask'] = torch.zeros(self.state[i][key]['prev_key'].shape[0], self.state[i][key]['prev_key'].shape[2]).to(self.state[i][key]['prev_key'].device).bool()
                    self.state[i][key]['prev_key_padding_mask'] = pad_to_length(self.state[i][key]['prev_key_padding_mask'], self.state[i][key]['prev_key_padding_mask'].shape[0] + num_new_sources, 0, value=True)
                    for key2 in ['prev_key', 'prev_value']:
                        self.state[i][key][key2] = pad_to_length(self.state[i][key][key2], self.state[i][key][key2].shape[0] + num_new_sources, 0, value=0)
                else: # encoder attn keys
                    max_seq = max(self.state[i][key]['prev_key_padding_mask'].shape[1], dummy_state[i][key]['prev_key_padding_mask'].shape[1])
                    self.state[i][key]['prev_key_padding_mask'] = torch.cat([pad_to_length(self.state[i][key]['prev_key_padding_mask'], max_seq, 1, side='left', value=True).bool(), 
                                                                                pad_to_length(dummy_state[i][key]['prev_key_padding_mask'], max_seq, 1, side='left', value=True).bool()], dim=0)
                    for key2 in ['prev_key', 'prev_value']:
                        self.state[i][key][key2] = torch.cat([pad_to_length(self.state[i][key][key2], max_seq, 2, side='left', value=0), 
                                                                pad_to_length(dummy_state[i][key][key2], max_seq, 2, side='left', value=0)], dim=0)
        self.num_sources += num_new_sources
    
    def select_incremental_state(self, selected_master_indices, master_remove_indices, prog_min, return_value=True): # NOTE deletes the selected indices out of this cached state
        if len(selected_master_indices) == 0 or self.state is None:
            return torch.LongTensor(0).to(self.device), [{} for _ in range(self.ensemble_size)]
        state_indices_mask = sum([(self.master_indices == smi).long() for smi in selected_master_indices]).clamp(max=1)
        remove_indices_mask = sum([(self.master_indices == smi).long() for smi in master_remove_indices])
        selected_state_indices = state_indices_mask.nonzero().flatten()
        unselected_mask = ((1 - state_indices_mask) - remove_indices_mask).clamp(min=0)
        unselected_state_indices = unselected_mask.nonzero().flatten()
        return_indices = self.master_indices.index_select(0, selected_state_indices) if return_value else None
        self.master_indices = self.master_indices.index_select(0, unselected_state_indices)
        return_state = []
        for i in range(len(self.state)):
            return_state.append({})
            for key in self.state[i].keys():
                return_state[i][key] = {}
                for key2 in self.state[i][key].keys():
                    return_state[i][key][key2] = None
                    if self.state[i][key][key2] is not None:
                        if return_value:
                            return_state[i][key][key2] = self.state[i][key][key2].index_select(0, selected_state_indices)
                            if key in self.decoder_keys[i]:
                                if key2 == 'prev_key_padding_mask':
                                    return_state[i][key][key2] = return_state[i][key][key2][:, -prog_min:]
                                    if prog_min == 0:
                                        return_state[i][key][key2] = return_state[i][key][key2][:, :0]
                                    assert return_state[i][key][key2].shape[1] == prog_min
                                else:
                                    return_state[i][key][key2] = return_state[i][key][key2][:, :, -prog_min:]
                                    if prog_min == 0:
                                        return_state[i][key][key2] = return_state[i][key][key2][:, :, :0]
                                    assert return_state[i][key][key2].shape[2] == prog_min
                        self.state[i][key][key2] = self.state[i][key][key2].index_select(0, unselected_state_indices)
        return return_indices, return_state


    def recache(self, new_master_indices, new_state):
        if self.state is None: # for variable beam, not streaming version
            self.state = new_state
            self.master_indices = torch.cat([new_master_indices, self.master_indices], dim=0)
            for i in range(len(self.state)): # initialize decoder keys
                for key in self.state[i]:
                    if self.state[i][key]['prev_key'].shape[2] == 1:
                        self.decoder_keys[i].add(key) # others are encoder-side keys and we don't need to do any tricks for those
                assert len(self.decoder_keys[i]) * 2 == len(self.state[i].keys())
            return
        self.master_indices = torch.cat([new_master_indices, self.master_indices], dim=0)
        for i in range(len(self.state)):
            for key in self.state[i].keys():
                max_seq = max(self.state[i][key]['prev_key'].shape[2] + 1, new_state[i][key]['prev_key'].shape[2])
                for key2 in self.state[i][key].keys():
                    if self.state[i][key][key2] is not None:
                        assert new_state[i][key][key2] is not None
                        if key in self.decoder_keys[i]:
                            if key2 == 'prev_key_padding_mask':
                                self.state[i][key][key2] = pad_to_length(self.state[i][key][key2], max_seq, 1, side='left', value=True)
                                new_state[i][key][key2] = pad_to_length(new_state[i][key][key2], max_seq, 1, side='left', value=True)
                            else:
                                self.state[i][key][key2] = pad_to_length(self.state[i][key][key2], max_seq, 2, side='left', value=0)
                                new_state[i][key][key2] = pad_to_length(new_state[i][key][key2], max_seq, 2, side='left', value=0)
                        self.state[i][key][key2] = torch.cat([new_state[i][key][key2].to(self.state[i][key][key2]), self.state[i][key][key2], ], dim=0)
                        del new_state[i][key][key2]
                    else:
                        assert new_state[i][key][key2] is None


    def clean_padding(self, num_pad):
        for i in range(len(self.state)):
            for key in self.decoder_keys[i]:
                if self.state[i][key]['prev_key_padding_mask'] is not None:
                    for key2 in self.state[i][key].keys():
                        if self.state[i][key][key2] is not None:
                            if key2 == 'prev_key_padding_mask':
                                self.state[i][key][key2] = self.state[i][key][key2][:, num_pad:]
                            else:
                                self.state[i][key][key2] = self.state[i][key][key2][:, :, num_pad:]


def variable_beam_stream_fast(sg, model, tokenized_sentences, k=5, max_length=100, rp=0.6, ap=2.5, rpl=0.02, mc=3, find_top_z=1, max_indices=32, encode_batch_size=64, max_si_tokens=7168, bos_token=None, len_penalty=1, one_batch=False):
    ensemble_size = len(model.models)

    BOS_ID = sg.eos if bos_token is None else bos_token
    EOS_ID = sg.eos

    if one_batch:
        full_data_size = tokenized_sentences['net_input']['src_tokens'].shape[0]
    else:
        full_data_size = len(tokenized_sentences)
        batch_iterator = model._build_batches(tokenized_sentences, False) # not streaming
    master_done_beams = [[] for _ in range(full_data_size)]
    master_batch_ids = [None for _ in range(full_data_size)]        

    parent_model = model
    model = model.models

    master_decoded_indices = torch.zeros(1, 0, k).long().to(parent_model.device) # seq, batch, k
    master_log_probs = torch.zeros(0, k).to(parent_model.device) # batch x k
    master_enc_out = []
    master_state = IncrementalState(0, k, ensemble_size, parent_model.device) # init incremental state

    master_valid_beam_mask = torch.zeros(0, k).to(parent_model.device) # batch x k
    master_num_valid_beams = torch.zeros(0).long().to(parent_model.device) # batch
    master_index = torch.zeros(0).long().to(parent_model.device) # batch
    master_src_lengths = torch.zeros(0).long().to(parent_model.device)
    master_progress = torch.zeros(0).long().to(parent_model.device) # batch
    master_end_found = torch.zeros(0, k).long().to(parent_model.device) # batch x k
    master_done_lengths = torch.zeros(0).long().to(parent_model.device) # batch
    master_best_finished_log_probs = torch.zeros(0).to(parent_model.device) - 1e8 # batch

    current_idx = 0
    has_more_batches = True
    decode_calls = 0
    n_expansions = 0
    master_remove_indices = torch.zeros(0).long().to(parent_model.device)
    num_pad = 0
    reselect = True
    while True: 
        while has_more_batches and master_src_lengths.sum() <= max_si_tokens - parent_model.args.max_tokens: # token-based limit
            assert reselect
            if one_batch: # not streaming
                batch = tokenized_sentences
                has_more_batches = False
            else:
                try:
                    batch = next(batch_iterator)
                except StopIteration:
                    has_more_batches = False
                    break
            batch = utils.apply_to_sample(lambda t: t.to(parent_model.device), batch)
            for i, id in enumerate(batch['id'].tolist()):
                master_batch_ids[current_idx + i] = id
            net_input = batch["net_input"]
            src_tokens = net_input["src_tokens"]
            num_new_sources = len(src_tokens)

            # encode add the next batch of source infos; update the index
            encoder_outs = sg.model.forward_encoder(net_input)
            # concatenate to the current master tensors
            # decoded_indices; note these are left padded
            current_seqlen = master_decoded_indices.size(0)
            master_decoded_indices = torch.cat([master_decoded_indices, 
                                         pad_to_length(torch.zeros(1, num_new_sources, k) + BOS_ID, current_seqlen, 0, side='left', value=0).long().to(parent_model.device)],
                                        dim=1)
            # log_probs
            master_log_probs = torch.cat([master_log_probs, 
                                   torch.cat([torch.zeros(num_new_sources, 1), torch.zeros(num_new_sources, k-1) - 1e8], dim=1).to(parent_model.device)],
                                dim=0)

            if len(master_enc_out) == 0:
                assert current_idx == 0
                master_enc_out = encoder_outs
            else:
                assert len(master_enc_out) == len(encoder_outs)
                for i in range(len(master_enc_out)):
                    meo, eo = master_enc_out[i], encoder_outs[i]
                    max_seq = max(meo.encoder_out.shape[0], eo.encoder_out.shape[0])
                    new_eo = EncoderOut(encoder_out=torch.cat([pad_to_length(meo.encoder_out, max_seq, 0, side='left', value=0), pad_to_length(eo.encoder_out, max_seq, 0, side='left', value=0)], dim=1), 
                                        encoder_padding_mask=torch.cat([pad_to_length(meo.encoder_padding_mask, max_seq, 1, side='left', value=True), pad_to_length(eo.encoder_padding_mask, max_seq, 1, side='left', value=True)], dim=0),
                                        encoder_embedding=torch.cat([pad_to_length(meo.encoder_embedding, max_seq, 1, side='left', value=0), pad_to_length(eo.encoder_embedding, max_seq, 1, side='left', value=0)], dim=0),
                                        encoder_states=None,
                                        src_tokens=None,
                                        src_lengths=None)
                    master_enc_out[i] = new_eo
            if not one_batch:
                # get the encoder attention keys
                sg.model.incremental_states = [{} for _ in range(ensemble_size)]
                sg.model.forward_decoder((torch.zeros(num_new_sources)+BOS_ID).long().to(parent_model.device).unsqueeze(1), encoder_outs, sg.temperature)
                dummy_state = sg.model.incremental_states
                master_state.append_new_incremental_state(num_new_sources, dummy_state, torch.arange(num_new_sources).long().to(parent_model.device) + current_idx)

            master_valid_beam_mask = torch.cat([master_valid_beam_mask, 
                                         torch.cat([torch.ones(num_new_sources, 1), torch.zeros(num_new_sources, k-1)], dim=1).to(parent_model.device)],
                                    dim=0)
            # print(net_input['src_lengths'].max())
            master_src_lengths = torch.cat([master_src_lengths, net_input['src_lengths']], dim=0)
            # num_valid_beams
            master_num_valid_beams = torch.cat([master_num_valid_beams, torch.ones(num_new_sources).long().to(parent_model.device)], dim=0)
            # index
            master_index = torch.cat([master_index, current_idx + torch.arange(num_new_sources).to(parent_model.device)], dim=0)
            # progress
            master_progress = torch.cat([master_progress, torch.zeros(num_new_sources).long().to(parent_model.device)], dim=0)
            # end_found
            master_end_found = torch.cat([master_end_found, torch.zeros(num_new_sources, k).long().to(parent_model.device)], dim=0)
            # done lengths
            master_done_lengths = torch.cat([master_done_lengths, torch.zeros(num_new_sources).long().to(parent_model.device)], dim=0)
            # best done log probs
            master_best_finished_log_probs = torch.cat([master_best_finished_log_probs, torch.zeros(num_new_sources).to(parent_model.device) - 1e8], dim=0)
            
            current_idx += num_new_sources
            # break # for debugging
        
        # break if none left
        if not has_more_batches and len(master_index) == 0:
            break

        # based on max_bs and source_info, select which indices to use (sort source_info), then create:
        selected_indices, unselected_indices, prog_min = select_source_indices(master_num_valid_beams, master_progress, master_index, max_indices, reverse=False, sort=False)
        if one_batch:
            assert len(unselected_indices) == 0 # for debugging
        selected_master_indices = master_index[selected_indices]
        batch_size = len(selected_indices)
        selected_enc_out = sg.model.reorder_encoder_out(master_enc_out, selected_indices.unsqueeze(1).expand(-1, k).flatten())
        # if decode_calls % 50 == 0:
        #     print(decode_calls)

        valid_beam_mask = master_valid_beam_mask[selected_indices]
        valid_beam_indices = valid_beam_mask.flatten().nonzero().flatten() # idk why need to flatten again
        reverse_idx = (torch.cumsum(valid_beam_mask.flatten(), dim=0) * valid_beam_mask.flatten()).long() - 1 # it's fine to select whatever position for padding as they'll be removed later
        if num_pad > 0:
            if num_pad >= len(master_decoded_indices): # edge case: we previously ran out of beams, and we are starting fresh now
                assert num_pad == len(master_decoded_indices)
                num_pad -= 1
            master_decoded_indices = master_decoded_indices[num_pad:]
            master_state.clean_padding(num_pad)  

        if reselect:
            selected_state_master_indices, selected_state = master_state.select_incremental_state(selected_master_indices, master_remove_indices, prog_min)
            master_state.num_sources -= len(master_remove_indices)
        sg.model.incremental_states = selected_state
        log_probs = master_log_probs[selected_indices]
        progress = master_progress[selected_indices]
        decoded_indices = master_decoded_indices[-progress.max() - 1:, selected_indices, :]
        end_found = master_end_found[selected_indices]
        done_lengths = master_done_lengths[selected_indices]
        best_finished_log_probs = master_best_finished_log_probs[selected_indices]

        # flattened_indices = last_indices.flatten().unsqueeze(0) # 1 x batch*k
        # create valid beam indices from valid beam mask
        if one_batch and decode_calls == 0:
            selected_state_master_indices = master_index.clone()
        assert len(selected_state_master_indices) == len(valid_beam_indices)
        decode_calls += 1
        n_expansions += len(valid_beam_indices)

        # use valid_beam_mask to select valid indices out of decoded_indices, encoder_outs, model incremental state
        decoding_selected_indices = decoded_indices.flatten(1)[:, valid_beam_indices] # seq x selected
        selected_enc_out = sg.model.reorder_encoder_out(selected_enc_out, valid_beam_indices)

        assert torch.all(decoding_selected_indices.flatten(1).permute(1, 0)[:, 0] == 2)
        next_log_probs, _ = sg.model.forward_decoder(
                decoding_selected_indices.flatten(1).permute(1, 0)[:, :master_progress.max()+1], selected_enc_out, sg.temperature
        )

        # remake next_scores, state with dummies
        next_log_probs = next_log_probs[reverse_idx].view(1, batch_size, k, -1)
        # reorder incremental model state
        reorder_idx = reverse_idx

        next_log_probs = next_log_probs.view(1, batch_size, k, -1)

        # for edge case where EOS_ID appears later down in the beam but still needs to be dealt with correctly on the next step!
        end_found = end_found.unsqueeze(0).unsqueeze(3) # batch_size x k x 1 of whether end index is in tgt_idx already; if so, make prob of padding 1
        end_found = (end_found + (progress + 1 == max_length).long().view(1, -1, 1, 1)).clamp(max=1)
        end_found_scores = torch.zeros_like(next_log_probs).to(parent_model.device) - 1e8
        end_found_scores[:, :, :, EOS_ID] = 0 # make it so you only pick eos for the sequences that are already done, and don't duplicate them, by making other probs -inf
        next_log_probs = end_found * end_found_scores + (1 - end_found) * next_log_probs # ~ is for inverting the mask

        next_log_probs = next_log_probs - 1e8 * (1 - valid_beam_mask.unsqueeze(0).unsqueeze(3)) # get rid of padding positions
        next_log_probs = next_log_probs + log_probs.unsqueeze(0).unsqueeze(3) # 1, batch, k, vocab
        mc_probs, mc_indices = next_log_probs.topk(mc, dim=3) # 1, batch, k, mc
        top_log_probs, top_indices = mc_probs.flatten(2).topk(k, dim=2) # 1, batch, k
        mc_vocab_indices = top_indices % mc
        beam_indices = top_indices // mc # 1, batch, k
        vocab_indices = torch.gather(mc_indices.flatten(2).flatten(0, 1), 1, (mc_vocab_indices + beam_indices*mc).flatten(0, 1)).unsqueeze(0) # 1, batch, k
        # check which vocab_indices are done (in the first beam position), and add the corresponding beam to an array of done predictions
        newly_done_all = (vocab_indices == EOS_ID).long() # 1, batch, k
        newly_done = torch.cumprod(newly_done_all, dim=2) # keep on beam if there's something above it that's not done yet
        done_lengths += newly_done.sum(dim=2).flatten() # update this one before others since we'll need it earlier
        newly_done_indices = newly_done.flatten().nonzero() # batch*k
        for j in newly_done_indices:
            source_idx = j // k
            # add to some master list with an entry for each source
            if len(master_done_beams[selected_master_indices[source_idx]]) < find_top_z:
                finished_cand = decoded_indices[:, source_idx, beam_indices[0, source_idx, j % k]].flatten()
                finished_cand_length = progress[source_idx]+1
                while len(finished_cand) > 0 and finished_cand[-1] == EOS_ID:
                    finished_cand = finished_cand[:-1]
                    finished_cand_length -= 1
                if len(finished_cand) > 0: # avoid length 0
                    master_done_beams[selected_master_indices[source_idx]].append( \
                            {'tokens': finished_cand.cpu(), 'score': (top_log_probs.flatten()[j] / ((finished_cand_length)**len_penalty)).item() })
                    best_finished_log_probs[source_idx] = max(best_finished_log_probs[source_idx], top_log_probs.flatten()[j])
                else: # rarely with greedy search (beam size k = 1) you get stuff with length 0... so avoid crashing but give it low score
                    master_done_beams[selected_master_indices[source_idx]].append( \
                            {'tokens': finished_cand.cpu(), 'score': -1e8 })

        # then, shift log_probs and beam_indices for those beams and delete that beam(s); put in placeholder beam and log_prob at the k^th position
        # need to shift top_log_probs, beam_indices, vocab_indices accordingly
        top_log_probs = torch.cat([top_log_probs, torch.zeros_like(top_log_probs).to(parent_model.device) - 1e8], dim=2) # 1, batch, 2k
        shift_indices = newly_done.sum(dim=2).unsqueeze(2) + torch.arange(k).to(parent_model.device).unsqueeze(0).unsqueeze(1) # 1, batch, k
        top_log_probs = torch.gather(top_log_probs, 2, shift_indices)
        shift_indices = shift_indices.clamp(max=k-1)
        beam_indices = torch.gather(beam_indices, 2, shift_indices)
        vocab_indices = torch.gather(vocab_indices, 2, shift_indices)
        newly_done_all = torch.gather(newly_done_all, 2, shift_indices)

        log_probs = top_log_probs.squeeze(0)
        state_indices = (beam_indices + k * torch.arange(batch_size).to(parent_model.device).unsqueeze(1).repeat(1, k)).flatten()
        reorder_idx = reorder_idx[state_indices]

        # update valid beam mask
        ap_thresholds = (torch.max(log_probs[:, 0], best_finished_log_probs) - ap).unsqueeze(1) # batch x 1
        valid_beam_mask = (log_probs > ap_thresholds).float() # batch x k
        # update valid beam mask based on how many beams are left for each source
        done_mask = pad_mask(k - done_lengths, parent_model.device, max_seqlen=k).permute(1, 0) # batch x k of beams to keep, up to k - num done already
        all_low_prob_mask = 1 - valid_beam_mask.max(dim=1)[0] # NOTE since we filter out by the absolute threshold including previously finished beams, we could get < k finished candidates, but always at least 1
        found_z_mask = (all_low_prob_mask.bool() | (done_lengths >= find_top_z)).unsqueeze(1)
        valid_beam_mask = valid_beam_mask * done_mask * (1-found_z_mask.long())
        # filter the done ones out of all the master tensors
        keep_indices = (~found_z_mask).flatten().nonzero().flatten().long()
        remove_indices = (found_z_mask).flatten().nonzero().flatten().long()
        keep_indices = torch.cat([selected_indices[keep_indices], unselected_indices], dim=0)
        master_remove_indices = master_index[selected_indices[remove_indices]]

        # update these quantities in their respective source_info objects after computing them
        # just deleting/concatenating to a single master tensor
        # master_decoded_indices seq x batch x k
        new_master_indices = torch.zeros(1, master_decoded_indices.size(1), k).long().to(parent_model.device) # 1 x batch x k
        new_master_indices[:, selected_indices] = vocab_indices
        master_decoded_indices[:, selected_indices] = torch.gather(master_decoded_indices[:, selected_indices], 2, beam_indices.expand(master_decoded_indices[:, selected_indices].shape))
        master_decoded_indices = torch.cat([master_decoded_indices, new_master_indices], dim=0)
        if prog_min + 2 >= master_decoded_indices.shape[0]:
            master_decoded_indices = torch.cat([torch.zeros(1, master_decoded_indices.size(1), k).long().to(parent_model.device), master_decoded_indices], dim=0)
        master_decoded_indices[:, selected_indices] = torch.roll(master_decoded_indices[:, selected_indices], -1, 0)
        master_decoded_indices = master_decoded_indices[:-1]
        # master_log_probs batch x k
        master_log_probs[selected_indices] = log_probs
        # master_valid_beam_mask batch x k
        master_valid_beam_mask[selected_indices] = valid_beam_mask
        # master_num_valid_beams batch
        master_num_valid_beams = master_valid_beam_mask.sum(dim=1).long()
        # master_progress batch
        master_progress[selected_indices] += 1
        # master_end_found batch x k
        master_end_found[selected_indices] = (torch.gather(end_found.squeeze(3), 2, beam_indices) | newly_done_all[0, :, :]).squeeze(0)
        # master_done_lengths batch
        master_done_lengths[selected_indices] = done_lengths
        # master_best_finished_log_probs batch
        master_best_finished_log_probs[selected_indices] = best_finished_log_probs        
        # update master versions of sg.model state
        reorder_idx = reorder_idx[valid_beam_mask.flatten().nonzero().flatten()]
        selected_state_master_indices = selected_state_master_indices[reorder_idx]
        reorder_incremental_state(sg.model, reorder_idx)

        master_src_lengths = master_src_lengths[keep_indices]

        if master_src_lengths.sum() <= max_si_tokens - parent_model.args.max_tokens:
            reselect = True
        elif len(progress) < (master_progress == prog_min + 1).sum():
            reselect = True
        else:
            reselect = False
        if reselect:
            # if not one_batch:
            #     print('reselect', decode_calls)
            master_state.recache(selected_state_master_indices, sg.model.incremental_states)
        
        master_decoded_indices = master_decoded_indices[:, keep_indices, :]
        master_log_probs = master_log_probs[keep_indices]
        master_enc_out = sg.model.reorder_encoder_out(master_enc_out, keep_indices)
        master_valid_beam_mask = master_valid_beam_mask[keep_indices]
        master_num_valid_beams = master_num_valid_beams[keep_indices]
        master_index = master_index[keep_indices]
        master_progress = master_progress[keep_indices]
        master_end_found = master_end_found[keep_indices]
        master_done_lengths = master_done_lengths[keep_indices]
        master_best_finished_log_probs = master_best_finished_log_probs[keep_indices]
        
        # delete any unnecessary padding so we don't keep increasing padding
        num_pad = (master_decoded_indices.sum(dim=1).sum(dim=1) == 0).sum(dim=0)
        if not reselect:
            assert num_pad == 0

    assert all([bid is not None for bid in master_batch_ids])
    for i in range(len(master_done_beams)):
        master_done_beams[i] = sorted(master_done_beams[i], key=lambda x: x['score'], reverse=True)
    if one_batch:
        return master_done_beams, decode_calls, n_expansions
    else:
        return master_batch_ids, master_done_beams, decode_calls, n_expansions