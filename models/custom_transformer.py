import math
from typing import Tuple
from models.base_model import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from utils.generation_utils import BeamHypotheses, top_k_top_p_filtering
from utils.quantize import MIDI_NOTES, DURATIONS, GAPS


class CustomModelTransformer(BaseModel):

    def __init__(self, encoder, device, SOS_token=0, EOS_token=1, MAX_LENGTH=100, dropout_p=0.1, train_encoder=False, expansion_factor=4, num_heads=8, num_layers=2):
        super().__init__()
        self.device = device
        self.MAX_LENGTH = MAX_LENGTH
        self.SOS_token = SOS_token
        self.EOS_token = EOS_token

        # Take pretrained encoder
        self.encoder = encoder

        # Define decoders
        # self.decoder = CustomTransformerDecoderMulti(
        #     hidden_size=encoder.config.n_positions,
        #     output_size_note=len(MIDI_NOTES)+2,
        #     output_size_duration=len(DURATIONS)+2,
        #     output_size_gap=len(GAPS)+2,
        #     MAX_LENGTH=MAX_LENGTH,
        #     dropout_p=dropout_p,
        #     device=device,
        #     num_layers=num_layers,
        #     expansion_factor=expansion_factor,
        #     n_heads=num_heads
        # )
        self.note_decoder = CustomTransformerDecoderSingle(
            hidden_size=encoder.config.n_positions,
            output_size=len(MIDI_NOTES)+2,
            MAX_LENGTH=MAX_LENGTH,
            dropout_p=dropout_p,
            device=device,
            num_layers=num_layers,
            expansion_factor=expansion_factor,
            n_heads=num_heads
        )
        self.duration_decoder = CustomTransformerDecoderSingle(
            hidden_size=encoder.config.n_positions,
            output_size=len(DURATIONS)+2,
            MAX_LENGTH=MAX_LENGTH,
            dropout_p=dropout_p,
            device=device,
            num_layers=num_layers,
            expansion_factor=expansion_factor,
            n_heads=num_heads
        )
        self.gap_decoder = CustomTransformerDecoderSingle(
            hidden_size=encoder.config.n_positions,
            output_size=len(GAPS)+2,
            MAX_LENGTH=MAX_LENGTH,
            dropout_p=dropout_p,
            device=device,
            num_layers=num_layers,
            expansion_factor=expansion_factor,
            n_heads=num_heads
        )


        self.train_encoder = train_encoder

        self.encoder.to(device)
        self.note_decoder.to(device)
        self.duration_decoder.to(device)
        self.gap_decoder.to(device)
        #self.decoder.to(device)

    def make_trg_mask(self, trg):
        """
        Args:
            trg: target sequence
        Returns:
            trg_mask: target mask
        """
        trg_len = trg.shape[1]
        # returns the lower triangular part of matrix filled with ones
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).to(self.device)
        return trg_mask   
    
    def forward(self, x, attn, target):
        if not self.train_encoder:
            with torch.no_grad():
                encoder_output = self.encoder(x, attn, output_hidden_states=True, output_attentions=False)
        else:
            encoder_output = self.encoder(x, attn, output_hidden_states=True, output_attentions=False)
        
        target_mask = self.make_trg_mask(target)
        note = self.note_decoder(target[:, :, 0], encoder_output[0], target_mask)
        duration = self.duration_decoder(target[:, :, 1], encoder_output[0], target_mask)
        gap = self.gap_decoder(target[:, :, 2], encoder_output[0], target_mask)
        # note, duration, gap = self.decoder(target, encoder_output[0], target_mask)
        return note, duration, gap    
    
    def generate(self, x, attn, do_sample=False, max_length=100, temperature=1.0, top_k=50, top_p=1.0, num_return_sequences=1, early_stopping=False, num_beams=1, min_length=10, use_cache=True, length_penalty=1.0):
        # target = torch.zeros(x.shape[0], 1, 3, dtype=torch.int64).to(self.device)
        target_note = torch.zeros((x.shape[0], 1), dtype=torch.int64).to(self.device)
        target_duration = torch.zeros((x.shape[0], 1), dtype=torch.int64).to(self.device)
        target_gap = torch.zeros((x.shape[0], 1), dtype=torch.int64).to(self.device)

        out_notes, out_durations, out_gaps = [], [], []
        logits_notes, logits_durations, logits_gaps = [], [], []
        batch_size = x.shape[0]

        # set effective batch size and effective batch multiplier according to do_sample
        # https://github.com/huggingface/transformers/blob/c4d4e8bdbd25d9463d41de6398940329c89b7fb6/src/transformers/generation_utils.py
        if do_sample:
            effective_batch_size = batch_size * num_return_sequences
            effective_batch_mult = num_return_sequences
        else:
            effective_batch_size = batch_size
            effective_batch_mult = 1
        
        # Get encoder output
        encoder_output = self.encoder(x, attn, output_hidden_states=True, output_attentions=False)

        if num_return_sequences > 1 or num_beams > 1:
            input_ids_len = x.shape[-1]
            input_ids = x.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, input_ids_len)
            attention_mask = attn.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, input_ids_len)

            input_ids = input_ids.contiguous().view(effective_batch_size * num_beams, input_ids_len)
            attention_mask = attention_mask.contiguous().view(effective_batch_size * num_beams, input_ids_len)
        
        input_ids = torch.full(
            (effective_batch_size * num_beams, 1),
            self.SOS_token,
            dtype=torch.long,
            device=self.device,
        )
        cur_len = 1

        expanded_batch_idxs = (
                torch.arange(batch_size)
                .view(-1, 1)
                .repeat(1, num_beams * effective_batch_mult)
                .view(-1)
                .to(input_ids.device)
            )
        # expand encoder_outputs
        encoder_outputs = (encoder_outputs[0].index_select(0, expanded_batch_idxs), *encoder_outputs[1:])

        if num_beams > 1:
            output = self._generate_beam_search(
                input_ids,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                early_stopping=early_stopping,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                use_cache=use_cache,
                attention_mask=attention_mask,
                length_penalty=length_penalty,
                batch_size=batch_size
            )
        else:
            output = self._generate_no_beam_search(
                input_ids,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                use_cache=use_cache,
                attention_mask=attention_mask,

            )

        return output
        # for _ in range(self.MAX_LENGTH-1): # -1 to account for the SOS token
        #     # note, duration, gap = self.decoder(target, encoder_output[0], target_mask)
        #     target_mask = self.make_trg_mask(target_note)
        #     note = self.note_decoder(target_note, encoder_output[0], target_mask)
        #     duration = self.duration_decoder(target_duration, encoder_output[0], target_mask)
        #     gap = self.gap_decoder(target_gap, encoder_output[0], target_mask)

        #     last_note = note[:, -1, :].unsqueeze(1)
        #     last_duration = duration[:, -1, :].unsqueeze(1)
        #     last_gap = gap[:, -1, :].unsqueeze(1)

        #     logits_notes.append(last_note)
        #     logits_durations.append(last_duration)
        #     logits_gaps.append(last_gap)

        #     note     = torch.argmax(last_note, dim=-1)
        #     duration = torch.argmax(last_duration, dim=-1)
        #     gap      = torch.argmax(last_gap, dim=-1)

        #     out_notes.append(note)
        #     out_durations.append(duration)
        #     out_gaps.append(gap)

        #     # target = torch.cat([note, duration, gap], dim=-1).unsqueeze(1)
        #     #print(target_note.shape, note.shape)
        #     target_note = torch.cat([target_note, note], dim=1)
        #     target_duration = torch.cat([target_duration, duration], dim=1)
        #     target_gap = torch.cat([target_gap, gap], dim=1)
        
        # out_notes = torch.cat(out_notes, dim=1)
        # out_durations = torch.cat(out_durations, dim=1)
        # out_gaps = torch.cat(out_gaps, dim=1)

        # logits_notes = torch.cat(logits_notes, dim=1)
        # logits_durations = torch.cat(logits_durations, dim=1)
        # logits_gaps = torch.cat(logits_gaps, dim=1)
        # return out_notes, out_durations, out_gaps, logits_notes, logits_durations, logits_gaps
    def _generate_beam_search(self, input_ids, cur_len, max_length, min_length, do_sample, early_stopping, temperature, top_k, top_p, num_beams, num_return_sequences, use_cache, encoder_outputs, attention_mask, length_penalty, batch_size, vocab_size):
        # generated hypotheses
        generated_hyps = [
            BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=early_stopping)
            for _ in range(batch_size)
        ]

        # scores for each sentence in the beam
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)

        # for greedy decoding it is made sure that only tokens of the first beam are considered to avoid sampling the exact same tokens three times
        if do_sample is False:
            beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

        # cache compute states
        past = (encoder_outputs, None) if encoder_outputs is not None else None

        # done sentences
        done = [False for _ in range(batch_size)]

        while cur_len < max_length:
            outputs_note = self.note_decoder(input_ids[:, :, 0], past, attention_mask)  # (batch_size * num_beams, cur_len, vocab_size)
            outputs_duration = self.duration_decoder(input_ids[:, :, 1], past, attention_mask)  # (batch_size * num_beams, cur_len, vocab_size)
            outputs_gap = self.gap_decoder(input_ids[:, :, 2], past, attention_mask)  # (batch_size * num_beams, cur_len, vocab_size)
            next_token_logits_note = outputs_note[:, -1, :]  # (batch_size * num_beams, vocab_size)
            next_token_logits_duration = outputs_duration[:, -1, :]  # (batch_size * num_beams, vocab_size)
            next_token_logits_gap = outputs_gap[:, -1, :]  # (batch_size * num_beams, vocab_size)

            scores_notes = F.log_softmax(next_token_logits_note, dim=-1)  # (batch_size * num_beams, vocab_size)
            scores_duration = F.log_softmax(next_token_logits_duration, dim=-1) # (batch_size * num_beams, vocab_size)
            scores_gap = F.log_softmax(next_token_logits_gap, dim=-1) # (batch_size * num_beams, vocab_size)

            
            assert scores_notes.shape == (batch_size * num_beams, vocab_size), "Shapes of scores: {} != {}".format(
                scores_notes.shape, (batch_size * num_beams, vocab_size)
            )
             # set eos token prob to zero if min_length is not reached
            if cur_len < min_length:
                scores_notes[:, self.EOS_token] = -float("inf")
                scores_duration[:, self.EOS_token] = -float("inf")
                scores_gap[:, self.EOS_token] = -float("inf")

            if do_sample:
                _scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
                # Temperature
                if temperature != 1.0:
                    _scores = _scores / temperature
                # Top-p/top-k filtering
                _scores = top_k_top_p_filtering(
                    _scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2
                )  # (batch_size * num_beams, vocab_size)
                # re-organize to group the beam together to sample from all beam_idxs
                _scores = _scores.contiguous().view(
                    batch_size, num_beams * vocab_size
                )  # (batch_size, num_beams * vocab_size)

                # Sample 2 next tokens for each beam (so we have some spare tokens and match output of greedy beam search)
                probs = F.softmax(_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)  # (batch_size, num_beams * 2)
                # Compute next scores
                next_scores = torch.gather(_scores, -1, next_tokens)  # (batch_size, num_beams * 2)
                # sort the sampled vector to make sure that the first num_beams samples are the best
                next_scores, next_scores_indices = torch.sort(next_scores, descending=True, dim=1)
                next_tokens = torch.gather(next_tokens, -1, next_scores_indices)  # (batch_size, num_beams * 2)

            else:
                next_scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)

                # re-organize to group the beam together (we are keeping top hypothesis accross beams)
                next_scores = next_scores.view(
                    batch_size, num_beams * vocab_size
                )  # (batch_size, num_beams * vocab_size)

                next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

            assert next_scores.size() == next_tokens.size() == (batch_size, 2 * num_beams)

            # next batch beam content
            next_batch_beam = []

            # for each sentence
            for batch_idx in range(batch_size):

                # if we are done with this sentence, add a pad token
                if done[batch_idx]:
                    assert (
                        len(generated_hyps[batch_idx]) >= num_beams
                    ), "Batch can only be done if at least {} beams have been generated".format(num_beams)
                    assert (
                        self.EOS_token is not None
                    ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                    next_batch_beam.extend([(0, self.EOS_token, 0)] * num_beams)  # pad the batch
                    continue

                # next sentence beam content, this will get added to next_batch_beam
                next_sent_beam = []

                # next tokens for this sentence
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                    zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    # get beam and token IDs
                    beam_id = beam_token_id // vocab_size
                    token_id = beam_token_id % vocab_size

                    effective_beam_id = batch_idx * num_beams + beam_id
                    # add to generated hypotheses if end of sentence
                    if (self.EOS_token is not None) and (token_id.item() == self.EOS_token):
                        # if beam_token does not belong to top num_beams tokens, it should not be added
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            input_ids[effective_beam_id].clone(), beam_token_score.item(),
                        )
                    else:
                        # add next predicted token since it is not eos_token
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                    # once the beam for next step is full, don't add more tokens to it.
                    if len(next_sent_beam) == num_beams:
                        break

                # Check if we are done so that we can save a pad step if all(done)
                done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                    next_scores[batch_idx].max().item(), cur_len
                )

                # update next beam content
                assert len(next_sent_beam) == num_beams, "Beam should always be full"
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == num_beams * (batch_idx + 1), "We should have added num_beams each step"

            # stop when we are done with each sentence
            if all(done):
                break

            # sanity check / prepare next batch
            assert len(next_batch_beam) == batch_size * num_beams
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])

            # re-order batch and update current length
            input_ids = input_ids[beam_idx, :]
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            cur_len = cur_len + 1

            # re-order internal states
            if past is not None:
                past = self._reorder_cache(past, beam_idx)


        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue

            # test that beam scores match previously calculated scores if not eos and batch_idx not done
            if self.EOS_token is not None and all(
                (token_id % vocab_size).item() != self.EOS_token for token_id in next_tokens[batch_idx]
            ):
                assert torch.all(
                    next_scores[batch_idx, :num_beams] == beam_scores.view(batch_size, num_beams)[batch_idx]
                ), "If batch_idx is not done, final next scores: {} have to equal to accumulated beam_scores: {}".format(
                    next_scores[:, :num_beams][batch_idx], beam_scores.view(batch_size, num_beams)[batch_idx],
                )

            # need to add best num_beams hypotheses to generated hyps
            for beam_id in range(num_beams):
                effective_beam_id = batch_idx * num_beams + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)

        # depending on whether greedy generation is wanted or not define different output_batch_size and output_num_return_sequences_per_batch
        output_batch_size = batch_size if do_sample else batch_size * num_return_sequences
        output_num_return_sequences_per_batch = 1 if do_sample else num_return_sequences

        # select the best hypotheses
        sent_lengths = input_ids.new(output_batch_size)
        best = []

        # retrieve best hypotheses
        for i, hypotheses in enumerate(generated_hyps):
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)

        # shorter batches are padded
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert self.EOS_token is not None, "`Pad_token_id` has to be defined"
            sent_max_len = min(sent_lengths.max().item() + 1, max_length)
            decoded = input_ids.new(output_batch_size, sent_max_len).fill_(self.EOS_token)

            # fill with hypothesis and eos_token_id if necessary
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < max_length:
                    decoded[i, sent_lengths[i]] = self.EOS_token
        else:
            # none of the hypotheses have an eos_token
            assert (len(hypo) == max_length for hypo in best)
            decoded = torch.stack(best).type(torch.long).to(next(self.parameters()).device)

        return decoded

    @staticmethod
    def _reorder_cache(past: Tuple, beam_idx: Tensor) -> Tuple[Tensor]:
        return tuple(layer_past.index_select(1, beam_idx) for layer_past in past)        

    def _generate_no_beam_search(input_ids, cur_len, max_length, min_length, do_sample, temperature, top_k, top_p, num_return_sequences, use_cache, attention_mask):
        pass

class CustomTransformerDecoderMulti(nn.Module):
    def __init__(self, hidden_size, output_size_note, output_size_duration, output_size_gap, MAX_LENGTH, dropout_p, device, num_layers=2, expansion_factor=4, n_heads=8,):
        super(CustomTransformerDecoderMulti, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.output_size_note = output_size_note
        self.output_size_duration = output_size_duration
        self.output_size_gap = output_size_gap
        self.MAX_LENGTH = MAX_LENGTH
        self.dropout_p = dropout_p

        self.note_embedding = nn.Embedding(output_size_note, int(hidden_size/2))
        self.duration_embedding = nn.Embedding(output_size_duration, int(hidden_size/4))
        self.gap_embedding = nn.Embedding(output_size_gap, int(hidden_size/4))
        self.position_embedding = PositionalEmbedding(MAX_LENGTH, hidden_size)
        self.dropout = nn.Dropout(dropout_p)

        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=n_heads,
                dim_feedforward=hidden_size*expansion_factor,
                dropout=dropout_p,
                activation='relu',
                batch_first=True
            ) for _ in range(num_layers)
        ])
        self.fc_note = nn.Linear(hidden_size, output_size_note)
        self.fc_duration = nn.Linear(hidden_size, output_size_duration)
        self.fc_gap = nn.Linear(hidden_size, output_size_gap)

    def forward(self, x, enc_out, mask):
        note     = x[:, :, 0]
        duration = x[:, :, 1]
        gap      = x[:, :, 2]

        note     = self.note_embedding(note)
        duration = self.duration_embedding(duration)
        gap      = self.gap_embedding(gap)

        x = torch.cat([note, duration, gap], dim=-1)
        x = self.position_embedding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, enc_out, mask)
        note     = F.softmax(self.fc_note(x), dim=-1)
        duration = F.softmax(self.fc_duration(x), dim=-1)
        gap      = F.softmax(self.fc_gap(x), dim=-1)
        return note, duration, gap
    
class CustomTransformerDecoderSingle(nn.Module):
    def __init__(self, hidden_size, output_size, MAX_LENGTH, dropout_p, device, num_layers=2, expansion_factor=4, n_heads=8,):
        super(CustomTransformerDecoderSingle, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.MAX_LENGTH = MAX_LENGTH
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.position_embedding = PositionalEmbedding(MAX_LENGTH, hidden_size)
        self.dropout = nn.Dropout(dropout_p)

        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=n_heads,
                dim_feedforward=hidden_size*expansion_factor,
                dropout=dropout_p,
                activation='relu',
                batch_first=True
            ) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x, enc_out, mask):
        x = self.embedding(x)
        x = self.position_embedding(x)
        #x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, enc_out, mask)

        out = F.softmax(self.fc_out(x), dim=-1)
        return out
    
class PositionalEmbedding(nn.Module):
    def __init__(self,max_seq_len,embed_model_dim):
        """
        Args:
            seq_len: length of input sequence
            embed_model_dim: demension of embedding
        """
        super(PositionalEmbedding, self).__init__()
        self.embed_dim = embed_model_dim

        pe = torch.zeros(max_seq_len,self.embed_dim)
        for pos in range(max_seq_len):
            for i in range(0,self.embed_dim,2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/self.embed_dim)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: input vector
        Returns:
            x: output
        """
      
        # make embeddings relatively larger
        x = x * math.sqrt(self.embed_dim)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:,:seq_len], requires_grad=False)
        return x