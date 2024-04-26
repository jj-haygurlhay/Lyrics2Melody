import math
from typing import Tuple
from models.base_model import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from project_utils.quantize import MIDI_NOTES, DURATIONS, GAPS


class CustomModelTransformer(BaseModel):

    def __init__(self, encoder, EOS_token, SOS_token, device, MAX_LENGTH=21, dropout_p=0.1, train_encoder=False, expansion_factor=4, num_heads=8, num_layers=2):
        super().__init__()
        self.device = device
        self.MAX_LENGTH = MAX_LENGTH
        self.SOS_token = SOS_token
        self.EOS_token = EOS_token

        # Take pretrained encoder
        self.encoder = encoder

        # Define decoder
        self.decoder = CustomTransformerDecoderMulti(
            hidden_size=encoder.config.n_positions,
            output_size_note=len(MIDI_NOTES)+2,
            output_size_duration=len(DURATIONS)+2,
            output_size_gap=len(GAPS)+2,
            MAX_LENGTH=MAX_LENGTH,
            dropout_p=dropout_p,
            device=device,
            num_layers=num_layers,
            expansion_factor=expansion_factor,
            n_heads=num_heads
        )

        self.train_encoder = train_encoder

        self.encoder.to(device)
        self.decoder.to(device)
        
        self._init_weights(self.encoder)
        self._init_weights(self.decoder)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def make_trg_mask(self, trg):
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
        
        note, duration, gap = self.decoder(target, encoder_output[0], target_mask)

        return note, duration, gap    
    
    def generate(self, x, attn, do_sample=True, max_length=21, temperature=0.5, top_k=None):
        sos_note = torch.full((x.shape[0], 1), self.SOS_token, dtype=torch.int64).to(self.device)
        sos_duration = torch.full((x.shape[0], 1), self.SOS_token, dtype=torch.int64).to(self.device)
        sos_gap = torch.full((x.shape[0], 1), self.SOS_token, dtype=torch.int64).to(self.device)
        target = torch.cat([sos_note, sos_duration, sos_gap], dim=-1).unsqueeze(1)
        
        # Encoder output
        encoder_output = self.encoder(x, attn, output_hidden_states=True, output_attentions=False)
        sampled_notes, sampled_durations, sampled_gaps = [], [], []
        logits_notes, logits_durations, logits_gaps = [], [], []

        for _ in range(self.MAX_LENGTH):
            target_mask = self.make_trg_mask(target)
            note_logits, duration_logits, gap_logits = self.decoder(target, encoder_output[0], target_mask)

            # Store logits for loss calculation
            logits_notes.append(note_logits[:, -1, :].unsqueeze(1))
            logits_durations.append(duration_logits[:, -1, :].unsqueeze(1))
            logits_gaps.append(gap_logits[:, -1, :].unsqueeze(1))

            if do_sample:
                # Apply temperature scaling and top-k sampling
                note_logits = note_logits[:, -1, :] / temperature
                duration_logits = duration_logits[:, -1, :] / temperature
                gap_logits = gap_logits[:, -1, :] / temperature

                # Apply top-k sampling
                if top_k is not None:
                    note_logits = self.apply_top_k_sampling(note_logits, top_k[0])
                    duration_logits = self.apply_top_k_sampling(duration_logits, top_k[1])
                    gap_logits = self.apply_top_k_sampling(gap_logits, top_k[2])

                # Sampling using softmax
                note = torch.multinomial(torch.softmax(note_logits, dim=-1), 1)
                duration = torch.multinomial(torch.softmax(duration_logits, dim=-1), 1)
                gap = torch.multinomial(torch.softmax(gap_logits, dim=-1), 1)
            else:
                # Use argmax
                note = torch.argmax(note_logits.unsqueeze(1), dim=-1, keepdim=True)
                duration = torch.argmax(duration_logits.unsqueeze(1), dim=-1, keepdim=True)
                gap = torch.argmax(gap_logits.unsqueeze(1), dim=-1, keepdim=True)

            sampled_notes.append(note)
            sampled_durations.append(duration)
            sampled_gaps.append(gap)
            
            # Prepare for next step
            new_entry = torch.cat([note, duration, gap], dim=-1).unsqueeze(1)
            target = torch.cat([target, new_entry], dim=1)

        # Concatenate along sequence dimension
        sampled_notes = torch.cat(sampled_notes, dim=1)
        sampled_durations = torch.cat(sampled_durations, dim=1)
        sampled_gaps = torch.cat(sampled_gaps, dim=1)
        logits_notes = torch.cat(logits_notes, dim=1)
        logits_durations = torch.cat(logits_durations, dim=1)
        logits_gaps = torch.cat(logits_gaps, dim=1)

        return sampled_notes, sampled_durations, sampled_gaps, logits_notes, logits_durations, logits_gaps

    def apply_top_k_sampling(self, logits, top_k):
        if top_k > 0:
            top_values, _ = torch.topk(logits, top_k)
            kth_value = top_values[:, -1].view(-1, 1)
            mask = logits < kth_value
            logits[mask] = -float('Inf')
        return logits

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

        return self.fc_note(x), self.fc_duration(x), self.fc_gap(x)
    
class PositionalEmbedding(nn.Module):
    def __init__(self,max_seq_len,embed_model_dim):
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
        # make embeddings relatively larger
        x = x * math.sqrt(self.embed_dim)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:,:seq_len], requires_grad=False)
        return x