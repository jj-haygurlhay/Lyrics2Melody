import math
from models.base_model import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.quantize import MIDI_NOTES, DURATIONS, GAPS


class CustomModelTransformer(BaseModel):

    def __init__(self, encoder, device, SOS_token=0, MAX_LENGTH=100, dropout_p=0.1, train_encoder=False, expansion_factor=4, num_heads=8, num_layers=2):
        super().__init__()
        self.device = device
        self.MAX_LENGTH = MAX_LENGTH

        # Take pretrained encoder
        self.encoder = encoder

        # Define decoders
        self.decoder = CustomTransformerDecoder(
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

    def make_trg_mask(self, trg):
        """
        Args:
            trg: target sequence
        Returns:
            trg_mask: target mask
        """
        batch_size, trg_len, _ = trg.shape
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
    
    def generate(self, x, attn):
        encoder_output = self.encoder(x, attn, output_hidden_states=True, output_attentions=False)
        target = torch.zeros(x.shape[0], 1, 3, dtype=torch.int64).to(self.device)
        target_mask = self.make_trg_mask(target)

        out_notes, out_durations, out_gaps = [], [], []
        logits_notes, logits_durations, logits_gaps = [], [], []
        for _ in range(self.MAX_LENGTH-1): # -1 to account for the SOS token
            note, duration, gap = self.decoder(target, encoder_output[0], target_mask)

            logits_notes.append(note)
            logits_durations.append(duration)
            logits_gaps.append(gap)

            note     = torch.argmax(note, dim=-1)
            duration = torch.argmax(duration, dim=-1)
            gap      = torch.argmax(gap, dim=-1)

            out_notes.append(note)
            out_durations.append(duration)
            out_gaps.append(gap)

            target = torch.cat([note, duration, gap], dim=-1).unsqueeze(1)
        
        out_notes = torch.cat(out_notes, dim=1)
        out_durations = torch.cat(out_durations, dim=1)
        out_gaps = torch.cat(out_gaps, dim=1)

        logits_notes = torch.cat(logits_notes, dim=1)
        logits_durations = torch.cat(logits_durations, dim=1)
        logits_gaps = torch.cat(logits_gaps, dim=1)
        return out_notes, out_durations, out_gaps, logits_notes, logits_durations, logits_gaps


class CustomTransformerDecoder(nn.Module):
    def __init__(self, hidden_size, output_size_note, output_size_duration, output_size_gap, MAX_LENGTH, dropout_p, device, num_layers=2, expansion_factor=4, n_heads=8,):
        super(CustomTransformerDecoder, self).__init__()
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