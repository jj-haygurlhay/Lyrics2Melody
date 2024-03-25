import torch
import torch.nn as nn
from labml_nn.transformers.xl.model import TransformerXL

class MusicTransformerXL(nn.Module):
    def __init__(self, n_token, d_model, n_head, d_head, d_inner, n_layer, dropout, note_vocab_size, duration_vocab_size, gap_vocab_size):
        super(MusicTransformerXL, self).__init__()
        
        self.transformer_xl = TransformerXL(n_token=n_token, d_model=d_model, n_head=n_head, 
                                            d_head=d_head, d_inner=d_inner, n_layer=n_layer, dropout=dropout)
        
        # Output layers for MIDI notes and durations
        self.note_head = nn.Linear(d_model, note_vocab_size)
        self.duration_head = nn.Linear(d_model, duration_vocab_size)
        self.gap_head = nn.Linear(d_model, gap_vocab_size)

    def forward(self, input_ids):
        transformer_outputs = self.transformer_xl(input_ids)
        
        # assuming the output of the last layer is what we use for prediction
        last_layer_output = transformer_outputs[-1]
        
        note_logits = self.note_head(last_layer_output)
        duration_logits = self.duration_head(last_layer_output)
        gap_logits = self.gap_head(last_layer_output)
        
        return note_logits, duration_logits, gap_logits
