import torch
import torch.nn as nn
from transformers import T5Model 

class MusicT5(nn.Module):
    def __init__(self, config, num_gaps=7, num_notes=128, num_durations=12):
        super().__init__()
        self.t5 = T5Model(config)

        self.note_head = nn.Linear(config.n_embd, num_notes)
        self.duration_head = nn.Linear(config.n_embd, num_durations)
        self.gap_head = nn.Linear(config.n_embd, num_gaps)

    def forward(self, input_ids, attention_mask=None, encoder_outputs=None):
        outputs = self.t5(input_ids = input_ids, attention_mask = attention_mask, encoder_outputs = encoder_outputs)
        sequence_output = outputs.last_hidden_state

        # Note: [:, -1, :] because we treat it as a classification problem 
        note_logits = self.note_head(sequence_output[:, -1, :]) 
        duration_logits = self.duration_head(sequence_output[:, -1, :])
        gap_logits = self.gap_head(sequence_output[:,-1,:])

        return note_logits, duration_logits, gap_logits
    
    def custom_loss(note_logits, duration_logits, gap_logits, note_targets, duration_targets, gap_targets):
        note_loss = nn.CrossEntropyLoss()(note_logits, note_targets)
        duration_loss = nn.CrossEntropyLoss()(duration_logits, duration_targets)
        gap_loss = nn.CrossEntropyLoss()(gap_logits, gap_targets)

        return note_loss + duration_loss + gap_loss
    
    # Example instantiation
    # config = T5Config.from_pretrained('t5-small')
    # t5_model = MusicT5(config)

# Assuming MIDI notes are encoded as integers [0, 127] and durations are also integers
def decode_model_output(note_logits, duration_logits, gap_logits):
    # Softmax to convert logits to probabilities, then argmax to get most likely MIDI note
    predicted_notes = torch.max(torch.softmax(note_logits, dim=-1), dim=-1)
    
    # Similarly, for durations, assuming they are categorized (you may need to adjust based on your actual encoding)
    predicted_durations = torch.max(torch.softmax(duration_logits, dim=-1), dim=-1)
    
    # Softmax to convert logits to probabilities then argmax to get most likely gap
    predicted_gaps = torch.max(torch.softmax(gap_logits, dim=-1), dim=-1)

    # Convert tensors to lists for further processing if necessary
    decoded_notes = predicted_notes.tolist()
    decoded_durations = predicted_durations.tolist()
    decoded_gaps = predicted_gaps.tolist()
    return decoded_notes, decoded_durations, decoded_gaps 