import torch
import torch.nn as nn
from transformers import GPT2PreTrainedModel, GPT2Model

from models.base_model import BaseModel

class MusicGPT2(GPT2PreTrainedModel, BaseModel):
    def __init__(self, config, note_loss_weight=1.0, duration_loss_weight=1.0, gap_loss_weight=1.0):
        super(BaseModel).__init__(note_loss_weight=note_loss_weight, duration_loss_weight=duration_loss_weight, gap_loss_weight=gap_loss_weight)
        super(GPT2PreTrainedModel).__init__(config)
        self.gpt2 = GPT2Model(config)
        
        # Custom head for predicting MIDI notes and durations
        self.note_head = nn.Linear(config.n_embd, len(self.valid_midi_notes))
        self.duration_head = nn.Linear(config.n_embd, len(self.valid_durations))
        self.gap_head = nn.Linear(config.n_embd, len(self.valid_gaps))

    def forward(self, input_ids, attention_mask=None):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        note_logits = self.note_head(sequence_output[:, -1, :])
        duration_logits = self.duration_head(sequence_output[:, -1, :])
        gap_logits = self.gap_head(sequence_output[:,-1,:])

        return note_logits, duration_logits, gap_logits

    def custom_loss(self, note_logits, duration_logits, gap_logits, note_targets, duration_targets, gap_targets):
        note_loss = nn.CrossEntropyLoss()(note_logits, note_targets)
        duration_loss = nn.CrossEntropyLoss()(duration_logits, duration_targets)
        gap_loss = nn.CrossEntropyLoss()(gap_logits, gap_targets)

        return self.note_loss_weight * note_loss + self.duration_loss_weight * duration_loss + self.gap_loss_weight * gap_loss

    # Assuming MIDI notes are encoded as integers [0, 127] and durations are also integers
    def decode_model_output(self, note_logits, duration_logits, gap_logits):
        # Softmax to convert logits to probabilities, then argmax to get most likely MIDI note
        predicted_notes = torch.argmax(torch.softmax(note_logits, dim=-1), dim=-1)
        
        # Similarly, for durations, assuming they are categorized (you may need to adjust based on your actual encoding)
        predicted_durations = torch.argmax(torch.softmax(duration_logits, dim=-1), dim=-1)
        
        # Softmax to convert logits to probabilities then argmax to get most likely gap
        predicted_gaps = torch.argmax(torch.softmax(gap_logits, dim=-1), dim=-1)

        # Convert tensors to lists for further processing if necessary
        decoded_notes = predicted_notes.tolist()
        decoded_durations = predicted_durations.tolist()
        decoded_gaps = predicted_gaps.tolist()
        
        # Mapping back to actual durations/gap times -> implement nice global variables for this
        decoded_durations = [self.valid_durations[index] for index in decoded_durations]
        decoded_gaps = [self.valid_gaps[index] for index in decoded_gaps]
        
        return decoded_notes, decoded_durations, decoded_gaps 