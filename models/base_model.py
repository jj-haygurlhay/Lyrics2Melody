import torch
import torch.nn as nn
import numpy as np

MIDI_NOTES = np.arange(128)
DURATIONS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 6.0, 6.5, 8.0, 8.5, 16.0, 16.5, 32.0, 32.5]
GAPS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]

class BaseModel(nn.Module):
    def __init__(self, note_loss_weight=1.0, duration_loss_weight=1.0, gap_loss_weight=1.0):
        super().__init__()

        self.valid_midi_notes = MIDI_NOTES
        self.valid_durations = DURATIONS
        self.valid_gaps = GAPS

        self.note_loss_weight = note_loss_weight
        self.duration_loss_weight = duration_loss_weight
        self.gap_loss_weight = gap_loss_weight

    def forward(self, x):
        raise NotImplementedError