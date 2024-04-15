import numpy as np

MIDI_NOTES = np.arange(128)
DURATIONS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 6.0, 6.5, 8.0, 8.5, 16.0, 16.5, 32.0, 32.5]
GAPS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]

def encode_note(note):
    return MIDI_NOTES.tolist().index(note)

def decode_note(encoded_note):
    return MIDI_NOTES[encoded_note]

def encode_duration(duration):
    return DURATIONS.index(duration)

def decode_duration(encoded_duration):
    return DURATIONS[encoded_duration]

def encode_gap(gap):
    return GAPS.index(gap)

def decode_gap(encoded_gap):
    return GAPS[encoded_gap]