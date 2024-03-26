import torch
import torch.nn as nn
import mido

class GenerateMidi(nn.Module):
    def __init__(self, decoded_notes, decoded_durations, decoded_gaps):
        self.decoded_notes = decoded_notes
        self.decoded_durations = decoded_durations
        self.decoded_gaps = decoded_gaps

    def create_MIDI(self):
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        
        # Set an initial delay if necessary
        track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(120)))
        
        # Iterate over the notes and durations to create MIDI messages
        for note, duration, gap in zip(self.decoded_notes, self.decoded_durations, self.decoded_gaps):
            # Add a note on
            track.append(mido.Message('note_on', note=note, velocity=64, time=0))
            # Add a note off, note the time here is the duration
            track.append(mido.Message('note_off', note=note, velocity=64, time=duration))
            # Add a gap if necessary
            if gap:
                track.append(mido.Message('note_off', note=note, velocity=0, time=gap))
        
        return mid
    
    def save_midi(self, filename):
        # Assumes that a MIDI file has been created with create_midi
        mid = self.create_midi()
        mid.save(filename)
        print(f'MIDI file saved as {filename}')

