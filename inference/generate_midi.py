from utils import midi_utils
import pretty_midi

class GenerateMidi:
    def __init__(self, decoded_notes=None, decoded_durations=None, decoded_gaps=None):
        self.decoded_notes = decoded_notes
        self.decoded_durations = decoded_durations
        self.decoded_gaps = decoded_gaps
    
    def create_midi(self, bpm=120, title="", copyright=""):
        mid = midi_utils.create_new_midi()  # Assuming midi_utils has this function
        midi_utils.set_tempo(mid, bpm)
        midi_utils.add_metadata(mid, title, copyright)
        
        track = midi_utils.create_track(mid)  # Or however you create a track in midi_utils
        
        for note, duration, gap in zip(self.decoded_notes, self.decoded_durations, self.decoded_gaps):
            midi_utils.add_note(track, note, duration, gap)
        
        return mid
    
    def save_midi(self, filename):
        mid = self.create_midi()
        midi_utils.save_midi(mid, filename)
        print(f'MIDI file saved as {filename}')

    def create_midi_pattern_from_discretized_data(self, midi_sequence):
        new_midi = pretty_midi.PrettyMIDI()
        voice = pretty_midi.Instrument(1)  # It's here to change the used instruments !
        tempo = 120
        ActualTime = 0  # Time since the beginning of the song, in seconds
        for i in range(0,len(midi_sequence)):
            length = midi_sequence[i][1] * 60 / tempo  # Conversion Duration to Time
            if i < len(midi_sequence) - 1:
                gap = midi_sequence[i + 1][2] * 60 / tempo
            else:
                gap = 0  # The Last element doesn't have a gap
            note = pretty_midi.Note(velocity=100, pitch=int(midi_sequence[i][0]), start=ActualTime,
                                    end=ActualTime + length)
            voice.notes.append(note)
            ActualTime += length + gap  # Update of the time

        new_midi.instruments.append(voice)

        return new_midi


