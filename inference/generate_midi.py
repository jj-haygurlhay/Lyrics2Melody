from utils import midi_utils

class GenerateMidi:
    def __init__(self, decoded_notes, decoded_durations, decoded_gaps):
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

