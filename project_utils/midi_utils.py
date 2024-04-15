import mido

def create_new_midi():
    # Create a new MIDI file with one track
    midi_file = mido.MidiFile()
    track = mido.MidiTrack()
    midi_file.tracks.append(track)
    return midi_file

def create_track(midi_file):
    # Create a new track in the given MIDI file
    track = mido.MidiTrack()
    midi_file.tracks.append(track)
    return track

def set_tempo(midi_file, bpm):
    # Set the tempo for the MIDI file
    tempo = mido.bpm2tempo(bpm)
    for track in midi_file.tracks:
        track.append(mido.MetaMessage('set_tempo', tempo=tempo))
        
def add_metadata(midi_file, title="", copyright=""):
    # Add metadata to the MIDI file
    track = midi_file.tracks[0]  # Typically, metadata goes in the first track
    track.append(mido.MetaMessage('track_name', name=title))
    if copyright:
        track.append(mido.MetaMessage('copyright', text=copyright))

def add_note(track, note, duration, gap, velocity=64):
    # Add a note to the given track, with optional velocity, duration, and gap
    track.append(mido.Message('note_on', note=note, velocity=velocity, time=0))
    track.append(mido.Message('note_off', note=note, velocity=velocity, time=duration))
    # If there is a gap after the note, insert a pause
    if gap:
        track.append(mido.Message('note_off', note=note, velocity=0, time=gap))

def save_midi(midi_file, filename):
    # Save the MIDI file to disk
    midi_file.save(filename)