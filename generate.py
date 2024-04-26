import os
import torch 
from inference import Generator, GenerateMidi
from midi2audio import FluidSynth

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PARAMETERS TO CHECK AND CHANGE
model_dir = './runs/RNN/SlidingWindow_Shift0.8/' # Change this to the path of the model you want to use
topk = [30, 5, 5] # [notes, durations, gaps]
temperature = 0.5
log_multiple_gen = True
model_type = 'rnn'
shift = 1 # POUR LE BEST T5, j'ai seulement utilisé le PAD token, donc shift=1

# Load generator
generator = Generator(model_dir, './vocab/syllables.txt', 'model_best.pt', model_type, device=device)

text = 'Peo ple get rea dy a train a you need no bag gage you just get on board you need is faith to hear the die sels need no tic ket you just thank the Lord so peo ple get rea dy coast the doors and board hope for among those loved the most there no room for the hope less sin ner who would hurt man kind be lieve me now have pi ty on grow thin ner for no hi ding place against the throne so peo ple get rea dy a train a you need no bag gage you just get on board you need is faith to hear the die sels'

output_folder = os.path.join(model_dir, 'outputs')
os.makedirs(output_folder, exist_ok=True)
midi_sequence_logging_path = os.path.join(output_folder, 'midi_sequences.txt')

log_amount = 50 if log_multiple_gen else 1

for i in range(log_amount):
    midi_sequence = generator.predict(text, temperature=temperature, topk=topk, shift=shift)
    print('MIDI sequence', midi_sequence)
    
    # Save MIDI sequence
    with open(midi_sequence_logging_path, 'a') as f:
        f.write(f'{midi_sequence}\n')

# Generate MIDI file
n = 1
while os.path.exists(os.path.join(output_folder, f'midi-{n}.mid')):
    n += 1
output_file_prefix = os.path.join(output_folder, f'midi-{n}')
    
midi = GenerateMidi()
midi_output = midi.create_midi_pattern(midi_sequence)
destination_midi = output_file_prefix + '.mid'
midi_output.write(destination_midi)

try:
    destination_wav = output_file_prefix + '.wav'
    fs = FluidSynth(sound_font='.fluidsynth/default_sound_font.sf2')
    fs.midi_to_audio(destination_midi, destination_wav)
except:
    print("Error converting midi to wav, missing sound font file. Please install fluidsynth and download a sound font file.")