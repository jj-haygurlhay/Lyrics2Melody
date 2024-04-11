import os
import re
import yaml
from dataloader.vocab import Lang
from models.custom_rnn import CustomModelRNN
from utils.quantize import decode_note, decode_duration, decode_gap
import torch 
from inference.generate_midi import GenerateMidi
from midi2audio import FluidSynth

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dir = './runs/RNN/2024-04-11_16-52-50/' # Change this to the path of the model you want to use

model_path = os.path.join(model_dir, 'model.pt')
config_path = os.path.join(model_dir, 'config.yaml')

def serialize_lyrics(lyrics, max_length, syllables_lang, eos_token):
    lyrics_tokens = []
    lyrics = lyrics.lower()
    lyrics = re.sub(r'[^a-z0-9\s]', '', lyrics)
    for syllable in lyrics.split(' ')[:max_length - 1]:
        lyrics_tokens.append(syllables_lang.word2index[syllable])
    lyrics_tokens.append(eos_token)
    return lyrics_tokens

with open(config_path, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Create language objects
syllables = Lang('syllables')
lines = open('./vocab/syllables.txt', 'r', encoding='utf-8').read().strip().split('\n')
for syllable in lines:
    syllables.addWord(syllable)

model = CustomModelRNN(
    input_size=syllables.n_words,
    hidden_size=config['model']['hidden_size'], 
    SOS_token=0, 
    MAX_LENGTH=config['data']['max_sequence_length'], 
    dropout_p=config['model']['dropout'],
    num_layers=config['model']['num_layers'],
    device=device, 
    )
model.load_state_dict(torch.load(model_path))
model.to(device)


max_length = config['data']['max_sequence_length']

text = 'Peo ple get rea dy a train a you need no bag gage you just get on board you need is faith to hear the die sels need no tic ket you just thank the Lord so peo ple get rea dy coast the doors and board hope for among those loved the most there no room for the hope less sin ner who would hurt man kind be lieve me now have pi ty on grow thin ner for no hi ding place against the throne so peo ple get rea dy a train a you need no bag gage you just get on board you need is faith to hear the die sels'
inputs = [serialize_lyrics(text, max_length, syllables, 1)]
input_tensor = torch.tensor(inputs).to(device)

decoder_outputs_notes, decoder_outputs_durations, decoder_outputs_gaps, _, _ = model(input_tensor)

def decode_midi_sequence(decoder_outputs_notes, decoder_outputs_durations, decoder_outputs_gaps):
    sequence = []
    err_count = 0
    for note, duration, gap in zip(decoder_outputs_notes[0], decoder_outputs_durations[0], decoder_outputs_gaps[0]):
            note_id = note.argmax().item()
            duration_id = duration.argmax().item()
            gap_id = gap.argmax().item()
            if note_id > 1 and duration_id > 1 and gap_id > 1:
                try:
                    note = decode_note(note_id-2)
                    duration = decode_duration(duration_id-2)
                    gap = decode_gap(gap_id-2)
                    sequence.append([note, duration, gap])
                except:
                    err_count += 1
                    continue
            else:
                break # EOS token reached
    if err_count > 0:
        print(f"Error count: {err_count}")
    return sequence

midi_sequence = decode_midi_sequence(decoder_outputs_notes, decoder_outputs_durations, decoder_outputs_gaps)
print('MIDI sequence', midi_sequence)
print('Input Sequence length: ', len(inputs[0]))
print('Output Sequence length: ', len(midi_sequence))

# Generate MIDI file
midi = GenerateMidi()
midi_output = midi.create_midi_pattern_from_discretized_data(midi_sequence)
destination_midi = os.path.join(model_dir, "test.mid")
midi_output.write(destination_midi)

try:
    destination_wav = os.path.join(model_dir, "test.wav")
    fs = FluidSynth(sound_font='./.fluidsynth/default_sound_font.sf2')
    fs.midi_to_audio(destination_midi, destination_wav)
except:
    print("Error converting midi to wav, missing sound font file. Please install fluidsynth and download a sound font file.")