import re
from time import sleep
from transformers import (
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    AutoTokenizer,
    T5Tokenizer,
)
import yaml
from dataloader.vocab import Lang
from models.custom_rnn import CustomModelRNN
from utils.quantize import decode_note, decode_duration, decode_gap
import torch 
from inference.generate_midi import GenerateMidi
from midi2audio import FluidSynth

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HYPS_FILE = './config/hyps.yaml'

def serialize_lyrics(lyrics, max_length, syllables_lang, eos_token):
    lyrics_tokens = []
    lyrics = lyrics.lower()
    lyrics = re.sub(r'[^a-z0-9\s]', '', lyrics)
    for syllable in lyrics.split(' ')[:max_length - 1]:
        lyrics_tokens.append(syllables_lang.word2index[syllable])
    lyrics_tokens.append(eos_token)
    return lyrics_tokens

with open(HYPS_FILE, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

model_path = './model.pth'

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
    device=device, 
)
model.load_state_dict(torch.load(model_path))
model.to(device)


max_length = config['data']['max_sequence_length']

# text = 'People get ready a train a you need no baggage you just get on board you need is faith to hear the diesels need no ticket you just thank the Lord so people get ready coast the doors and board hope for among those loved the most there no room for the hopeless sinner who would hurt mankind believe me now have pity on grow thinner for no hiding place against the throne so people get ready a train a you need no baggage you just get on board you need is faith to hear the diesels'
# text = 'Peo ple get rea dy a train a you need no bag gage you just get on board you need is faith to hear the die sels need no tic ket you just thank the Lord so peo ple get rea dy coast the doors and board hope for among those loved the most there no room for the hope less sin ner who would hurt man kind be lieve me now have pi ty on grow thin ner for no hi ding place against the throne so peo ple get rea dy a train a you need no bag gage you just get on board you need is faith to hear the die sels'
text = 'Peo ple get rea dy a train a you need no bag gage you just get on board you need'
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
            try:
                note = decode_note(note_id-2)
                duration = decode_duration(duration_id-2)
                gap = decode_gap(gap_id-2)
                sequence.append([note, duration, gap])
            except:
                err_count += 1
                continue
    if err_count > 0:
        print(f"Error count: {err_count}")
    return sequence

midi_sequence = decode_midi_sequence(decoder_outputs_notes, decoder_outputs_durations, decoder_outputs_gaps)
print(midi_sequence)
print('Input Sequence length: ', len(inputs[0]))
print('Output Sequence length: ', len(midi_sequence))

# Generate MIDI file
midi = GenerateMidi()
fs = FluidSynth(sound_font='./.fluidsynth/default_sound_font.sf2')

midi_output = midi.create_midi_pattern_from_discretized_data(midi_sequence)
destination_midi = "./test.mid"
destination_wav = "./test.wav"
midi_output.write(destination_midi)
fs.midi_to_audio(destination_midi, destination_wav)


