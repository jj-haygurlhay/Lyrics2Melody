from time import sleep
from transformers import (
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    AutoTokenizer,
    T5Tokenizer,
)
from utils.quantize import decode_note, decode_duration, decode_gap
import torch 
from inference.generate_midi import GenerateMidi
from midi2audio import FluidSynth

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_path = './runs/2024-04-10_13-51-53/checkpoint-1600'

model = T5ForConditionalGeneration.from_pretrained(model_path).to('cuda')
tokenizer = T5Tokenizer.from_pretrained(model_path)


max_length = 20

# text = 'People get ready a train a you need no baggage you just get on board you need is faith to hear the diesels need no ticket you just thank the Lord so people get ready coast the doors and board hope for among those loved the most there no room for the hopeless sinner who would hurt mankind believe me now have pity on grow thinner for no hiding place against the throne so people get ready a train a you need no baggage you just get on board you need is faith to hear the diesels'
# text = 'Peo ple get rea dy a train a you need no bag gage you just get on board you need is faith to hear the die sels need no tic ket you just thank the Lord so peo ple get rea dy coast the doors and board hope for among those loved the most there no room for the hope less sin ner who would hurt man kind be lieve me now have pi ty on grow thin ner for no hi ding place against the throne so peo ple get rea dy a train a you need no bag gage you just get on board you need is faith to hear the die sels'
text = 'Peo ple get rea dy a train a you need no bag gage you just get on board you need'
inputs = ['notes: ' + text]

inputs = tokenizer(inputs, truncation=True, max_length=max_length*2, return_tensors='pt').to(device)
output = model.generate(
    **inputs, 
    num_beams=4, 
    min_length=10, 
    max_length=max_length, 
    do_sample=True, 
    temperature=1.0, 
    repetition_penalty=1.0,
    # top_k=50, 
    #top_p=0.98, 
    eos_token_id=tokenizer.eos_token_id, 
    pad_token_id=tokenizer.pad_token_id,
    #early_stopping=True,
    num_return_sequences=1,
    bad_words_ids=[[42612]]
    )

decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
decoded_output = decoded_output.replace('.', '')

def decode_midi_sequence(decoded_output):
    sequence = []
    note, duration, gap = -1, -1, -1
    err_count = 0
    notes = []
    durations = []
    gaps = []
    for token in decoded_output.split(' '):
        for sub_token in token[1:-1].split('_'):
            if 'note' in sub_token:
                note = decode_note(int(sub_token[4:]))
            elif 'duration' in sub_token:
                duration = decode_duration(int(sub_token[8:]))
            elif 'gap' in sub_token:
                gap = decode_gap(int(sub_token[3:]))
                sequence.append([note, duration, gap])
                notes.append(note)
                durations.append(duration)
                gaps.append(gap)
                note, duration, gap = -1, -1, -1
            else:
                note, duration, gap = -1, -1, -1
                err_count += 1
    if err_count > 0:
        print(f"Error count: {err_count}")
    return sequence, notes, durations, gaps

midi_sequence, notes, durations, gaps = decode_midi_sequence(decoded_output)
print(tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)[0])
print(midi_sequence)
print('Input Sequence length: ', len(inputs['input_ids'][0]))
print('Output Sequence length: ', len(midi_sequence))

# Generate MIDI file
midi = GenerateMidi()
fs = FluidSynth(sound_font='./.fluidsynth/default_sound_font.sf2')

midi_output = midi.create_midi_pattern_from_discretized_data(midi_sequence)
destination_midi = "./test.mid"
destination_wav = "./test.wav"
midi_output.write(destination_midi)
fs.midi_to_audio(destination_midi, destination_wav)


