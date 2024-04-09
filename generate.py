from transformers import (
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    AutoTokenizer,
    T5Tokenizer,
)
from utils.quantize import decode_note, decode_duration, decode_gap
import torch 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_path = './runs/2024-04-02_14-51-40/checkpoint-2864'

model = T5ForConditionalGeneration.from_pretrained(model_path).to('cuda')
tokenizer = T5Tokenizer.from_pretrained(model_path)


max_length = 512

text = 'People get ready a train a you need no baggage you just get on board you need is faith to hear the diesels need no ticket you just thank the Lord so people get ready coast the doors and board hope for among those loved the most there no room for the hopeless sinner who would hurt mankind believe me now have pity on grow thinner for no hiding place against the throne so people get ready a train a you need no baggage you just get on board you need is faith to hear the diesels'

inputs = ['notes: ' + text]

inputs = tokenizer(inputs, truncation=True, max_length=max_length, return_tensors='pt').to(device)
output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=10, max_length=max_length)

decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
print(output)
print(decoded_output)

def decode_midi_sequence(decoded_output):
    sequence = []
    note, duration, gap = -1, -1, -1
    err_count = 0
    end_gap = -1
    for token in decoded_output.split(' '):
        try:
            if 'note' in token and duration == -1 and gap == -1:
                duration = -1
                gap = -1
                note = decode_note(int(token[5:end_gap]))
            elif 'duration' in token and note != -1:
                gap = -1
                duration = decode_duration(int(token[9:end_gap]))
            elif 'gap' in token and note != -1 and duration != -1:
                gap = decode_gap(int(token[4:end_gap]))
                sequence.append([note, duration, gap])
                note, duration, gap = -1, -1, -1
            else:
                note, duration, gap = -1, -1, -1
                err_count += 1
        except:
            print(decoded_output)
            print(f"Error decoding token {token}")
    if err_count > 0:
        print(f"Error count: {err_count}")
    return sequence

#midi_sequence = decode_midi_sequence(decoded_output)
#print(midi_sequence)