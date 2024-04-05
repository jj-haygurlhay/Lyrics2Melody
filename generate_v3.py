from transformers import (
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    AutoTokenizer,
    T5Tokenizer,
    T5EncoderModel,
)
import yaml
from models.custom_rnn import CustomModelRNN
from models.custom_transformer import CustomModelTransformer
from utils.quantize import decode_note, decode_duration, decode_gap
import torch 

HYPS_FILE = './config/hyps.yaml'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = './model.pth'

# Load model and tokenizer
encoder_name = 't5-small'
encoder = T5EncoderModel.from_pretrained(encoder_name)
tokenizer = T5Tokenizer.from_pretrained(encoder_name, legacy=False)

with open(HYPS_FILE, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

model = CustomModelTransformer(
    encoder=encoder, 
    device=device, 
    SOS_token=0, 
    MAX_LENGTH=config['data']['max_sequence_length'], 
    train_encoder=False, 
    dropout_p=config['model']['dropout'],
    expansion_factor=config['model']['expansion_factor'],
    num_heads=config['model']['num_heads'],
    num_layers=config['model']['num_layers']
    )

max_length = config['data']['max_sequence_length']

model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

text = 'People get ready a train a you need no baggage you just get on board you need is faith to hear the diesels need no ticket you just thank the Lord so people get ready coast the doors and board hope for among those loved the most there no room for the hopeless sinner who would hurt mankind believe me now have pity on grow thinner for no hiding place against the throne so people get ready a train a you need no baggage you just get on board you need is faith to hear the diesels'
inputs = tokenizer(text, truncation=True, max_length=max_length, return_tensors='pt').to(device)

with torch.no_grad():
    decoder_outputs_notes, decoder_outputs_durations, decoder_outputs_gaps, _, _, _ = model.generate(inputs['input_ids'], inputs['attention_mask'])

def decode_midi_sequence(decoded_output_notes, decoded_output_durations, decoded_output_gaps):
    sequence = []
    for note, duration, gap in zip(decoded_output_notes[0], decoded_output_durations[0], decoded_output_gaps[0]):
        if note == 1 or duration == 1 or gap == 1: # EOS token
            break
        if note == 0 or duration == 0 or gap == 0: # SOS token
            continue
        sequence.append([decode_note(note-2), decode_duration(duration-2), decode_gap(gap-2)])
    return sequence

midi_sequence = decode_midi_sequence(decoder_outputs_notes, decoder_outputs_durations, decoder_outputs_gaps)
print(midi_sequence)
print('Inputs length:', len(inputs['input_ids'][0]))
print('Outputs length:', len(midi_sequence))