import os
import re

import torch
import yaml

from dataloader.vocab import Lang
from models import CustomModelRNN, CustomModelTransformer
from project_utils.quantize import decode_duration, decode_gap, decode_note
from transformers import T5EncoderModel, T5Tokenizer

class Generator:
    def __init__(self, model_dir, vocab_dir, model_name, model_type, device='cpu', eos_token=1):
        model_path = os.path.join(model_dir, model_name)
        config_path = os.path.join(model_dir, 'config.yaml')
        self.device = device
        self.eos_token = eos_token
        self.model_type = model_type.lower()
        self.load_config(config_path)
        if self.model_type == 'rnn':
            self.load_syllables_vocab(vocab_dir)
            self.load_RNN_model(model_path)
        elif self.model_type == 'transformer':
            self.load_t5_model(model_path)
        else:
            raise ValueError('Invalid model type! Choose between "rnn" and "transformer"')
    
    def predict(self, lyrics, temperature=1.0, topk=None, shift=2):
        if self.model_type == 'rnn':
            return self.predict_rnn(lyrics, temperature, topk, shift=shift)
        elif self.model_type == 'transformer':
            return self.predict_transformer(lyrics, temperature, topk, shift=shift)
        else:
            raise ValueError('Invalid model type! Choose between "load_RNN_model" and "load_t5_model"')
    
    def predict_rnn(self, lyrics, temperature=1.0, topk=None, shift=2):
        inputs = [self.serialize_lyrics_rnn(lyrics, self.config['data']['max_sequence_length'], self.eos_token)]
        input_tensor = torch.tensor(inputs).to(self.device)

        with torch.no_grad():
            _, _, _, _, _, decoded_notes, decoded_durations, decoded_gaps = self.model(input_tensor, generate_temp=temperature, topk=topk)

        midi_sequence = self.decode_outputs(decoded_notes, decoded_durations, decoded_gaps, shift=shift)

        return midi_sequence
        
    def predict_transformer(self, lyrics, temperature=1.0, topk=None, shift=2):
        inputs = [self.serialize_lyrics_transformer(lyrics, self.config['data']['max_sequence_length'])]
        input_tensor = torch.tensor(inputs['input_ids']).to(self.device)
        
        with torch.no_grad():
           decoder_outputs_notes, decoder_outputs_durations, decoder_outputs_gaps, _,_,_  = self.model.generate(input_tensor, generate_temp=temperature, topk=topk)

        midi_sequence = self.decode_outputs(decoder_outputs_notes, decoder_outputs_durations, decoder_outputs_gaps, shift=shift)
        
        return midi_sequence

    # the shift is used when we use EOS/SOS/PAD tokens, so 1 means we used a EOS or PAD or SOS token. In the case of RNN, we used SOS and EOS tokens, so we need to shift by 2
    def decode_outputs(self, decoder_outputs_notes, decoder_outputs_durations, decoder_outputs_gaps, shift=2):
        sequence = []
        err_count = 0
        for note, duration, gap in zip(decoder_outputs_notes[0], decoder_outputs_durations[0], decoder_outputs_gaps[0]):
                note_id = note.item()
                duration_id = duration.item()
                gap_id = gap.item()
                if note_id > 1 and duration_id > 1 and gap_id > 1:
                    try:
                        note = decode_note(note_id-shift)
                        duration = decode_duration(duration_id-shift)
                        gap = decode_gap(gap_id-shift)
                        sequence.append([note, duration, gap])
                    except:
                        err_count += 1
                        continue
                else:
                    break # EOS token reached
        if err_count > 0:
            print(f"Error count: {err_count}")
        return sequence

    def load_syllables_vocab(self, vocab_dir):
        self.syllables = Lang('syllables')
        lines = open(vocab_dir, 'r', encoding='utf-8').read().strip().split('\n')
        for syllable in lines:
            self.syllables.addWord(syllable)

    def load_config(self, config_path):
        with open(config_path, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
    
    def load_RNN_model(self, model_path):
        self.model = CustomModelRNN(
            input_size=self.syllables.n_words,
            decoder_hidden_size=self.config['model']['decoder_hidden_size'],
            encoder_hidden_size=self.config['model']['encoder_hidden_size'],
            embedding_dim=self.config['model']['embedding_dim'], 
            PAD_token=1,
            SOS_token=0, 
            MAX_LENGTH=self.config['data']['max_sequence_length'], 
            dropout_p=self.config['model']['dropout'],
            num_layers=self.config['model']['num_layers'],
            device=self.device, 
            )
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()
        
    def load_t5_model(self, model_path):
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        encoder = T5EncoderModel.from_pretrained('t5-small')
        self.model = CustomModelTransformer(
            encoder=encoder,
            PAD_token=1,
            SOS_token=0,
            device=self.device,
            MAX_LENGTH=self.config['data']['max_sequence_length'],
            train_encoder=self.config['training']['train_encoder'],
            dropout_p=self.config['model']['dropout'],
            expansion_factor=self.config['model']['expansion_factor'],
            num_heads=self.config['model']['num_heads'],
            num_layers=self.config['model']['num_layers']
        )
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()

    def serialize_lyrics_rnn(self, lyrics, max_length, eos_token):
        lyrics_tokens = []
        lyrics = lyrics.lower()
        lyrics = re.sub(r'[^a-z0-9\s]', '', lyrics)
        for syllable in lyrics.split(' ')[:max_length - 1]:
            lyrics_tokens.append(self.syllables.word2index[syllable])
        lyrics_tokens.append(eos_token)
        return lyrics_tokens
    
    def serialize_lyrics_transformer(self, lyrics, max_length):
        return self.tokenizer(lyrics, return_tensors='pt', padding=True, truncation=True, max_length=max_length)