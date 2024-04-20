import os
import re

import torch
import yaml

from dataloader.vocab import Lang
from models import CustomModelRNN
from project_utils.quantize import decode_duration, decode_gap, decode_note


class Generator:
    def __init__(self, model_dir, vocab_dir, model_name, device='cpu', eos_token=1):
        model_path = os.path.join(model_dir, model_name)
        config_path = os.path.join(model_dir, 'config.yaml')
        self.device = device
        self.eos_token = eos_token
        self.load_config(config_path)
        self.load_syllables_vocab(vocab_dir)
        self.load_model(model_path)
    
    def predict(self, lyrics, temperature=1.0, topk=None):
        inputs = [self.serialize_lyrics(lyrics, self.config['data']['max_sequence_length'], self.eos_token)]
        input_tensor = torch.tensor(inputs).to(self.device)

        with torch.no_grad():
            _, _, _, _, _, decoded_notes, decoded_durations, decoded_gaps = self.model(input_tensor, generate_temp=temperature, topk=topk)

        midi_sequence = self.decode_outputs(decoded_notes, decoded_durations, decoded_gaps)

        return midi_sequence

    def decode_outputs(self, decoder_outputs_notes, decoder_outputs_durations, decoder_outputs_gaps):
        sequence = []
        err_count = 0
        for note, duration, gap in zip(decoder_outputs_notes[0], decoder_outputs_durations[0], decoder_outputs_gaps[0]):
                note_id = note.item()
                duration_id = duration.item()
                gap_id = gap.item()
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

    def load_syllables_vocab(self, vocab_dir):
        self.syllables = Lang('syllables')
        lines = open(vocab_dir, 'r', encoding='utf-8').read().strip().split('\n')
        for syllable in lines:
            self.syllables.addWord(syllable)

    def load_config(self, config_path):
        with open(config_path, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
    
    def load_model(self, model_path):
        self.model = CustomModelRNN(
            input_size=self.syllables.n_words,
            decoder_hidden_size=self.config['model']['decoder_hidden_size'],
            encoder_hidden_size=self.config['model']['encoder_hidden_size'],
            embedding_dim=self.config['model']['embedding_dim'], 
            SOS_token=0, 
            MAX_LENGTH=self.config['data']['max_sequence_length'], 
            dropout_p=self.config['model']['dropout'],
            num_layers=self.config['model']['num_layers'],
            device=self.device, 
            )
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()

    def serialize_lyrics(self, lyrics, max_length, eos_token):
        lyrics_tokens = []
        lyrics = lyrics.lower()
        lyrics = re.sub(r'[^a-z0-9\s]', '', lyrics)
        for syllable in lyrics.split(' ')[:max_length - 1]:
            lyrics_tokens.append(self.syllables.word2index[syllable])
        lyrics_tokens.append(eos_token)
        return lyrics_tokens