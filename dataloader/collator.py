import torch
import json

class SongsCollator:
    def __init__(self, tokenizer, output_eos=[0,0,0], max_length=128, use_syllables=False):
        self.tokenizer = tokenizer
        self.output_eos = output_eos
        self.max_length = max_length
        self.use_syllables = use_syllables

    def __call__(self, batch):
        if self.use_syllables:
            lyrics = [item['syl_lyrics'] for item in batch]
        else:
            lyrics = [item['lyrics'] for item in batch]

        # Tokenize the lyrics
        encoding = self.tokenizer(lyrics, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)

        # Add the midi notes (labels) to the encoding
        midi_notes = []
        for item in batch:
            # Truncate the midi notes to the max_length
            midi_seq = json.loads(item['midi_notes'])[:self.max_length]
            
            # Pad the midi notes to the max_length
            midi_seq += [self.output_eos] * (self.max_length - len(midi_seq))
            
            midi_notes.append(midi_seq)

        encoding['midi_notes'] = torch.tensor(midi_notes)

        return encoding