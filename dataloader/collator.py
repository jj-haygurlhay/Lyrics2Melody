import torch

class SongsCollator:
    def __init__(self, tokenizer, max_length=128, use_syllables=False):
        self.tokenizer = tokenizer
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
        midi_notes = [item['midi_notes'] for item in batch]
        encoding['midi_notes'] = torch.tensor(midi_notes)

        return encoding