import re
import torch
import json
import numpy as np

from project_utils.quantize import encode_note, encode_duration, encode_gap, MIDI_NOTES, DURATIONS, GAPS
    
class SongsCollator:
    def __init__(self, syllables_lang, SOS_token, EOS_token, max_length=128, octave_shift_percentage=0):
        self.syllables_lang = syllables_lang
        self.SOS_token = SOS_token
        self.EOS_token = EOS_token
        self.max_length = max_length
        self.octave_shift_percentage = octave_shift_percentage

    def serialize_melody(self, midi_seq):
        """
        Serialize MIDI sequence to a string format for tokenization.
        Each note-duration-gap triplet is concatenated into a single string with separators.
        """
        notes, durations, gaps = [], [], []
        for note, duration, gap in midi_seq:
            notes.append(encode_note(note) + 2)
            durations.append(encode_duration(duration) + 2)
            gaps.append(encode_gap(gap) + 2)
        return notes, durations, gaps
    
    def serialize_lyrics(self, lyrics):
        lyrics_tokens = []
        lyrics = lyrics.lower()
        lyrics = re.sub(r'[^a-z0-9\s]', '', lyrics)
        for syllable in lyrics.split(' ')[:self.max_length - 1]:
            lyrics_tokens.append(self.syllables_lang.word2index[syllable])
        # lyrics_tokens.append(self.EOS_token)
        return lyrics_tokens

    def __call__(self, batch):
        all_notes, all_durations, all_gaps = [], [], []
        all_lyrics = []

        # Randomly shift the melody by an octave for a percentage of the batch
        octave_shift_ids = np.random.choice(len(batch), int(len(batch) * self.octave_shift_percentage), replace=False)

        for i, item in enumerate(batch):
            # Serialize lyrics
            lyrics_tokens = self.serialize_lyrics(item['syl_lyrics'])
            if len(lyrics_tokens) < self.max_length:
                lyrics_tokens += [self.EOS_token] * (self.max_length - len(lyrics_tokens))
            all_lyrics.append(lyrics_tokens)

            # Serialize melody
            midi_seq = json.loads(item['midi_notes'])[:self.max_length - 1]
            notes, durations, gaps = self.serialize_melody(midi_seq)
            if i in octave_shift_ids:
                shift = np.random.choice([-1,1]) * np.random.choice([2, 4, 6, 8, 10, 12]) # Randomly add or subtract tones
                if np.min(notes) < np.abs(shift):
                    shift = np.abs(shift)
                elif np.max(notes) + np.abs(shift) > MIDI_NOTES[-1]:
                    shift = -np.abs(shift)

                notes = [note + shift for note in notes]

            notes = [self.SOS_token] + notes
            durations = [self.SOS_token] + durations
            gaps = [self.SOS_token] + gaps
            
            if len(notes) < self.max_length:
                notes += [self.EOS_token] * (self.max_length - len(notes) )
                durations += [self.EOS_token] * (self.max_length - len(durations) )
                gaps += [self.EOS_token] * (self.max_length - len(gaps))
                
            all_notes.append(notes)
            all_durations.append(durations)
            all_gaps.append(gaps)


        encoding = {
            "input_ids": torch.tensor(all_lyrics),
            "labels": {
                "notes": torch.tensor(all_notes),
                "durations": torch.tensor(all_durations),
                "gaps": torch.tensor(all_gaps)
            }
        }

        return encoding
    
class SongsCollatorTransformer:
    def __init__(self, tokenizer, SOS_token, EOS_token, max_length=128, use_syllables=False, octave_shift_percentage=0):
        self.tokenizer = tokenizer
        self.SOS_token = SOS_token
        self.EOS_token = EOS_token
        self.max_length = max_length
        self.use_syllables = use_syllables
        self.octave_shift_percentage = octave_shift_percentage

    def serialize_melody(self, midi_seq):
        """
        Serialize MIDI sequence to a string format for tokenization.
        Each note-duration-gap triplet is concatenated into a single string with separators.
        """
        notes, durations, gaps = [], [], []
        for note, duration, gap in midi_seq:
            notes.append(int(note) + 2)
            durations.append(encode_duration(duration) + 2)
            gaps.append(encode_gap(gap) + 2)
        return notes, durations, gaps

    def __call__(self, batch):
        lyrics_texts = [item['syl_lyrics'] if self.use_syllables else item['lyrics'] for item in batch]
        lyrics_encoding = self.tokenizer(lyrics_texts, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)

        all_notes, all_durations, all_gaps = [], [], []

        # Randomly shift the melody by an octave for a percentage of the batch
        octave_shift_ids = np.random.choice(len(batch), int(len(batch) * self.octave_shift_percentage), replace=False)

        for i, item in enumerate(batch):

            midi_seq = json.loads(item['midi_notes'])[:self.max_length - 1]
            notes, durations, gaps = self.serialize_melody(midi_seq)
            
            if i in octave_shift_ids:
                shift = np.random.choice([-1,1]) * np.random.choice([2, 4, 6, 8, 10, 12]) # Randomly add or subtract tones
                if np.min(notes) < np.abs(shift):
                    shift = np.abs(shift)
                elif np.max(notes) + np.abs(shift) > MIDI_NOTES[-1]:
                    shift = -np.abs(shift)

                notes = [note + shift for note in notes]
            
            notes = [self.SOS_token] + notes
            durations = [self.SOS_token] + durations
            gaps = [self.SOS_token] + gaps
            
            if len(notes) < self.max_length:
                notes += [self.EOS_token] * (self.max_length - len(notes) )
                durations += [self.EOS_token] * (self.max_length - len(durations) )
                gaps += [self.EOS_token] * (self.max_length - len(gaps))
            all_notes.append(notes)
            all_durations.append(durations)
            all_gaps.append(gaps)


        # input_ids for labels as T5 expects the decoder input in the form of token IDs
        encoding = {
            "input_ids": lyrics_encoding['input_ids'],
            "attention_mask": lyrics_encoding['attention_mask'],
            "labels": {
                "notes": torch.tensor(all_notes),
                "durations": torch.tensor(all_durations),
                "gaps": torch.tensor(all_gaps)
            }
        }

        return encoding