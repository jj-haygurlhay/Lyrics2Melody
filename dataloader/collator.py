import re
import torch
import json
import numpy as np

from utils.quantize import encode_note, encode_duration, encode_gap
    
class SongsCollator:
    def __init__(self, syllables_lang, output_eos=1, max_length=128):
        self.syllables_lang = syllables_lang
        self.output_eos = output_eos
        self.max_length = max_length

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
        lyrics_tokens.append(self.output_eos)
        return lyrics_tokens

    def __call__(self, batch):
        all_notes, all_durations, all_gaps = [], [], []
        all_lyrics = []

        for item in batch:
            # Serialize lyrics
            lyrics_tokens = self.serialize_lyrics(item['syl_lyrics'])
            if len(lyrics_tokens) < self.max_length:
                lyrics_tokens += [self.output_eos] * (self.max_length - len(lyrics_tokens))
            all_lyrics.append(lyrics_tokens)

            # Serialize melody
            midi_seq = json.loads(item['midi_notes'])[:self.max_length - 1]
            notes, durations, gaps = self.serialize_melody(midi_seq)
            notes.append(self.output_eos)
            durations.append(self.output_eos)
            gaps.append(self.output_eos)
            if len(notes) < self.max_length:
                notes += [0] * (self.max_length - len(notes) )
                durations += [0] * (self.max_length - len(durations) )
                gaps += [0] * (self.max_length - len(gaps))
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