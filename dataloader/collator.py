import re
import torch
import json
import numpy as np

from project_utils.quantize import encode_note, encode_duration, encode_gap, MIDI_NOTES, DURATIONS, GAPS
    
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
    
class SongsCollatorTransformer:
    def __init__(self, tokenizer, output_eos=1, output_sos=0, max_length=128, use_syllables=False):
        self.tokenizer = tokenizer
        self.output_eos = output_eos
        self.output_sos = output_sos
        self.max_length = max_length
        self.use_syllables = use_syllables

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

    def __call__(self, batch):
        lyrics_texts = [item['syl_lyrics'] if self.use_syllables else item['lyrics'] for item in batch]
        lyrics_encoding = self.tokenizer(lyrics_texts, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)

        all_notes, all_durations, all_gaps = [], [], []

        for item in batch:

            midi_seq = json.loads(item['midi_notes'])[:self.max_length - 2]
            notes, durations, gaps = self.serialize_melody(midi_seq)
            notes = [self.output_sos] + notes
            durations = [self.output_sos] + durations
            gaps = [self.output_sos] + gaps
            notes.append(self.output_eos)
            durations.append(self.output_eos)
            gaps.append(self.output_eos)
            if len(notes) < self.max_length:
                notes += [self.output_eos] * (self.max_length - len(notes) )
                durations += [self.output_eos] * (self.max_length - len(durations) )
                gaps += [self.output_eos] * (self.max_length - len(gaps))
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
    
class SongsCollatorTransformerV2:
    def __init__(self, tokenizer, max_length=128, use_syllables=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_syllables = use_syllables
        self.customize_t5_tokenizer()

    def customize_t5_tokenizer(self):
        tokens = []
        for note in MIDI_NOTES:
            for duration in DURATIONS:
                for gap in GAPS:
                    tokens.append(f"note{note}_dur{duration}_gap{gap}")

        # Add new tokens to the tokenizer
        num_added_toks = self.tokenizer.add_tokens(tokens)
        print(f"Added {num_added_toks} new tokens.")

    def serialize_melody(self, midi_seq):
        serialized_seq = []
        for note, duration, gap in midi_seq:
            token = f"note{note}_dur{duration}_gap{gap}"
            serialized_seq.append(token)
        return serialized_seq

    def __call__(self, batch):
        lyrics_texts = [item['syl_lyrics'] if self.use_syllables else item['lyrics'] for item in batch]
        lyrics_encoding = self.tokenizer(lyrics_texts, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)

        all_midi_tokens = []

        for item in batch:
            midi_seq = json.loads(item['midi_notes'])[:self.max_length]
            midi_tokens = self.serialize_melody(midi_seq)

            midi_encoding = self.tokenizer(' '.join(midi_tokens), return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)

            all_midi_tokens.append(midi_encoding['input_ids'].squeeze(0))  # Remove batch dimension since we are creating one manually

        # input_ids for labels as T5 expects the decoder input in the form of token IDs
        encoding = {
            "input_ids": lyrics_encoding['input_ids'],
            "attention_mask": lyrics_encoding['attention_mask'],
            "labels": torch.stack(all_midi_tokens)  # Stack to create a single tensor for all MIDI sequences
        }

        return encoding