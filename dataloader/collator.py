import torch
import json
import numpy as np

class SongsCollator:
    def __init__(self, tokenizer, output_eos="[PAD]", max_length=128, use_syllables=False):
        self.tokenizer = tokenizer
        self.output_eos = output_eos
        self.max_length = max_length
        self.use_syllables = use_syllables

    def serialize_melody(self, midi_seq):
        """
        Serialize MIDI sequence to a string format for tokenization.
        Each note-duration-gap triplet is concatenated into a single string with separators.
        """
        serialized_seq = []
        for note, duration, gap in midi_seq:
            note_token = f"<note{note}>"
            duration_token = f"<duration{duration}>"
            gap_token = f"<gap{gap}>"
            serialized_seq.append(f"{note_token} {duration_token} {gap_token}")
        return "_".join(serialized_seq)

    def __call__(self, batch):
        lyrics_texts = [item['syl_lyrics'] if self.use_syllables else item['lyrics'] for item in batch]
        lyrics_encoding = self.tokenizer(lyrics_texts, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)

        target_texts = []

        for item in batch:
            midi_seq = json.loads(item['midi_notes'])[:self.max_length]
            serialized_melody = self.serialize_melody(midi_seq)
            target_texts.append(serialized_melody)

        # Note: T5's tokenizer will handle padding of tokenized sequences to max_length
        targets_encoding = self.tokenizer(target_texts, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)

        # input_ids for labels as T5 expects the decoder input in the form of token IDs
        encoding = {
            "input_ids": lyrics_encoding['input_ids'],
            "attention_mask": lyrics_encoding['attention_mask'],
            "labels": targets_encoding['input_ids']
        }

        return encoding