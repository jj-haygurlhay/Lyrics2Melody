import torch
import json

from utils.quantize import encode_note, encode_duration, encode_gap
class SongsCollator:
    def __init__(self, tokenizer, output_eos=0, max_length=128, use_syllables=False):
        self.tokenizer = tokenizer
        self.output_eos = output_eos
        self.max_length = max_length
        self.use_syllables = use_syllables

    def serialize_melody(self, midi_seq):
        notes, durations, gaps = [], [], []
        for note, duration, gap in midi_seq:
            notes.append(encode_note(note))
            durations.append(encode_duration(duration))
            gaps.append(encode_gap(gap))
        return notes, durations, gaps

    def __call__(self, batch):
        lyrics_texts = [item['syl_lyrics'] if self.use_syllables else item['lyrics'] for item in batch]
        lyrics_encoding = self.tokenizer(lyrics_texts, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)

        all_notes, all_durations, all_gaps = [], [], []
        decoder_input_ids = []

        for item in batch:
            midi_seq = json.loads(item['midi_notes'])[:self.max_length - 1]
            notes, durations, gaps = self.serialize_melody(midi_seq)

            notes.append(self.output_eos)
            durations.append(self.output_eos)
            gaps.append(self.output_eos)

            notes += [self.output_eos] * (self.max_length - len(notes))
            durations += [self.output_eos] * (self.max_length - len(durations))
            gaps += [self.output_eos] * (self.max_length - len(gaps))

            # decoder_input_ids by interleaving note, duration, and gap sequences
            interleaved_seq = []
            for n, d, g in zip(notes, durations, gaps):
                interleaved_seq.extend([n, d, g])

            decoder_input_ids.append(interleaved_seq[:self.max_length])

            all_notes.append(notes)
            all_durations.append(durations)
            all_gaps.append(gaps)

        encoding = {
            "input_ids": lyrics_encoding['input_ids'],
            "attention_mask": lyrics_encoding['attention_mask'],
            "decoder_input_ids": torch.tensor(decoder_input_ids, dtype=torch.long),
            "note_targets": torch.tensor(all_notes, dtype=torch.long),
            "duration_targets": torch.tensor(all_durations, dtype=torch.long),
            "gap_targets": torch.tensor(all_gaps, dtype=torch.long)
        }

        return encoding