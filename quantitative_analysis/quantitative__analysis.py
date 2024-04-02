import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)
import pandas as pd
from utils.quantize import decode_note, decode_duration, decode_gap
import torch
import numpy as np
from utils.ngram import ngram_repetition, count_ngrams
from utils.BLEUscore import bleu_score
from dataloader.dataset import SongsDataset
import json
# from inference import decode_midi_sequence

def decode_midi_sequence(decoded_output):
    sequence = []
    note, duration, gap = -1, -1, -1
    err_count = 0
    for token in decoded_output.split('<'):
        try:
            if 'note' in token and duration == -1 and gap == -1:
                duration = -1
                gap = -1
                note = decode_note(int(token[4:-1]))
            elif 'duration' in token and note != -1:
                gap = -1
                duration = decode_duration(int(token[8:-1]))
            elif 'gap' in token and note != -1 and duration != -1:
                gap = decode_gap(int(token[3:-1]))
                sequence.append([note, duration, gap])
                note, duration, gap = -1, -1, -1
            else:
                note, duration, gap = -1, -1, -1
                err_count += 1
        except:
            print(decoded_output)
            print(f"Error decoding token {token}")
    if err_count > 0:
        print(f"Error count: {err_count}")
    return sequence

class quantitative_analysis:
    def __init__(self, model_path, model_type, dataset_path) -> None:
        # self.dataset = SongsDataset(dataset_path, )
        self.dataset = pd.read_csv(dataset_path)
        self.model_type = model_type
        if model_type == "seq2seq":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to('cuda')
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def generate_midi(self, midi_per_lyrics, number_of_lyrics):
        self.midi_per_lyrics = midi_per_lyrics
        self.number_of_lyrics = number_of_lyrics
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        inputs = ['Generate notes: ' + lyrics for lyrics in self.dataset.lyrics[0:number_of_lyrics]]*midi_per_lyrics
        lyrics_ref = [lyrics for lyrics in self.dataset.lyrics[0:number_of_lyrics]]*midi_per_lyrics
        decoded_outputs = []
        for input in inputs:
            input = self.tokenizer([input], truncation=True, padding=True, max_length=512, return_tensors='pt').to(device)
            outputs = self.model.generate(**input, num_beams=8, do_sample=True, min_length=10, max_length=512)
            decoded_outputs += self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        self.generated_midi = dict.fromkeys(self.dataset.lyrics[0:number_of_lyrics], [])
        for (decoded_seq, lyrics) in zip(decoded_outputs, lyrics_ref):
            self.generated_midi[lyrics] += [decode_midi_sequence(decoded_seq)]

    def analyse(self, midi_per_lyrics, number_of_lyrics):
        assert midi_per_lyrics <= self.midi_per_lyrics, 'Must analyse <= than computed'
        assert number_of_lyrics <= self.number_of_lyrics, 'Must analyse <= than computed'

        analysis = lambda:0
        analysis.midi_per_lyrics = midi_per_lyrics
        analysis.number_of_lyrics = number_of_lyrics
        analysis.lyrics_set = self.dataset.lyrics[0:number_of_lyrics]

        analysis.span = dict.fromkeys(analysis.lyrics_set)
        analysis.rep2 = dict.fromkeys(analysis.lyrics_set)
        analysis.rep3 = dict.fromkeys(analysis.lyrics_set)
        analysis.rep4 = dict.fromkeys(analysis.lyrics_set)
        analysis.unique = dict.fromkeys(analysis.lyrics_set)
        analysis.restless = dict.fromkeys(analysis.lyrics_set)
        analysis.avg_rest = dict.fromkeys(analysis.lyrics_set)
        analysis.song_len = dict.fromkeys(analysis.lyrics_set)
        analysis.bleu2_notes = dict.fromkeys(analysis.lyrics_set)
        analysis.bleu3_notes = dict.fromkeys(analysis.lyrics_set)
        analysis.bleu4_notes = dict.fromkeys(analysis.lyrics_set)
        for lyrics in analysis.lyrics_set:
            notes = [[self.generated_midi[lyrics][j][i][0] for i in range(len(self.generated_midi[lyrics][j]))]for j in range(midi_per_lyrics)]
            durations = [[self.generated_midi[lyrics][j][i][1] for i in range(len(self.generated_midi[lyrics][j]))]for j in range(midi_per_lyrics)]
            gaps = [[self.generated_midi[lyrics][j][i][2] for i in range(len(self.generated_midi[lyrics][j]))]for j in range(midi_per_lyrics)]
            analysis.span[lyrics] = np.average([max(notes[i])-min(notes[i]) for i in range(len(notes))])
            analysis.rep2[lyrics] = np.average([ngram_repetition(notes[i],2) for i in range(len(notes))])
            analysis.rep3[lyrics] = np.average([ngram_repetition(notes[i],3) for i in range(len(notes))])
            analysis.rep4[lyrics] = np.average([ngram_repetition(notes[i],4) for i in range(len(notes))])
            analysis.unique[lyrics] = np.average([len(count_ngrams(notes[i], 1).keys()) for i in range(len(notes))])
            analysis.restless[lyrics] = np.average([count_ngrams(gaps[i],1)[(0.0,)]for i in range(len(gaps))])
            analysis.avg_rest[lyrics] = np.average([np.average(gaps[i])for i in range(len(gaps))])
            analysis.song_len[lyrics] = np.average([sum(gaps[i])+sum(durations[i]) for i in range(len(gaps))])
            ind = list(self.dataset.lyrics).index(lyrics)
            reference = json.loads(self.dataset.midi_notes[ind])
            analysis.bleu2_notes[lyrics] = bleu_score(notes, [reference]*len(notes), max_n=2, weights=[1/2]*2)
            analysis.bleu3_notes[lyrics] = bleu_score(notes, [reference]*len(notes), max_n=3, weights=[1/3]*3)
            analysis.bleu4_notes[lyrics] = bleu_score(notes, [reference]*len(notes), max_n=4, weights=[1/4]*4)
        
        return analysis


    





