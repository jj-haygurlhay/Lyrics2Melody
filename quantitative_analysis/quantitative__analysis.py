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
from utils.scale import scale, find_closest_fit
# from inference import decode_midi_sequence

SCALES = [scale(scale.MAJOR_SCALE, i) for i in range(12)] + [scale(scale.MINOR_SCALE, i) for i in range(12)]

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
    def __init__(self, model_path, model_type, dataset_path, lengths) -> None:
        # self.dataset = SongsDataset(dataset_path, )
        self.lengths = lengths
        self.dataset = pd.read_csv(dataset_path)
        self.model_type = model_type
        if model_type == "seq2seq":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to('cuda')
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.ref = reference(dataset_path, lengths)

    def generate_midi(self, number_of_lyrics, midi_per_lyrics=1):
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

    def analyse(self):
        lyrics_set = self.generated_midi.keys()
        max_len = 512

        self.analysis = [None]*max_len
        self.average = [None]*max_len
        for leng in self.lengths:
            self.analysis[leng] = dict()
            self.average[leng] = dict()
            transition_temp = []
            span_temp = []
            rep2_temp = []
            rep3_temp = []
            rep4_temp = []
            unique_temp = []
            restless_temp = []
            avg_rest_temp = []
            song_len_temp = []
            bleu2_temp = []
            bleu3_temp=[]
            bleu4_temp = []
            scale_diff_temp = []
            for id_lyrics, lyrics in enumerate(lyrics_set):
                self.analysis[leng][lyrics] = dict()
                transition_song = []
                span_song = []
                rep2_song = []
                rep3_song = []
                rep4_song = []
                unique_song = []
                restless_song = []
                avg_rest_song = []
                song_len_song = []
                scale_diff_song = []
                for midi_set in self.generated_midi[lyrics]:
                    if len(midi_set) > leng:
                        song_notes = [midi[0] for midi in midi_set[:leng+1]]
                        song_durations = [midi[1] for midi in midi_set[:leng+1]]
                        song_gaps = [midi[2] for midi in midi_set[:leng+1]]
                        span_song += [max(song_notes) - min(song_notes)]
                        rep2_song += [ngram_repetition(song_notes, 2)]
                        rep3_song += [ngram_repetition(song_notes, 3)]
                        rep4_song += [ngram_repetition(song_notes, 4)]
                        unique_song += [len(count_ngrams(song_notes, 1).keys())]
                        restless_song += [count_ngrams(song_gaps,1)[(0.0,)]]
                        avg_rest_song += [np.average(song_gaps)]
                        song_len_song += [sum(song_gaps) +sum( song_durations)]
                        scale, scale_diff = find_closest_fit(song_notes, song_durations, SCALES)
                        scale_diff_song += [scale_diff]
                        transition_song += [dict().fromkeys(range(-128, 128), 0)]
                        for bigram, count in count_ngrams(song_notes, 2).items():
                            transition_song[-1][bigram[0] - bigram[1]] += count

                self.analysis[leng][lyrics]["span"] = np.average(span_song)
                span_temp += [self.analysis[leng][lyrics]["span"]]
                self.analysis[leng][lyrics]["rep2"] = np.average(rep2_song)
                rep2_temp += [self.analysis[leng][lyrics]["rep2"]]
                self.analysis[leng][lyrics]["rep3"] = np.average(rep3_song)
                rep3_temp += [self.analysis[leng][lyrics]["rep3"]]
                self.analysis[leng][lyrics]["rep4"] = np.average(rep4_song)
                rep4_temp += [self.analysis[leng][lyrics]["rep4"]]
                self.analysis[leng][lyrics]["unique"] = np.average(unique_song)
                unique_temp += [self.analysis[leng][lyrics]["unique"]]
                self.analysis[leng][lyrics]["restless"] = np.average(restless_song)
                restless_temp += [self.analysis[leng][lyrics]["restless"]]
                self.analysis[leng][lyrics]["avg_rest"] = np.average(avg_rest_song)
                avg_rest_temp += [self.analysis[leng][lyrics]["avg_rest"]]
                self.analysis[leng][lyrics]["song_len"] = np.average(song_len_song)
                song_len_temp += [self.analysis[leng][lyrics]["song_len"]]
                reference_song_midi = self.ref.midi_set[self.ref.lyrics_set.index(lyrics)]
                reference_song_notes = [midi[0] for midi in reference_song_midi]
                reference_song_notes = [[reference_song_notes]]*len(self.generated_midi[lyrics])
                #TODO? BLEU score does not use the leng 
                generated_notes=[[midi[0] for midi in midis] for midis in self.generated_midi[lyrics]]
                self.analysis[leng][lyrics]["bleu2"] = bleu_score(generated_notes, reference_song_notes, max_n=2, weights=[1/2]*2)
                bleu2_temp += [self.analysis[leng][lyrics]["bleu2"]]
                self.analysis[leng][lyrics]["bleu3"] = bleu_score(generated_notes, reference_song_notes, max_n=3, weights=[1/3]*3)
                bleu3_temp += [self.analysis[leng][lyrics]["bleu3"]]
                self.analysis[leng][lyrics]["bleu4"] = bleu_score(generated_notes, reference_song_notes, max_n=4, weights=[1/4]*4)
                bleu4_temp += [self.analysis[leng][lyrics]["bleu4"]]

                self.analysis[leng][lyrics]["scale_diff"] = np.average(scale_diff_song)
                scale_diff_temp += [self.analysis[leng][lyrics]["scale_diff"]]
                transition_temp += [dict().fromkeys(range(-128, 128), 0)]
                for trans in transition_temp[-1].keys():
                    transition_temp[-1][trans] = np.average([transition_song[i][trans] for i in range(len(transition_song))])

            self.average[leng]["span"] = np.average(span_temp)
            self.average[leng]["rep2"] = np.average(rep2_temp)
            self.average[leng]["rep3"] = np.average(rep3_temp)
            self.average[leng]["rep4"] = np.average(rep4_temp)
            self.average[leng]["unique"] = np.average(unique_temp)
            self.average[leng]["restless"] = np.average(restless_temp)
            self.average[leng]["avg_rest"] = np.average(avg_rest_temp)
            self.average[leng]["song_len"] = np.average(song_len_temp)
            self.average[leng]["bleu2"] = np.average(bleu2_temp)
            self.average[leng]["bleu3"] = np.average(bleu3_temp)
            self.average[leng]["bleu4"] = np.average(bleu4_temp)
            self.average[leng]["scale_diff"] = np.average(scale_diff_temp)
            self.average[leng]["transitions"] = dict().fromkeys(range(-128, 128), 0)
            for trans in self.average[leng]["transitions"].keys():
                self.average[leng]["transitions"][trans] = np.average([transition_temp[i][trans] for i in range(len(transition_temp))])
            
class reference:
    def __init__(self, dataset_path, lengths) -> None:
        self.dataset = pd.read_csv(dataset_path)
        self.lyrics_set = list(self.dataset.lyrics)
        self.midi_set = [json.loads(midi) for midi in self.dataset.midi_notes] #Array(Array(triplets))
        max_len = 512 # ???

        self.reference = [None]*(max_len+1)
        self.average = [None]*(max_len+1)
        for leng in lengths:
            self.reference[leng] = dict()
            self.average[leng] = dict()
            transition_temp = []
            span_temp = []
            rep2_temp = []
            rep3_temp = []
            rep4_temp = []
            unique_temp = []
            restless_temp = []
            avg_rest_temp = []
            song_len_temp = []
            scale_diff_temp = []
            for id_lyrics, lyrics in enumerate(self.lyrics_set):
                if len(self.midi_set[id_lyrics]) > leng:
                    self.reference[leng][lyrics] = dict()
                    song_ref = self.reference[leng][lyrics]
                    song_notes = [midi[0] for midi in self.midi_set[id_lyrics][0:leng+1]]
                    song_durations = [midi[1] for midi in self.midi_set[id_lyrics][0:leng+1]]
                    song_gaps = [midi[2] for midi in self.midi_set[id_lyrics][0:leng+1]]
                    song_ref["span"] = max(song_notes) - min(song_notes)
                    span_temp += [song_ref["span"]]
                    song_ref["rep2"] = ngram_repetition(song_notes, 2)
                    rep2_temp += [song_ref["rep2"]]
                    song_ref["rep3"] = ngram_repetition(song_notes, 3)
                    rep3_temp += [song_ref["rep3"]]
                    song_ref["rep4"] = ngram_repetition(song_notes, 4)
                    rep4_temp += [song_ref["rep4"]]
                    song_ref["unique"] = len(count_ngrams(song_notes, 1).keys())
                    unique_temp += [song_ref["unique"]]
                    song_ref["restless"] = count_ngrams(song_gaps,1)[(0.0,)]
                    restless_temp += [song_ref["restless"]]
                    song_ref["avg_rest"] = np.average(song_gaps)
                    avg_rest_temp += [song_ref["avg_rest"]]
                    song_ref["song_len"] = sum(song_gaps) +sum( song_durations)
                    song_len_temp += [song_ref["song_len"]]
                    scale, scale_diff = find_closest_fit(song_notes, song_durations, SCALES)
                    song_ref["scale_diff"] = scale_diff
                    scale_diff_temp += [scale_diff]
                    transition_temp += [dict().fromkeys(range(-128, 128), 0)]
                    for bigram, count in count_ngrams(song_notes, 2).items():
                        transition_temp[-1][bigram[0] - bigram[1]] += count


            self.average[leng]["span"] = np.average(span_temp)
            self.average[leng]["rep2"] = np.average(rep2_temp)
            self.average[leng]["rep3"] = np.average(rep3_temp)
            self.average[leng]["rep4"] = np.average(rep4_temp)
            self.average[leng]["unique"] = np.average(unique_temp)
            self.average[leng]["restless"] = np.average(restless_temp)
            self.average[leng]["avg_rest"] = np.average(avg_rest_temp)
            self.average[leng]["song_len"] = np.average(song_len_temp)
            self.average[leng]["scale_diff"] = np.average(scale_diff_temp)
            self.average[leng]["transitions"] = dict().fromkeys(range(-128, 128), 0)
            for trans in self.average[leng]["transitions"].keys():
                self.average[leng]["transitions"][trans] = np.average([transition_temp[i][trans] for i in range(len(transition_temp))])

def test():
    analyser = quantitative_analysis("../runs/1-Avril-Rapport", "seq2seq", "../data/new_dataset/test.csv", [19,39,59])
    analyser.generate_midi(4,4)
    analyser.analyse()
    print(analyser.ref.average[19])
    print(analyser.average[19])

if __name__ == "__main__":
    test()

class analyser:
    def __init__(self, lyrics, notes, durations, gaps, ref_notes, ref_durations, ref_gaps):
        assert len(lyrics) == len(notes) == len(gaps) == len(durations)
        self.lyrics = lyrics
        self.notes = notes
        self.durations = durations
        self.gaps = gaps
        self.ref_notes = ref_notes
        self.ref_durations = ref_durations
        self.ref_gaps = ref_gaps
        self.results = dict()
        pass
    def analyse(self):
            self.results["span"] = []
            for lyric, note, duration, gap in zip(self.lyrics, self.notes, self.durations, self.gaps):
                self.results["span"] += [max(note) - min(note)]
                self.results["rep2"] += [ngram_repetition(note, 2)]
                self.results["rep3"] += [ngram_repetition(note, 3)]
                self.results["rep4"] += [ngram_repetition(note, 4)]
                self.results["unique"] += [[len(count_ngrams(note, 1).keys())]]
                self.results["restless"] += [count_ngrams(gap,1)[(0.0,)]]
                self.results["avg_rest"] += [np.average(gap)]
                self.results["song_len"] += [sum(duration)+ sum(gap)]
                self.results["scale_diff"] += [find_closest_fit(note, duration, SCALES)]
    def compute_references():
        pass

