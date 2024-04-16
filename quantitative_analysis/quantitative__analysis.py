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
from utils.ngram import ngram_repetition, count_ngrams, transitions, transition_map
from utils.BLEUscore import bleu_score
from dataloader.dataset import SongsDataset
import json
from utils.scale import scale, find_closest_fit
from collections import defaultdict
from multiprocessing import Pool
from statistics import mean
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

# class quantitative_analysis:
#     def __init__(self, model_path, model_type, dataset_path, lengths) -> None:
#         # self.dataset = SongsDataset(dataset_path, )
#         self.lengths = lengths
#         self.dataset = pd.read_csv(dataset_path)
#         self.model_type = model_type
#         if model_type == "seq2seq":
#             self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to('cuda')
#             self.tokenizer = AutoTokenizer.from_pretrained(model_path)
#         self.ref = reference(dataset_path, lengths)

#     def generate_midi(self, number_of_lyrics, midi_per_lyrics=1):
#         self.midi_per_lyrics = midi_per_lyrics
#         self.number_of_lyrics = number_of_lyrics
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         inputs = ['Generate notes: ' + lyrics for lyrics in self.dataset.lyrics[0:number_of_lyrics]]*midi_per_lyrics
#         lyrics_ref = [lyrics for lyrics in self.dataset.lyrics[0:number_of_lyrics]]*midi_per_lyrics
#         decoded_outputs = []
#         for input in inputs:
#             input = self.tokenizer([input], truncation=True, padding=True, max_length=512, return_tensors='pt').to(device)
#             outputs = self.model.generate(**input, num_beams=8, do_sample=True, min_length=10, max_length=512)
#             decoded_outputs += self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

#         self.generated_midi = dict.fromkeys(self.dataset.lyrics[0:number_of_lyrics], [])
#         for (decoded_seq, lyrics) in zip(decoded_outputs, lyrics_ref):
#             self.generated_midi[lyrics] += [decode_midi_sequence(decoded_seq)]

#     def analyse(self):
#         lyrics_set = self.generated_midi.keys()
#         max_len = 512

#         self.analysis = [None]*max_len
#         self.average = [None]*max_len
#         for leng in self.lengths:
#             self.analysis[leng] = dict()
#             self.average[leng] = dict()
#             transition_temp = []
#             span_temp = []
#             rep2_temp = []
#             rep3_temp = []
#             rep4_temp = []
#             unique_temp = []
#             restless_temp = []
#             avg_rest_temp = []
#             song_len_temp = []
#             bleu2_temp = []
#             bleu3_temp=[]
#             bleu4_temp = []
#             scale_diff_temp = []
#             for id_lyrics, lyrics in enumerate(lyrics_set):
#                 self.analysis[leng][lyrics] = dict()
#                 transition_song = []
#                 span_song = []
#                 rep2_song = []
#                 rep3_song = []
#                 rep4_song = []
#                 unique_song = []
#                 restless_song = []
#                 avg_rest_song = []
#                 song_len_song = []
#                 scale_diff_song = []
#                 for midi_set in self.generated_midi[lyrics]:
#                     if len(midi_set) > leng:
#                         song_notes = [midi[0] for midi in midi_set[:leng+1]]
#                         song_durations = [midi[1] for midi in midi_set[:leng+1]]
#                         song_gaps = [midi[2] for midi in midi_set[:leng+1]]
#                         span_song += [max(song_notes) - min(song_notes)]
#                         rep2_song += [ngram_repetition(song_notes, 2)]
#                         rep3_song += [ngram_repetition(song_notes, 3)]
#                         rep4_song += [ngram_repetition(song_notes, 4)]
#                         unique_song += [len(count_ngrams(song_notes, 1).keys())]
#                         restless_song += [count_ngrams(song_gaps,1)[(0.0,)]]
#                         avg_rest_song += [np.average(song_gaps)]
#                         song_len_song += [sum(song_gaps) +sum( song_durations)]
#                         scale, scale_diff = find_closest_fit(song_notes, song_durations, SCALES)
#                         scale_diff_song += [scale_diff]
#                         transition_song += [dict().fromkeys(range(-128, 128), 0)]
#                         for bigram, count in count_ngrams(song_notes, 2).items():
#                             transition_song[-1][bigram[0] - bigram[1]] += count

#                 self.analysis[leng][lyrics]["span"] = np.average(span_song)
#                 span_temp += [self.analysis[leng][lyrics]["span"]]
#                 self.analysis[leng][lyrics]["rep2"] = np.average(rep2_song)
#                 rep2_temp += [self.analysis[leng][lyrics]["rep2"]]
#                 self.analysis[leng][lyrics]["rep3"] = np.average(rep3_song)
#                 rep3_temp += [self.analysis[leng][lyrics]["rep3"]]
#                 self.analysis[leng][lyrics]["rep4"] = np.average(rep4_song)
#                 rep4_temp += [self.analysis[leng][lyrics]["rep4"]]
#                 self.analysis[leng][lyrics]["unique"] = np.average(unique_song)
#                 unique_temp += [self.analysis[leng][lyrics]["unique"]]
#                 self.analysis[leng][lyrics]["restless"] = np.average(restless_song)
#                 restless_temp += [self.analysis[leng][lyrics]["restless"]]
#                 self.analysis[leng][lyrics]["avg_rest"] = np.average(avg_rest_song)
#                 avg_rest_temp += [self.analysis[leng][lyrics]["avg_rest"]]
#                 self.analysis[leng][lyrics]["song_len"] = np.average(song_len_song)
#                 song_len_temp += [self.analysis[leng][lyrics]["song_len"]]
#                 reference_song_midi = self.ref.midi_set[self.ref.lyrics_set.index(lyrics)]
#                 reference_song_notes = [midi[0] for midi in reference_song_midi]
#                 reference_song_notes = [[reference_song_notes]]*len(self.generated_midi[lyrics])
#                 #TODO? BLEU score does not use the leng 
#                 generated_notes=[[midi[0] for midi in midis] for midis in self.generated_midi[lyrics]]
#                 self.analysis[leng][lyrics]["bleu2"] = bleu_score(generated_notes, reference_song_notes, max_n=2, weights=[1/2]*2)
#                 bleu2_temp += [self.analysis[leng][lyrics]["bleu2"]]
#                 self.analysis[leng][lyrics]["bleu3"] = bleu_score(generated_notes, reference_song_notes, max_n=3, weights=[1/3]*3)
#                 bleu3_temp += [self.analysis[leng][lyrics]["bleu3"]]
#                 self.analysis[leng][lyrics]["bleu4"] = bleu_score(generated_notes, reference_song_notes, max_n=4, weights=[1/4]*4)
#                 bleu4_temp += [self.analysis[leng][lyrics]["bleu4"]]

#                 self.analysis[leng][lyrics]["scale_diff"] = np.average(scale_diff_song)
#                 scale_diff_temp += [self.analysis[leng][lyrics]["scale_diff"]]
#                 transition_temp += [dict().fromkeys(range(-128, 128), 0)]
#                 for trans in transition_temp[-1].keys():
#                     transition_temp[-1][trans] = np.average([transition_song[i][trans] for i in range(len(transition_song))])

#             self.average[leng]["span"] = np.average(span_temp)
#             self.average[leng]["rep2"] = np.average(rep2_temp)
#             self.average[leng]["rep3"] = np.average(rep3_temp)
#             self.average[leng]["rep4"] = np.average(rep4_temp)
#             self.average[leng]["unique"] = np.average(unique_temp)
#             self.average[leng]["restless"] = np.average(restless_temp)
#             self.average[leng]["avg_rest"] = np.average(avg_rest_temp)
#             self.average[leng]["song_len"] = np.average(song_len_temp)
#             self.average[leng]["bleu2"] = np.average(bleu2_temp)
#             self.average[leng]["bleu3"] = np.average(bleu3_temp)
#             self.average[leng]["bleu4"] = np.average(bleu4_temp)
#             self.average[leng]["scale_diff"] = np.average(scale_diff_temp)
#             self.average[leng]["transitions"] = dict().fromkeys(range(-128, 128), 0)
#             for trans in self.average[leng]["transitions"].keys():
#                 self.average[leng]["transitions"][trans] = np.average([transition_temp[i][trans] for i in range(len(transition_temp))])
            
# class reference:
#     def __init__(self, dataset_path, lengths) -> None:
#         self.dataset = pd.read_csv(dataset_path)
#         self.lyrics_set = list(self.dataset.lyrics)
#         self.midi_set = [json.loads(midi) for midi in self.dataset.midi_notes] #Array(Array(triplets))
#         max_len = 512 # ???

#         self.reference = [None]*(max_len+1)
#         self.average = [None]*(max_len+1)
#         for leng in lengths:
#             self.reference[leng] = dict()
#             self.average[leng] = dict()
#             transition_temp = []
#             span_temp = []
#             rep2_temp = []
#             rep3_temp = []
#             rep4_temp = []
#             unique_temp = []
#             restless_temp = []
#             avg_rest_temp = []
#             song_len_temp = []
#             scale_diff_temp = []
#             for id_lyrics, lyrics in enumerate(self.lyrics_set):
#                 if len(self.midi_set[id_lyrics]) > leng:
#                     self.reference[leng][lyrics] = dict()
#                     song_ref = self.reference[leng][lyrics]
#                     song_notes = [midi[0] for midi in self.midi_set[id_lyrics][0:leng+1]]
#                     song_durations = [midi[1] for midi in self.midi_set[id_lyrics][0:leng+1]]
#                     song_gaps = [midi[2] for midi in self.midi_set[id_lyrics][0:leng+1]]
#                     song_ref["span"] = max(song_notes) - min(song_notes)
#                     span_temp += [song_ref["span"]]
#                     song_ref["rep2"] = ngram_repetition(song_notes, 2)
#                     rep2_temp += [song_ref["rep2"]]
#                     song_ref["rep3"] = ngram_repetition(song_notes, 3)
#                     rep3_temp += [song_ref["rep3"]]
#                     song_ref["rep4"] = ngram_repetition(song_notes, 4)
#                     rep4_temp += [song_ref["rep4"]]
#                     song_ref["unique"] = len(count_ngrams(song_notes, 1).keys())
#                     unique_temp += [song_ref["unique"]]
#                     song_ref["restless"] = count_ngrams(song_gaps,1)[(0.0,)]
#                     restless_temp += [song_ref["restless"]]
#                     song_ref["avg_rest"] = np.average(song_gaps)
#                     avg_rest_temp += [song_ref["avg_rest"]]
#                     song_ref["song_len"] = sum(song_gaps) +sum( song_durations)
#                     song_len_temp += [song_ref["song_len"]]
#                     scale, scale_diff = find_closest_fit(song_notes, song_durations, SCALES)
#                     song_ref["scale_diff"] = scale_diff
#                     scale_diff_temp += [scale_diff]
#                     transition_temp += [dict().fromkeys(range(-128, 128), 0)]
#                     for bigram, count in count_ngrams(song_notes, 2).items():
#                         transition_temp[-1][bigram[0] - bigram[1]] += count


#             self.average[leng]["span"] = np.average(span_temp)
#             self.average[leng]["rep2"] = np.average(rep2_temp)
#             self.average[leng]["rep3"] = np.average(rep3_temp)
#             self.average[leng]["rep4"] = np.average(rep4_temp)
#             self.average[leng]["unique"] = np.average(unique_temp)
#             self.average[leng]["restless"] = np.average(restless_temp)
#             self.average[leng]["avg_rest"] = np.average(avg_rest_temp)
#             self.average[leng]["song_len"] = np.average(song_len_temp)
#             self.average[leng]["scale_diff"] = np.average(scale_diff_temp)
#             self.average[leng]["transitions"] = dict().fromkeys(range(-128, 128), 0)
#             for trans in self.average[leng]["transitions"].keys():
#                 self.average[leng]["transitions"][trans] = np.average([transition_temp[i][trans] for i in range(len(transition_temp))])

class analyser:
    def __init__(self, lyrics, notes, durations, gaps, ref_notes, ref_durations, ref_gaps):
        """
        All input arrays should be aligned so that every entry on a single index corresponds to a single song
        """
        assert len(lyrics) == len(notes) == len(gaps) == len(durations) == len(ref_notes) == len(ref_durations) == len(ref_gaps), "All arrays of inputs must correspond between each other. i.e.: they must all have the same number of elements"
        self.lyrics = lyrics
        self.notes = notes
        self.durations = durations
        self.gaps = gaps
        self.ref_notes = ref_notes
        self.ref_durations = ref_durations
        self.ref_gaps = ref_gaps
        self.individual_results = defaultdict(list)
        self.average_results = dict()
        self.individual_references = defaultdict(list)
        self.average_references = dict()
        
    def analyse(self):
        # Individual songs generated
        for lyric, note, duration, gap, ref_note, ref_duration, ref_gap in zip(self.lyrics, self.notes, self.durations, self.gaps, self.ref_notes, self.ref_durations, self.ref_gaps):
            self.individual_results["span"] += [max(note) - min(note)]
            self.individual_results["rep2"] += [ngram_repetition(note, 2)]
            self.individual_results["rep3"] += [ngram_repetition(note, 3)]
            self.individual_results["rep4"] += [ngram_repetition(note, 4)]
            self.individual_results["unique"] += [[len(count_ngrams(note, 1).keys())]]
            self.individual_results["restless"] += [count_ngrams(gap,1)[(0.0,)]]
            self.individual_results["avg_rest"] += [np.average(gap)]
            self.individual_results["song_len"] += [sum(duration)+ sum(gap)]
            self.individual_results["scale_diff"] += [find_closest_fit(note, duration, SCALES)]
            self.individual_results["transitions"] += [self.compute_transitions(note)]
            self.individual_results["transitions_test"] += [transition_map(note)]
            self.individual_results["bleu2"] += [bleu_score(note, ref_note, max_n=2)]
            self.individual_results["bleu3"] += [bleu_score(note, ref_note, max_n=3)]
            self.individual_results["bleu4"] += [bleu_score(note, ref_note, max_n=4)]
        
        # Averages of generated
        self.average_results["span"] = np.average(self.individual_results["span"])
        self.average_results["rep2"] = np.average(self.individual_results["rep2"])
        self.average_results["rep3"] = np.average(self.individual_results["rep3"])
        self.average_results["rep4"] = np.average(self.individual_results["rep4"])
        self.average_results["unique"] = np.average(self.individual_results["unique"])
        self.average_results["restless"] = np.average(self.individual_results["restless"])
        self.average_results["avg_rest"] = np.average(self.individual_results["avg_rest"])
        self.average_results["song_len"] = np.average(self.individual_results["song_len"])
        self.average_results["scale_diff"] = np.average(self.individual_results["scale_diff"])
        self.average_results["transitions"] = self.average_transitions(self.individual_results["transitions"])
        self.average_results["transitions_test"] = sum(self.individual_results["transitions_test"])/len(self.individual_results["transitions_test"])
        self.average_results["bleu2"] = np.average(self.individual_results["bleu2"])
        self.average_results["bleu3"] = np.average(self.individual_results["bleu3"])
        self.average_results["bleu4"] = np.average(self.individual_results["bleu4"])
        return self.average_results

    def analyse_multi(self):
        # Individual songs generated
        with Pool(len(os.sched_getaffinity(0))) as p:
            self.individual_results["span"] = p.map(lambda note: max(note) - min(note), self.notes)
            self.individual_results["rep2"] = p.map(lambda note: ngram_repetition(note, 2), self.notes) 
            self.individual_results["rep3"] = p.map(lambda note: ngram_repetition(note, 3), self.notes) 
            self.individual_results["rep4"] = p.map(lambda note: ngram_repetition(note, 4), self.notes) 
            self.individual_results["unique"] = p.map(lambda note : len(count_ngrams(note, 1).keys()), self.notes)
            self.individual_results["restless"] = p.map(lambda gap : count_ngrams(gap, 1)[(0.0,)], self.gaps) 
            self.individual_results["avg_rest"] = p.map(lambda gap : np.average(gap), self.gaps)
            self.individual_results["song_len"] = p.map(lambda duration, gap : sum(duration)+sum(gap), self.gaps)
            self.individual_results["scale_diff"] = p.map(lambda note, duration : find_closest_fit(note, duration, SCALES), zip(self.notes, self.durations))
            self.individual_results["transitions"] = p.map(lambda note: transitions(note), self.notes)
            self.individual_results["transitions_test"] = p.map(lambda note: transition_map(note), self.notes)
            self.individual_results["bleu2"] = p.map(lambda note, ref_note : bleu_score(note, ref_note, max_n=2), zip(self.notes, self.ref_notes))
            self.individual_results["bleu3"] = p.map(lambda note, ref_note : bleu_score(note, ref_note, max_n=3), zip(self.notes, self.ref_notes))
            self.individual_results["bleu4"] = p.map(lambda note, ref_note : bleu_score(note, ref_note, max_n=4), zip(self.notes, self.ref_notes))
        
        # Averages of generated
        self.average_results["span"] = np.average(self.individual_results["span"])
        self.average_results["rep2"] = np.average(self.individual_results["rep2"])
        self.average_results["rep3"] = np.average(self.individual_results["rep3"])
        self.average_results["rep4"] = np.average(self.individual_results["rep4"])
        self.average_results["unique"] = np.average(self.individual_results["unique"])
        self.average_results["restless"] = np.average(self.individual_results["restless"])
        self.average_results["avg_rest"] = np.average(self.individual_results["avg_rest"])
        self.average_results["song_len"] = np.average(self.individual_results["song_len"])
        self.average_results["scale_diff"] = np.average(self.individual_results["scale_diff"])
        self.average_results["transitions"] = self.average_transitions(self.individual_results["transitions"])
        self.average_results["transitions_test"] = sum(self.individual_results["transitions_test"])/len(self.individual_results["transitions_test"])
        self.average_results["bleu2"] = np.average(self.individual_results["bleu2"])
        self.average_results["bleu3"] = np.average(self.individual_results["bleu3"])
        self.average_results["bleu4"] = np.average(self.individual_results["bleu4"])
        return self.average_results
    
    def references(self):
        # Individual reference songs
        for ref_note, ref_duration, ref_gap in zip(self.ref_notes, self.ref_durations, self.ref_gaps):
            self.individual_references["span"] += [max(ref_note) - min(ref_note)]
            self.individual_references["rep2"] += [ngram_repetition(ref_note, 2)]
            self.individual_references["rep3"] += [ngram_repetition(ref_note, 3)]
            self.individual_references["rep4"] += [ngram_repetition(ref_note, 4)]
            self.individual_references["unique"] += [[len(count_ngrams(ref_note, 1).keys())]]
            self.individual_references["restless"] += [count_ngrams(ref_gap,1)[(0.0,)]]
            self.individual_references["avg_rest"] += [np.average(ref_gap)]
            self.individual_references["song_len"] += [sum(ref_duration)+ sum(ref_gap)]
            self.individual_references["scale_diff"] += [find_closest_fit(ref_note, ref_duration, SCALES)]
            self.individual_references["transitions"] += [self.compute_transitions(ref_note)]
            self.individual_references["transitions_test"] += [transition_map(ref_note)]

        # Average of reference songs
        self.average_references["span"] = np.average(self.individual_references["span"])
        self.average_references["rep2"] = np.average(self.individual_references["rep2"])
        self.average_references["rep3"] = np.average(self.individual_references["rep3"])
        self.average_references["rep4"] = np.average(self.individual_references["rep4"])
        self.average_references["unique"] = np.average(self.individual_references["unique"])
        self.average_references["restless"] = np.average(self.individual_references["restless"])
        self.average_references["avg_rest"] = np.average(self.individual_references["avg_rest"])
        self.average_references["song_len"] = np.average(self.individual_references["song_len"])
        self.average_references["scale_diff"] = np.average(self.individual_references["scale_diff"])
        self.average_references["transitions"] = self.average_transitions(self.individual_references["transitions"])
        self.average_references["transitions_test"] = sum(self.individual_references["transitions_test"])/len(self.individual_references["transitions_test"])
        return self.average_references
    
    def references_multi(self):
        # Individual reference songs
        with Pool(len(os.sched_getaffinity(0))) as p:
            self.individual_references["span"] = p.map(lambda note: max(note) - min(note), self.ref_notes)
            self.individual_references["rep2"] = p.map(lambda note: ngram_repetition(note, 2), self.ref_notes) 
            self.individual_references["rep3"] = p.map(lambda note: ngram_repetition(note, 3), self.ref_notes) 
            self.individual_references["rep4"] = p.map(lambda note: ngram_repetition(note, 4), self.ref_notes) 
            self.individual_references["unique"] = p.map(lambda note : len(count_ngrams(note, 1).keys()), self.ref_notes)
            self.individual_references["restless"] = p.map(lambda gap : count_ngrams(gap, 1)[(0.0,)], self.ref_gaps) 
            self.individual_references["avg_rest"] = p.map(lambda gap : np.average(gap), self.ref_gaps)
            self.individual_references["song_len"] = p.map(lambda duration, gap : sum(duration)+sum(gap), self.ref_gaps)
            self.individual_references["scale_diff"] = p.map(lambda note, duration : find_closest_fit(note, duration, SCALES), zip(self.ref_notes, self.ref_durations))
            self.individual_references["transitions"] = p.map(lambda note: transitions(note), self.ref_notes)
            self.individual_references["transitions_test"] = p.map(lambda note: transition_map(note), self.ref_notes)

        # Average of reference songs
        self.average_references["span"] = np.average(self.individual_references["span"])
        self.average_references["rep2"] = np.average(self.individual_references["rep2"])
        self.average_references["rep3"] = np.average(self.individual_references["rep3"])
        self.average_references["rep4"] = np.average(self.individual_references["rep4"])
        self.average_references["unique"] = np.average(self.individual_references["unique"])
        self.average_references["restless"] = np.average(self.individual_references["restless"])
        self.average_references["avg_rest"] = np.average(self.individual_references["avg_rest"])
        self.average_references["song_len"] = np.average(self.individual_references["song_len"])
        self.average_references["scale_diff"] = np.average(self.individual_references["scale_diff"])
        self.average_references["transitions"] = self.average_transitions(self.individual_references["transitions"])
        self.average_references["transitions_test"] = sum(self.individual_references["transitions_test"])/len(self.individual_references["transitions_test"])
        
        return self.average_references

    def compute_transitions(notes):
        transitions = dict()
        for bigram, count in count_ngrams(notes, 2).items():
            transition = bigram[1] - bigram[0]
            if transitions in transitions.keys(): transitions[transition] += count
            else: transitions[transition] = count
        return transitions
    def average_transitions(transitions_maps):
        average_transitions = dict()
        for transitions_map in transitions_maps:
            for transition, count in transitions_map.items():
                if transition in average_transitions.keys(): average_transitions[transition] += count
                else: average_transitions[transition] = count
        return average_transitions

def test():
    # analold = quantitative_analysis("../runs/1-Avril-Rapport", "seq2seq", "../data/new_dataset/test.csv", [19,39,59])
    # analold.generate_midi(4,4)
    # analold.analyse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = pd.read_csv("../data/new_dataset/test.csv")
    model = AutoModelForSeq2SeqLM.from_pretrained("../runs/1-Avril-Rapport").to('cuda')
    tokenizer = AutoTokenizer.from_pretrained("../runs/1-Avril-Rapport")
    
    inputs = ['Generate notes: ' + lyrics for lyrics in dataset.lyrics[0:10]]*2
    references = [json.loads(midi) for midi in dataset.midi_notes[0:10]]*2

    notes = []
    durations = []
    gaps = []

    ref_notes = [[midi[0] for midi in reference] for reference in references]
    ref_durations = [[midi[1] for midi in reference] for reference in references]
    ref_gaps = [[midi[2] for midi in reference] for reference in references]

    for input, reference in zip(inputs, references):
        token_input = tokenizer([input], truncation=True, padding=True, max_length=512, return_tensors='pt').to(device)
        output = model.generate(**token_input, num_beams=8, do_sample=True, min_length=10, max_length=512)
        decoded_output = tokenizer.batch_decode([output], skip_special_tokens=True)
        midi_output = decode_midi_sequence(decoded_output)
        notes += [[midi[0] for midi in midi_output]]
        durations += [[midi[1] for midi in midi_output]]
        gaps += [[midi[2] for midi in midi_output]]
    
    anal = analyser(inputs, notes, durations, gaps, ref_notes, ref_durations, ref_gaps)
    print(anal.references())
    print(anal.analyse())
    print(anal.analyse_multi())
    print(anal.references_multi())
    

        


if __name__ == "__main__":
    test()