from itertools import repeat
import sys, os
from pathlib import Path
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
print(sys.path)
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)
import pandas as pd
# from utils import decode_note, decode_duration, decode_gap
import torch
import numpy as np
from utils.ngram import ngram_repetition, count_ngrams, transitions
from utils.BLEUscore import bleu_score
from utils.mmd import Compute_MMD as mmd
import json
from utils.scale import scale, find_closest_fit, ALL_SCALES
from collections import defaultdict
from multiprocessing import Pool
from statistics import mean
# from inference import decode_midi_sequence

def span(notes): return max(notes) - min(notes)
def rep2(notes): return ngram_repetition(notes, 2)
def rep3(notes): return ngram_repetition(notes, 3)
def rep4(notes): return ngram_repetition(notes, 4)
def unique(notes): return len(count_ngrams(notes, 1).keys())
def restless(gaps): return count_ngrams(gaps, 1)[(0.0,)]
def avg_rest(gaps): return np.average(gaps)
def song_len(durations, gaps): return sum(durations)+sum(gaps)
def scale_diff(notes, durations): return find_closest_fit(notes, durations, ALL_SCALES)[1]
def distribution(vals):
    distribution = defaultdict(int)
    for val in vals:
        distribution[val] +=1
    return distribution
def transition_distribution(vals):
    return distribution(transitions(vals))
def average_distribution(distributions):
    avg_distribution = defaultdict(int)
    for distribution in distributions:
        for val, count in distribution.items():
            avg_distribution[val] += count/len(distributions)
    return avg_distribution
def bleu2_trans(vals, ref_vals): return bleu_score([transitions(vals)], [[transitions(ref_vals)]], max_n=2)
def bleu3_trans(vals, ref_vals): return bleu_score([transitions(vals)], [[transitions(ref_vals)]], max_n=3)
def bleu4_trans(vals, ref_vals): return bleu_score([transitions(vals)], [[transitions(ref_vals)]], max_n=4)
def bleu2(vals, ref_vals): return bleu_score([vals], [[ref_vals]], max_n=2)
def bleu3(vals, ref_vals): return bleu_score([vals], [[ref_vals]], max_n=3)
def bleu4(vals, ref_vals): return bleu_score([vals], [[ref_vals]], max_n=4)
# def ndg_bleu2(notes, ref_notes,durations, ref_durations,gaps, ref_gaps): return (bleu_score([transitions(notes)], [[transitions(ref_notes)]], max_n=2) + bleu_score([durations], [[ref_durations]], max_n=2) + bleu_score([gaps], [[ref_gaps]], max_n=2))/3
# def ndg_bleu3(notes, ref_notes,durations, ref_durations,gaps, ref_gaps): return (bleu_score([transitions(notes)], [[transitions(ref_notes)]], max_n=3) + bleu_score([durations], [[ref_durations]], max_n=3) + bleu_score([gaps], [[ref_gaps]], max_n=3))/3
# def ndg_bleu4(notes, ref_notes,durations, ref_durations,gaps, ref_gaps): return (bleu_score([transitions(notes)], [[transitions(ref_notes)]], max_n=4) + bleu_score([durations], [[ref_durations]], max_n=4) + bleu_score([gaps], [[ref_gaps]], max_n=4))/3


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
        """DEPRECATED"""
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
            self.individual_results["scale_diff"] += [find_closest_fit(note, duration, ALL_SCALES)[1]]
            self.individual_results["transitions"] += [transitions_map(note)]
            # self.individual_results["transitions_test"] += [transition_map(note)]
            self.individual_results["bleu2"] += [bleu_score([note], [[ref_note]], max_n=2)]
            self.individual_results["bleu3"] += [bleu_score([note], [[ref_note]], max_n=3)]
            self.individual_results["bleu4"] += [bleu_score([note], [[ref_note]], max_n=4)]
        
        # Averages of generated
        self.average_results["span"] = np.average(self.individual_results["span"])
        self.average_results["rep2"] = np.average(self.individual_results["rep2"])
        self.average_results["rep3"] = np.average(self.individual_results["rep3"])
        self.average_results["rep4"] = np.average(self.individual_results["rep4"])
        self.average_results["unique"] = np.average(self.individual_results["unique"])
        self.average_results["restless"] = np.average(self.individual_results["restless"])
        self.average_results["avg_rest"] = np.average(self.individual_results["avg_rest"])
        self.average_results["song_len"] = np.average(self.individual_results["song_len"])
        self.average_results["scale_diff"] = np.average(self.individual_results["scale_diff"][1])
        self.average_results["transitions"] = average_transitions_map(self.individual_results["transitions"])
        # self.average_results["transitions_test"] = transition_map.average(self.individual_results["transitions_test"])
        self.average_results["bleu2"] = np.average(self.individual_results["bleu2"])
        self.average_results["bleu3"] = np.average(self.individual_results["bleu3"])
        self.average_results["bleu4"] = np.average(self.individual_results["bleu4"])
        return self.average_results

    def analyse_multi(self):
        # Individual songs generated
        with Pool(len(os.sched_getaffinity(0))) as p:
            self.individual_results["span"] = p.map(span, self.notes)
            self.individual_results["rep2"] = p.map(rep2, self.notes) 
            self.individual_results["rep3"] = p.map(rep3, self.notes) 
            self.individual_results["rep4"] = p.map(rep4, self.notes) 
            self.individual_results["unique"] = p.map(unique, self.notes)
            self.individual_results["restless"] = p.map(restless, self.gaps) 
            self.individual_results["avg_rest"] = p.map(avg_rest, self.gaps)
            self.individual_results["song_len"] = p.starmap(song_len, zip(self.durations, self.gaps))
            self.individual_results["scale_diff"] = p.starmap(scale_diff, zip(self.notes, self.durations))
            self.individual_results["transitions"] = p.map(transition_distribution, self.notes)
            self.individual_results["distribution_notes"] = p.map(distribution, self.notes)
            self.individual_results["bleu2_notes"] = p.starmap(bleu2_trans, zip(self.notes, self.ref_notes))
            self.individual_results["bleu3_notes"] = p.starmap(bleu3_trans, zip(self.notes, self.ref_notes))
            self.individual_results["bleu4_notes"] = p.starmap(bleu4_trans, zip(self.notes, self.ref_notes))
            self.individual_results["bleu2_durations"] = p.starmap(bleu2, zip(self.durations, self.ref_durations))
            self.individual_results["bleu3_durations"] = p.starmap(bleu3, zip(self.durations, self.ref_durations))
            self.individual_results["bleu4_durations"] = p.starmap(bleu4, zip(self.durations, self.ref_durations))
            self.individual_results["bleu2_gaps"] = p.starmap(bleu2, zip(self.gaps, self.ref_gaps))
            self.individual_results["bleu3_gaps"] = p.starmap(bleu3, zip(self.gaps, self.ref_gaps))
            self.individual_results["bleu4_gaps"] = p.starmap(bleu4, zip(self.gaps, self.ref_gaps))
        
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
        self.average_results["transitions"] = average_distribution(self.individual_results["transitions"])
        self.average_results["distribution_notes"] = average_distribution(self.individual_results["distribution_notes"])
        self.average_results["bleu2_notes"] = np.average(self.individual_results["bleu2_notes"])
        self.average_results["bleu3_notes"] = np.average(self.individual_results["bleu3_notes"])
        self.average_results["bleu4_notes"] = np.average(self.individual_results["bleu4_notes"])
        self.average_results["bleu2_durations"] = np.average(self.individual_results["bleu2_durations"])
        self.average_results["bleu3_durations"] = np.average(self.individual_results["bleu3_durations"])
        self.average_results["bleu4_durations"] = np.average(self.individual_results["bleu4_durations"])
        self.average_results["bleu2_gaps"] = np.average(self.individual_results["bleu2_gaps"])
        self.average_results["bleu3_gaps"] = np.average(self.individual_results["bleu3_gaps"])
        self.average_results["bleu4_gaps"] = np.average(self.individual_results["bleu4_gaps"])
        return self.average_results
    
    def references(self, ref_notes = None, ref_durations = None, ref_gaps = None):
        """DEPRECATED"""

        if ref_notes is None or ref_durations is None or ref_gaps is None or len(ref_notes) != len(ref_durations) != len(ref_gaps):
            ref_notes = self.ref_notes
            ref_durations = self.ref_durations
            ref_gaps = self.ref_gaps
        # Individual reference songs
        for ref_note, ref_duration, ref_gap in zip(ref_notes, ref_durations, ref_gaps):
            # print(ref_note)
            self.individual_references["span"] += [max(ref_note) - min(ref_note)]
            self.individual_references["rep2"] += [ngram_repetition(ref_note, 2)]
            self.individual_references["rep3"] += [ngram_repetition(ref_note, 3)]
            self.individual_references["rep4"] += [ngram_repetition(ref_note, 4)]
            self.individual_references["unique"] += [[len(count_ngrams(ref_note, 1).keys())]]
            self.individual_references["restless"] += [count_ngrams(ref_gap,1)[(0.0,)]]
            self.individual_references["avg_rest"] += [np.average(ref_gap)]
            self.individual_references["song_len"] += [sum(ref_duration)+ sum(ref_gap)]
            self.individual_references["scale_diff"] += [find_closest_fit(ref_note, ref_duration, ALL_SCALES)[1]]
            self.individual_references["transitions"] += [transitions_map(ref_note)]
            # self.individual_references["transitions_test"] += [transition_map(ref_note)]

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
        self.average_references["transitions"] = average_transitions_map(self.individual_references["transitions"])
        # self.average_references["transitions_test"] = transition_map.average(self.individual_references["transitions_test"])
        return self.average_references
    
    def references_multi(self):
        # Individual reference songs
        with Pool(len(os.sched_getaffinity(0))) as p:
            self.individual_references["span"] = p.map(span, self.ref_notes)
            self.individual_references["rep2"] = p.map(rep2, self.ref_notes) 
            self.individual_references["rep3"] = p.map(rep3, self.ref_notes) 
            self.individual_references["rep4"] = p.map(rep4, self.ref_notes) 
            self.individual_references["unique"] = p.map(unique, self.ref_notes)
            self.individual_references["restless"] = p.map(restless, self.ref_gaps) 
            self.individual_references["avg_rest"] = p.map(avg_rest, self.ref_gaps)
            self.individual_references["song_len"] = p.starmap(song_len, zip(self.ref_durations, self.ref_gaps))
            self.individual_references["scale_diff"] = p.starmap(scale_diff, zip(self.ref_notes, self.ref_durations))
            self.individual_references["transitions"] = p.map(transition_distribution, self.ref_notes)
            self.individual_references["distribution_notes"] = p.map(distribution, self.notes)

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
        self.average_references["transitions"] = average_distribution(self.individual_references["transitions"])
        self.average_references["distribution_notes"] = average_distribution(self.individual_references["distribution_notes"])
        # self.average_references["transitions_test"] = transition_map.average(self.individual_references["transitions_test"])
        
        return self.average_references
    class Encoder(json.JSONEncoder):
        def default(self, o):
            return o.__dict__
import timeit
def test():
    dataset = pd.read_csv("./data/new_dataset/test.csv")

    inputs = ['Generate notes: ' + lyrics for lyrics in dataset.lyrics[0:50]]*5
    references = [json.loads(midi) for midi in dataset.midi_notes[0:50]]*5

    notes = [[midi[0] for midi in reference][0:20] for reference in references]
    durations = [[midi[1] for midi in reference][0:20] for reference in references]
    gaps = [[midi[2] for midi in reference][0:20] for reference in references]

    ref_notes = [[midi[0] for midi in reference][0:20] for reference in references]
    ref_durations = [[midi[1] for midi in reference][0:20] for reference in references]
    ref_gaps = [[midi[2] for midi in reference][0:20] for reference in references]
    
    anal = analyser(inputs, notes, durations, gaps, ref_notes, ref_durations, ref_gaps)
    t2 = timeit.default_timer()
    print(anal.analyse_multi())
    print(anal.references_multi())
    t3 = timeit.default_timer()      


if __name__ == "__main__":
    test()