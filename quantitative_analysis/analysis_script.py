import sys
import os
from pathlib import Path
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
import torch
import numpy as np
from quantitative_analysis import Analyser
from dataloader.dataset import SongsDataset
import json
from inference import Generator
from project_utils import scale

dirtytestdata = np.load("/mnt/1TB-HDD-2/Documents/Univ/Lyrics-Conditioned-Neural-Melody-Generation/data/processed_dataset_matrices/test_data_matrix.npy")
dirtytestdata = dirtytestdata[:,0:60]
dirtytestdata = dirtytestdata.reshape((1051,20,3))

ref_not = dirtytestdata[:,:,0]
ref_dur = dirtytestdata[:,:,1]
ref_gap = dirtytestdata[:,:,2]

gen_not = np.load("/mnt/1TB-HDD-2/Documents/Univ/Lyrics-Conditioned-Neural-Melody-Generation/saved_gan_models/saved_model_best_overall_mmd/generated_pitches.npy")
gen_dur = np.load("/mnt/1TB-HDD-2/Documents/Univ/Lyrics-Conditioned-Neural-Melody-Generation/saved_gan_models/saved_model_best_overall_mmd/generated_lengths.npy")
gen_gap = np.load("/mnt/1TB-HDD-2/Documents/Univ/Lyrics-Conditioned-Neural-Melody-Generation/saved_gan_models/saved_model_best_overall_mmd/generated_rests.npy")

lstm_analyser = Analyser([["N/A"]]*1051, (gen_not).tolist(), (gen_dur).tolist(), (gen_gap).tolist(), (ref_not).tolist(), (ref_dur).tolist(), (ref_gap).tolist())

gen_stats = lstm_analyser.analyse_multi()
ref_stats = lstm_analyser.references_multi()

with open("quantitative_analysis/lstm-gan_stats/gen_stats", "w") as f:
    json.dump(gen_stats, f, indent=4)
with open("quantitative_analysis/lstm-gan_stats/ref_stats", "w") as f:
    json.dump(ref_stats, f, indent=4)


target_anal_count = 100
test_dataset = SongsDataset("data/new_dataset", split="test")
lyrics = test_dataset.syllables[0:target_anal_count]
dirty_midi = [json.loads(midi)[0:20] for midi in test_dataset.midi_notes]
dirty_midi = np.asarray(dirty_midi)

ref_not = dirty_midi[:target_anal_count,:,0]
ref_dur = dirty_midi[:target_anal_count,:,1]
ref_gap = dirty_midi[:target_anal_count,:,2]

def full_analysis(model_dir, model_type, ref_not, ref_dur, ref_gap, lyrics):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    topk = [30, 5, 5] # [notes, durations, gaps]
    temperature = 0.6 
    generator = Generator(model_dir, './vocab/syllables.txt', 'model_best.pt',model_type, device=device)
    outputs = list(map(lambda text: generator.predict(text, temperature=temperature, topk=topk), lyrics))

    #Some of our models cant garantee that 20 notes are generated, to fix this we clamp it to the minimum length or 20
    min_len_seq = min(map(len, outputs))
    target_len = min_len_seq if min_len_seq < 20 else 20
    if(target_len < 20): print(f"FOR {model_dir} SEQENCES OF {target_len} MIDI ARE BEING USED, EVERYTHING IN OUR ANALYSIS ASSUMES 20!!!!")
    for i in range(len(outputs)):
        outputs[i] = outputs[i][:target_len]

    clamped_outputs = list(map(scale.fit_to_closest, outputs))
    outputs = np.array(outputs)
    gen_not = outputs[:,:,0]
    gen_clamped_not = np.array(clamped_outputs)[:,:,0]
    gen_dur = outputs[:,:,1]
    gen_gap = outputs[:,:,2]
    os.makedirs(os.path.dirname(model_dir + "/cached_generation/"), exist_ok=True)
    np.save(model_dir + "/cached_generation/generated_pitches.npy", gen_not)
    np.save(model_dir + "/cached_generation/generated_lengths.npy", gen_dur)
    np.save(model_dir + "/cached_generation/generated_rests.npy", gen_gap)

    analyser = Analyser(lyrics, (gen_not).tolist(), (gen_dur).tolist(), (gen_gap).tolist(), (ref_not).tolist(), (ref_dur).tolist(), (ref_gap).tolist())

    gen_stats = analyser.analyse_multi()
    ref_stats = analyser.references_multi()

    os.makedirs(os.path.dirname(model_dir + "/stats/"), exist_ok=True)
    with open(model_dir + "/stats/gen_stats", "w") as f:
        json.dump(gen_stats, f, indent=4)
    with open(model_dir+ "/stats/ref_stats", "w") as f:
        json.dump(ref_stats, f, indent=4)    

    clamped_analyser = Analyser(lyrics, (gen_clamped_not).tolist(), (gen_dur).tolist(), (gen_gap).tolist(), (ref_not).tolist(), (ref_dur).tolist(), (ref_gap).tolist())

    gen_clamped_stats = clamped_analyser.analyse_multi()

    with open(model_dir + "/stats/gen_clamped_stats", "w") as f:
        json.dump(gen_clamped_stats, f, indent=4)

model_dir = "/home/max/Documents/Univ/Lyrics2Melody/runs/SlidingWindow_only_2024-04-25_15-43-53"
full_analysis(model_dir,"rnn", ref_not, ref_dur, ref_gap, lyrics)
model_dir ="/home/max/Documents/Univ/Lyrics2Melody/runs/SlidingWindow_Shift0.2_2024-04-25_19-17-02"
full_analysis(model_dir,"rnn", ref_not, ref_dur, ref_gap, lyrics)
model_dir = "/home/max/Documents/Univ/Lyrics2Melody/runs/SlidingWindow_Shift0.8_2024-04-25_17-22-42"
full_analysis(model_dir,"rnn", ref_not, ref_dur, ref_gap, lyrics)
model_dir = "/home/max/Documents/Univ/Lyrics2Melody/runs/Without_Aug_2024-04-25_14-39-15"
full_analysis(model_dir,"rnn", ref_not, ref_dur, ref_gap, lyrics)
model_dir = "/home/max/Documents/Univ/Lyrics2Melody/runs/data_aug_0.9_all_tones"
full_analysis(model_dir,"rnn", ref_not, ref_dur, ref_gap, lyrics)
model_dir = "/home/max/Documents/Univ/Lyrics2Melody/runs/t5_shift0_2024-04-26_05-04-46"
full_analysis(model_dir,"transformer", ref_not, ref_dur, ref_gap, lyrics)