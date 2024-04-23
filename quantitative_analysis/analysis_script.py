import sys
from pathlib import Path
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
import torch
import numpy as np
from quantitative_analysis import Analyser
from dataloader.dataset import SongsDataset
import json
from inference import Generator

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
try:
    gen_not = np.load("/home/max/Documents/Univ/Lyrics2Melody/runs/data_aug_0.9_all_tones/cached_generation/generated_pitches.npy")
    gen_dur = np.load("/home/max/Documents/Univ/Lyrics2Melody/runs/data_aug_0.9_all_tones/cached_generation/generated_lengths.npy")
    gen_gap = np.load("/home/max/Documents/Univ/Lyrics2Melody/runs/data_aug_0.9_all_tones/cached_generation/generated_rests.npy")
    assert gen_not.shape == gen_dur.shape == gen_gap.shape
    assert gen_not.shape[0] >= target_anal_count
    gen_not = gen_not[:target_anal_count,:]
    gen_dur = gen_dur[:target_anal_count,:]
    gen_gap = gen_gap[:target_anal_count,:]
except:
    print("Not enough cached melodies generated: generating enough now")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = 'runs/data_aug_0.9_all_tones' # Change this to the path of the model you want to use
    topk = [30, 5, 5] # [notes, durations, gaps]
    temperature = 0.5

    # Load generator
    generator = Generator(model_dir, './vocab/syllables.txt', 'model_best.pt', device=device)
    outputs = list(map(lambda text: generator.predict(text, temperature=temperature, topk=topk), lyrics))
    outputs = np.asarray(outputs)

    gen_not = outputs[:,:,0]
    gen_dur = outputs[:,:,1]
    gen_gap = outputs[:,:,2]

    np.save("/home/max/Documents/Univ/Lyrics2Melody/runs/data_aug_0.9_all_tones/cached_generation/generated_pitches.npy", gen_not)
    np.save("/home/max/Documents/Univ/Lyrics2Melody/runs/data_aug_0.9_all_tones/cached_generation/generated_lengths.npy", gen_dur)
    np.save("/home/max/Documents/Univ/Lyrics2Melody/runs/data_aug_0.9_all_tones/cached_generation/generated_rests.npy", gen_gap)

analyser = Analyser(lyrics, (gen_not).tolist(), (gen_dur).tolist(), (gen_gap).tolist(), (ref_not).tolist(), (ref_dur).tolist(), (ref_gap).tolist())
gen_stats = analyser.analyse_multi()
ref_stats = analyser.references_multi()
with open("quantitative_analysis/model_best/gen_stats", "w") as f:
    json.dump(gen_stats, f, indent=4)
with open("quantitative_analysis/model_best/ref_stats", "w") as f:
    json.dump(ref_stats, f, indent=4)    

# with open("quantitative_analysis/lstm-gan_stats/analyser", "w") as f:
#     json.dump(analyser, f, cls=Analyser.Encoder, indent=4)