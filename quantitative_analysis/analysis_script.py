import sys
from pathlib import Path
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
import numpy as np
from quantitative_analysis import analyser as Analyser
import json

dirtytestdata = np.load("/mnt/1TB-HDD-2/Documents/Univ/Lyrics-Conditioned-Neural-Melody-Generation/data/processed_dataset_matrices/test_data_matrix.npy")
dirtytestdata = dirtytestdata[:,0:60]
dirtytestdata = dirtytestdata.reshape((1051,20,3))

ref_not = dirtytestdata[:,:,0]
ref_dur = dirtytestdata[:,:,1]
ref_gap = dirtytestdata[:,:,2]

gen_not = np.load("/mnt/1TB-HDD-2/Documents/Univ/Lyrics-Conditioned-Neural-Melody-Generation/saved_gan_models/saved_model_best_overall_mmd/generated_pitches.npy")
gen_dur = np.load("/mnt/1TB-HDD-2/Documents/Univ/Lyrics-Conditioned-Neural-Melody-Generation/saved_gan_models/saved_model_best_overall_mmd/generated_lengths.npy")
gen_gap = np.load("/mnt/1TB-HDD-2/Documents/Univ/Lyrics-Conditioned-Neural-Melody-Generation/saved_gan_models/saved_model_best_overall_mmd/generated_rests.npy")

analyser = Analyser([["N/A"]]*1051, (gen_not).tolist(), (gen_dur).tolist(), (gen_gap).tolist(), (ref_not).tolist(), (ref_dur).tolist(), (ref_gap).tolist())

gen_stats = analyser.analyse_multi()
ref_stats = analyser.references_multi()

with open("quantitative_analysis/lstm-gan_stats/gen_stats", "w") as f:
    json.dump(gen_stats, f, indent=4)
with open("quantitative_analysis/lstm-gan_stats/ref_stats", "w") as f:
    json.dump(ref_stats, f, indent=4)
# with open("quantitative_analysis/lstm-gan_stats/analyser", "w") as f:
#     json.dump(analyser, f, cls=Analyser.Encoder, indent=4)