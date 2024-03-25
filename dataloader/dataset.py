from torch.utils.data import Dataset
import os
import pandas as pd

class SongsDataset(Dataset):

    def __init__(self, path, split='train'):
        if not os.path.isdir(path):
            raise ValueError('Invalid `path` variable! Needs to be a directory')

        self.path = path
        self.lyrics = []
        self.syllables = []
        self.midi_notes = []
        self.file_names = []
        self.split = split
        self.load_songs()

    def load_songs(self):
        dataset = pd.read_csv(os.path.join(self.path, f'{self.split}.csv'))
        self.lyrics = dataset['lyrics'].tolist()
        self.syllables = dataset['syl_lyrics'].tolist()
        self.midi_notes = dataset['midi_notes'].tolist()
        self.file_names = dataset['filename'].tolist()

    def __len__(self):
        return len(self.lyrics)

    def __getitem__(self, item):
        return {
            'lyrics': self.lyrics[item],
            'syl_lyrics': self.syllables[item],
            'midi_notes': self.midi_notes[item],
            'filename': self.file_names[item]
        }