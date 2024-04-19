import numpy as np

class RandomBaselineModel:
    def __init__(self, midi_note_distribution, duration_distribution, gap_distribution, sequence_length=20):
        """
        Initializes the RandomBaselineModel with separate distributions for MIDI notes, durations, and gaps.
        :param midi_note_distribution: A list or array with the probabilities for each MIDI note.
        :param duration_distribution: A list or array with the probabilities for each duration.
        :param gap_distribution: A list or array with the probabilities for each gap.
        :param sequence_length: The length of the melody sequence to be generated.
        """
        self.midi_note_distribution = midi_note_distribution
        self.duration_distribution = duration_distribution
        self.gap_distribution = gap_distribution
        self.sequence_length = sequence_length

    def generate_melody(self):
        """
        Generates a melody sequence based on the specified distributions.
        :return: A list of lists representing the melody with each inner list containing [MIDI note, duration, gap].
        """
        melody = []
        for _ in range(self.sequence_length):
            midi_note = np.random.choice(len(self.midi_note_distribution), p=self.midi_note_distribution)
            duration = np.random.choice(len(self.duration_distribution), p=self.duration_distribution)
            gap = np.random.choice(len(self.gap_distribution), p=self.gap_distribution)
            melody.append([midi_note, duration, gap])

        return melody
