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
        
        self.durations_array = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 6.0, 6.5, 8.0, 8.5, 16.0, 16.5, 32.0, 32.5])
        self.gaps_array = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])

    def generate_melody(self):
        """
        Generates a melody sequence based on the specified distributions.
        :return: A list of lists representing the melody with each inner list containing [MIDI note, duration, gap].
        """
        melody = []
        
        for _ in range(self.sequence_length):
            midi_note = np.random.choice(len(self.midi_note_distribution), p=self.midi_note_distribution)
            duration = np.random.choice(self.durations_array, p=self.duration_distribution)
            gap = np.random.choice(self.gaps_array, p=self.gap_distribution)
            melody.append([midi_note, duration, gap])

        melody_array = np.array(melody)
        return melody_array

