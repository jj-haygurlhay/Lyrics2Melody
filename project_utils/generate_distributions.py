import csv
from collections import Counter
import numpy as np

def calculate_probabilities(items_list):
    counts = Counter(items_list)
    total_count = sum(counts.values())
    probabilities = [counts.get(i, 0) / total_count for i in range(128)]
    return probabilities

def create_distributions(csv_filename):
    # Initialize lists to hold MIDI notes, durations, and gaps
    midi_notes_list = []
    durations_list = []
    gaps_list = []

    with open(csv_filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Assuming 'midi_note' is the key and contains a list of lists
            note_info_list = eval(row['midi_note'])
            for note_info in note_info_list:
                # Extract and append individual elements to their respective lists
                midi_notes_list.append(note_info[0])
                durations_list.append(note_info[1])
                gaps_list.append(note_info[2])

    # Calculate the distributions
    midi_note_distribution = calculate_probabilities(midi_notes_list)
    duration_distribution = calculate_probabilities(durations_list)
    gap_distribution = calculate_probabilities(gaps_list)

    return midi_note_distribution, duration_distribution, gap_distribution