import pandas as pd
import re  # For converting string representations of lists back to actual lists

class SlidingWindow:
    def __init__(self, window_size, step_size=1):
        self.window_size = window_size
        self.step_size = step_size

    def slide_paired_sequences(self, syl_lyrics, midi_notes):
        paired_windows = []
        for i in range(0, len(syl_lyrics) - self.window_size + 1, self.step_size):
            syl_window = syl_lyrics[i:i + self.window_size]
            midi_window = midi_notes[i:i + self.window_size]
            paired_windows.append((syl_window, midi_window))
        return paired_windows

    def process_csv(self, input_csv, output_csv):
        df = pd.read_csv(input_csv)
        augmented_rows = []

        for index, row in df.iterrows():
            # Assume syl_lyrics are separated by space and midi_notes by comma
            midi_notes = self.parse_midi_notes(row['midi_notes'])
            syl_lyrics = row['syl_lyrics'].split(' ')
            lyrics = row['lyrics'].split(' ')

            # Check if we have enough data to create at least one window
            if len(midi_notes) < self.window_size or len(syl_lyrics) < self.window_size:
                print(f"Row {index} does not have enough data for windowing.")
                continue

            # Create paired windows for syllables and MIDI notes
            paired_windows = self.slide_paired_sequences(syl_lyrics, midi_notes)
            print(f"Row {index} generated {len(paired_windows)} paired windows.")

            for syl_window, midi_window in paired_windows:
                words = self.retrieve_words(syl_window, lyrics, row['syl_lyrics'])
                
                augmented_rows.append({
                    'filename': row['filename'],
                    'lyrics': words,
                    'syl_lyrics': ' '.join(syl_window),
                    'midi_notes': str(midi_window)
                })

        print(f"Total augmented rows: {len(augmented_rows)}")

        if augmented_rows:
            augmented_df = pd.DataFrame(augmented_rows)
            augmented_df.to_csv(output_csv, index=False)
        else:
            print("No data was processed. CSV will not be written.")
    
    def parse_midi_notes(self, midi_notes_str):
        # Parse the midi_notes using regular expressions or another safe method
        # Example assuming midi_notes are in the format [[note1, duration1, gap1], ...]
        # You will need to adjust the regular expression based on your actual data format
        pattern = re.compile(r'\[\s*(\d+\.\d+|\d+),\s*(\d+\.\d+|\d+),\s*(\d+\.\d+|\d+)\s*\]')
        return [list(map(float, match.groups())) for match in pattern.finditer(midi_notes_str)]

    def retrieve_words(self, syl_window, full_lyrics_words, syl_lyrics):
        words_to_keep = []
        syl_lyrics_list = syl_lyrics.split()

        # Mapping of syllables to their index in the full lyrics
        syl_to_index = {}
        for index, word in enumerate(full_lyrics_words):
            for syl in word.split(' '):
                syl_to_index[syl] = index

        # Go through each syllable in the syl_window
        for syl in syl_window:
            # Find which word the syllable could be a part of
            word_indices = [index for part, index in syl_to_index.items() if syl in part]
            
            # Find the correct word that the syllable is part of and add to words_to_keep
            for index in word_indices:
                if ' '.join(full_lyrics_words[index].split()).startswith(syl):
                    words_to_keep.append(full_lyrics_words[index])
                    break  # Once the correct word is found, stop checking

        # Deduplicate while maintaining order
        seen = set()
        deduplicated_words = [x for x in words_to_keep if not (x in seen or seen.add(x))]

        return ' '.join(deduplicated_words)