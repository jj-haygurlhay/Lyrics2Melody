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

            # Check if we have enough data to create at least one window
            if len(midi_notes) < self.window_size or len(syl_lyrics) < self.window_size:
                print(f"Row {index} does not have enough data for windowing.")
                continue

            # Create paired windows for syllables and MIDI notes
            paired_windows = self.slide_paired_sequences(syl_lyrics, midi_notes)
            print(f"Row {index} generated {len(paired_windows)} paired windows.")

            for syl_window, midi_window in paired_windows:
                words = self.retrieve_words(syl_window, row['lyrics'], row['syl_lyrics'])
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

    def retrieve_words(self, syl_window, full_lyrics, syl_lyrics):
        words_to_keep = []
        full_lyrics_words = full_lyrics.split()
        syl_lyrics_list = syl_lyrics.split(',')  

    # Create a list of syllables for each word
        word_syllables = []
        for word in full_lyrics_words:
            word_syls = [syl for syl in syl_lyrics_list if syl.startswith(word)]
            word_syllables.append(word_syls)
    
    # Determine which words to keep based on the presence of their syllables in syl_window
        for word_syls in word_syllables:
            if any(syl in syl_window for syl in word_syls):
            # The word is kept if any of its syllables are in syl_window
                words_to_keep.append(word_syls[0].split('_')[0])  # Splitting to get the word without syllable index
    
        return ' '.join(words_to_keep)