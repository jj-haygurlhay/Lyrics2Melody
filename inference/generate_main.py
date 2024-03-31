from inference.generate_midi import GenerateMidi
import yaml
import torch
from torch.utils.data import DataLoader
from models import MusicT5, MusicGPT2
from training import Trainer as CustomTrainer
# from pytorch_lightning import Trainer as PLTrainer
from transformers import (
    GPT2Tokenizer, 
    GPT2Config, 
    T5Tokenizer, 
    T5Config, 
    set_seed
)
from dataloader import SongsDataset, SongsCollator

HYPS_FILE = './config/hyps.yaml'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_main():
    model_path = 'path/to/your/trained_model.bin'
    test_dataset_path = 'path/to/your/test_dataset'

# Load the trained model
    model = MusicT5()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode

# Load your test dataset
# Replace 'YourDataset' with your actual dataset class and provide the necessary arguments
    test_dataset = SongsDataset(test_dataset_path, split='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Decode generated tokens
    notes, durations, gaps = [], [], []
    for token_ids in test_loader:
        n, d, g = model.decode_generated_sequence(model.tokenizer, token_ids)
        notes.extend(n)
        durations.extend(d)
        gaps.extend(g)

    # Generate MIDI file using your GenerateMidi class
    # Assuming your GenerateMidi class has a method called 'create_midi'
    generate_midi = GenerateMidi(notes, durations, gaps)  # or however you instantiate this class
    midi_file_path = 'output.mid'  # Define your desired output file path
    generate_midi.save_midi(midi_file_path)
    print(f"MIDI file generated: {midi_file_path}")


if __name__ == "__main__":
    generate_main()