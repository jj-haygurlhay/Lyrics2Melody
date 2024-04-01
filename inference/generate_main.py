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
from .generate_midi import GenerateMidi

HYPS_FILE = './config/hyps.yaml'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_main(model_path, model_name, test_dataset_path):
    
    if 't5' in model_name:
        t5_config = T5Config.from_pretrained(
            't5-small'
            )
        print(t5_config)
        model = MusicT5(t5_config, **config['model']) # TODO create config object
        tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name_or_path='t5-small')
    elif 'gpt' in model_name:
        gpt2_config = GPT2Config.from_pretrained('gpt2')
        model = MusicGPT2(gpt2_config, **config['model']) # TODO create config object
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path='gpt2')
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise ValueError("Model not implemented")
    
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
    model_path = 'checkpoints/model_epoch_4.bin'
    data_path = 'data/dataset_matrices/test_data_matrix.npy'
    model_name = 't5'
    generate_main(model_path=model_path, model_name=model_name, test_dataset_path = data_path)