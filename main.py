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
from inference.generate_midi import GenerateMidi

HYPS_FILE = './config/hyps.yaml'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    with open(HYPS_FILE, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Set seed
    set_seed(config['seed'])

    # pl_trainer = PLTrainer()

    # Load model and tokenizer
    model_name = config['model']['model_name']
    del config['model']['model_name']

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

    model.to(device)

    # Create dataset and collator
    batch_size = config['training']['batch_size']

    train_dataset = SongsDataset(config['data']['data_dir'], split='train')
    valid_dataset = SongsDataset(config['data']['data_dir'], split='valid')
    test_dataset  = SongsDataset(config['data']['data_dir'], split='test')
    collator = SongsCollator(tokenizer=tokenizer, output_eos=config['data']['output_eos'], max_length=config['data']['max_sequence_length'], use_syllables=config['data']['use_syllables'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator, pin_memory=True, num_workers=4)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator, pin_memory=True, num_workers=4)

    # pl_trainer.fit(model, tokenizer, train_loader, val_loader, test_loader)

    # Create trainer
    custom_trainer = CustomTrainer(
        model=model, 
        pl_trainer=None,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        collator=collator,
        **config['training']
    )

    # Train model
    custom_trainer.train()

    # Decode generated tokens
    # notes, durations, gaps = [], [], []
    # for token_ids in generated_tokens:
    #     n, d, g = custom_trainer.model.decode_generated_sequence(custom_trainer.tokenizer, token_ids)
    #     notes.extend(n)
    #     durations.extend(d)
    #     gaps.extend(g)

    # # Generate MIDI file using your GenerateMidi class
    # # Assuming your GenerateMidi class has a method called 'create_midi'
    # generate_midi = GenerateMidi(notes, durations, gaps)  # or however you instantiate this class
    # midi_file_path = 'output.mid'  # Define your desired output file path
    # generate_midi.save_midi(midi_file_path)
    # print(f"MIDI file generated: {midi_file_path}")

if __name__ == "__main__":
    main()