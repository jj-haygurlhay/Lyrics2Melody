import os
import yaml
import torch
from torch.utils.data import DataLoader
from dataloader.collator import SongsCollator
from dataloader.vocab import Lang
from transformers import (
    set_seed,
)
from dataloader import SongsDataset
from models import CustomModelRNN
from datetime import datetime
from training import Trainer

HYPS_FILE = './config/hyps.yaml'
EOS_token = 1
SOS_token = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    with open(HYPS_FILE, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Create output file
    out_dir = config['out_dir']
    run_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = os.path.join(out_dir, run_dir)
    os.makedirs(output_folder, exist_ok=True)
    config['training']['out_dir'] = output_folder

    # Save config file
    with open(os.path.join(output_folder, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Set seed
    set_seed(config['seed'])
    
    # Create language objects
    syllables = Lang('syllables')
    lines = open('./vocab/syllables.txt', 'r', encoding='utf-8').read().strip().split('\n')
    for syllable in lines:
        syllables.addWord(syllable)

    # Initialize model
    model = CustomModelRNN(
        input_size=syllables.n_words,
        decoder_hidden_size=config['model']['decoder_hidden_size'],
        encoder_hidden_size=config['model']['encoder_hidden_size'],
        embedding_dim=config['model']['embedding_dim'], 
        SOS_token=SOS_token, 
        MAX_LENGTH=config['data']['max_sequence_length'], 
        dropout_p=config['model']['dropout'],
        num_layers=config['model']['num_layers'],
        device=device, 
        )

    # Create dataset and collator
    train_dataset = SongsDataset(config['data']['data_dir'], split='train')
    valid_dataset = SongsDataset(config['data']['data_dir'], split='valid')
    test_dataset  = SongsDataset(config['data']['data_dir'], split='test')
    collator = SongsCollator(syllables_lang=syllables, output_eos=EOS_token, max_length=config['data']['max_sequence_length'])
    
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator, pin_memory=True, num_workers=0)
    val_loader   = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator, pin_memory=True, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator, pin_memory=True,  num_workers=0)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        **config['training']
    )

    # Train model
    trainer.train()

if __name__ == "__main__":
    main()