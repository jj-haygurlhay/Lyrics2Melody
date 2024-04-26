import os
import yaml
import torch
from torch.utils.data import DataLoader
from dataloader import SongsCollator, SongsCollatorTransformer
from dataloader.vocab import Lang
from transformers import (
    set_seed,
)
from dataloader import SongsDataset
from models import CustomModelRNN, CustomModelTransformer
from datetime import datetime
from training import Trainer, TrainerTransformer
from transformers import T5Tokenizer, T5EncoderModel

HYPS_FILE = './config/hyps.yaml'
EOS_token = 1
SOS_token = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    with open(HYPS_FILE, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    set_seed(config['seed'])
    
    # Create output file
    out_dir = config['out_dir']
    run_dir = config['add_to_runname'] + '_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = os.path.join(out_dir, run_dir)
    os.makedirs(output_folder, exist_ok=True)
    config['training']['out_dir'] = output_folder

    # Save config file
    with open(os.path.join(output_folder, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    if config['model']['type'].lower() == 'rnn':
        main_rnn(config)
    elif config['model']['type'].lower() == 'transformer':
        main_transformer(config)
    else:
        raise ValueError('Invalid model type! Choose between "rnn" and "transformer"')

def main_rnn(config):   
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
    collator = SongsCollator(syllables_lang=syllables, output_eos=EOS_token, max_length=config['data']['max_sequence_length'], octave_shift_percentage=config['data']['octave_shift_percentage'])
    
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

def main_transformer(config):
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    encoder = T5EncoderModel.from_pretrained('t5-small')

    # Initialize model
    model = CustomModelTransformer(
        encoder=encoder,
        PAD_token=EOS_token,
        SOS_token=SOS_token,
        device=device,
        MAX_LENGTH=config['data']['max_sequence_length'],
        train_encoder=config['training']['train_encoder'],
        dropout_p=config['model']['dropout'],
        expansion_factor=config['model']['expansion_factor'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers']
        )

    # Create dataset and collator
    train_dataset = SongsDataset(config['data']['data_dir'], split='train')
    valid_dataset = SongsDataset(config['data']['data_dir'], split='valid')
    test_dataset  = SongsDataset(config['data']['data_dir'], split='test')
    collator = SongsCollatorTransformer(
        tokenizer=tokenizer, 
        PAD_token=EOS_token,
        SOS_token=SOS_token,
        max_length=config['data']['max_sequence_length'], 
        use_syllables=config['data']['use_syllables'],
        octave_shift_percentage=config['data']['octave_shift_percentage']
        )
    
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator, pin_memory=True, num_workers=0)
    val_loader   = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator, pin_memory=True, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator, pin_memory=True,  num_workers=0)

    # Create trainer
    trainer = TrainerTransformer(
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