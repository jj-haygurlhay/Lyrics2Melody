import yaml
import torch
from models import MusicT5, MusicGPT2
from training import Trainer
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

def main():
    with open(HYPS_FILE, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Set seed
    set_seed(config['seed'])

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
    

    # Create dataset and collator
    train_dataset = SongsDataset(config['data']['data_dir'], split='train')
    valid_dataset = SongsDataset(config['data']['data_dir'], split='valid')
    test_dataset  = SongsDataset(config['data']['data_dir'], split='test')
    collator = SongsCollator(tokenizer=tokenizer, output_eos=config['data']['output_eos'], max_length=config['data']['max_sequence_length'], use_syllables=config['data']['use_syllables'])

    # Create trainer
    trainer = Trainer(
        model=model, 
        train_dataset=train_dataset,  
        val_dataset=valid_dataset,
        test_dataset=test_dataset,
        device=device,
        collator=collator,
        **config['training']
    )

    # Train model
    trainer.train()

if __name__ == "__main__":
    main()