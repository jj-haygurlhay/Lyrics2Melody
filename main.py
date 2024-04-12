import yaml
import torch
from torch.utils.data import DataLoader
from models import CustomSeq2SeqModel, LyricsEncoder, MultiHeadMusicDecoder
from training import Trainer as CustomTrainer
# from pytorch_lightning import Trainer as PLTrainer
from transformers import (
    T5Tokenizer,
    T5Config, 
    set_seed
)
from dataloader import SongsDataset, SongsCollator
from utils.quantize import MIDI_NOTES, DURATIONS, GAPS

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
        tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name_or_path='t5-small')
        
        t5_config = T5Config.from_pretrained('t5-small')
        
        t5_config.note_vocab_size = len(MIDI_NOTES)
        t5_config.duration_vocab_size = len(DURATIONS)
        t5_config.gap_vocab_size = len(GAPS)
        
        encoder = LyricsEncoder(pretrained_model_name="t5-small")
        decoder = MultiHeadMusicDecoder(config=t5_config, feedback_mode=config['model']['feedback_mode'])
    
        model = CustomSeq2SeqModel(encoder, decoder, **config['model'])

    elif 'gpt' in model_name:
        pass
    else:
        raise ValueError("Model not implemented")

    model.to(device)

    # Create dataset and collator
    batch_size = config['training']['batch_size']

    train_dataset = SongsDataset(config['data']['data_dir'], split='train')
    valid_dataset = SongsDataset(config['data']['data_dir'], split='valid')
    test_dataset  = SongsDataset(config['data']['data_dir'], split='test')
    collator = SongsCollator(tokenizer=tokenizer, output_eos=config['data']['output_eos'], max_length=config['data']['max_sequence_length'], use_syllables=config['data']['use_syllables'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator, pin_memory=True, num_workers=4,persistent_workers=True)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator, pin_memory=True, num_workers=4,persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator, pin_memory=True, num_workers=4, persistent_workers=True)

    # pl_trainer.fit(model, tokenizer, train_loader, val_loader, test_loader)

    # Create trainer
    custom_trainer = CustomTrainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        **config['training']
    )

    # Train model
    custom_trainer.train()
    custom_trainer.test()

if __name__ == "__main__":
    main()