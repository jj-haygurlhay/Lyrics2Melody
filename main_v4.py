import json
from tqdm import tqdm
import yaml
import torch
from torch.utils.data import DataLoader
from models import MusicT5, MusicGPT2
from datasets import load_dataset, load_metric
from models.custom_transformer import CustomModelTransformer
from training import Trainer as CustomTrainer
import evaluate
# from pytorch_lightning import Trainer as PLTrainer
from transformers import (
    GPT2Tokenizer, 
    GPT2Config, 
    T5Tokenizer, 
    T5Config,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer, 
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    AutoTokenizer,
    GenerationConfig,
    T5EncoderModel,
    set_seed
)
from dataloader import SongsDataset, SongsCollator_v3
from utils.quantize import encode_note, encode_duration, encode_gap, MIDI_NOTES, DURATIONS, GAPS
import numpy as np
import nltk
from datetime import datetime
from models import CustomModelRNN
from torch.optim import AdamW
import torch.nn as nn

HYPS_FILE = './config/hyps.yaml'
EOS_token = -1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def train_epoch(dataloader, model, encoder_optimizer,
          decoder_optimizer, criterion, epoch, note_loss_weight, duration_loss_weight, gap_loss_weight):

    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    print_data = True

    for data in progress_bar:
        input_tensor = data['input_ids'].to(device)
        attn_mask = data['attention_mask'].to(device)
        target_notes = data['labels']['notes'].to(device)
        target_durations = data['labels']['durations'].to(device)
        target_gaps = data['labels']['gaps'].to(device)

        target_tensor = torch.cat([target_notes.unsqueeze(-1), target_durations.unsqueeze(-1), target_gaps.unsqueeze(-1)], dim=-1)

        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()


        decoder_outputs_notes, decoder_outputs_durations, decoder_outputs_gaps = model(input_tensor, attn_mask, target_tensor[:, :-1, :])

        loss_notes = criterion(decoder_outputs_notes.transpose(1, 2), target_notes[:, 1:])
        loss_durations = criterion(decoder_outputs_durations.transpose(1, 2), target_durations[:, 1:])
        loss_gaps = criterion(decoder_outputs_gaps.transpose(1, 2), target_gaps[:, 1:])
        loss = note_loss_weight * loss_notes + duration_loss_weight * loss_durations + gap_loss_weight * loss_gaps

        if print_data:
            print_data = False
            # Compute notes
            notes = torch.argmax(decoder_outputs_notes[0], dim=-1).cpu().numpy()
            durations = torch.argmax(decoder_outputs_durations[0], dim=-1).cpu().numpy()
            gaps = torch.argmax(decoder_outputs_gaps[0], dim=-1).cpu().numpy()
            print('Notes:', notes[:20])
            print('Target Notes', target_notes[0, 1:21].cpu().numpy())
            print('\nDurations:', durations[:20])
            print('Target Durations', target_durations[0, 1:21].cpu().numpy())
            print('\nGaps:', gaps[:20])
            print('Target Gaps', target_gaps[0, 1:21].cpu().numpy())
        loss.backward()

        if encoder_optimizer is not None:
            encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'Training Loss': total_loss / len(dataloader)})

    return total_loss / len(dataloader)

def train(train_dataloader, val_dataloader, model, n_epochs, learning_rate=0.001, weight_decay=0.01,
               plot_every=1, train_encoder=False, note_loss_weight=0.8, duration_loss_weight=0.2, gap_loss_weight=0.2):
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    if train_encoder:
        encoder_optimizer = AdamW(model.encoder.parameters(), lr=float(learning_rate), weight_decay=weight_decay)
    else:
        encoder_optimizer = None
    decoder_optimizer = AdamW(model.decoder.parameters(), lr=float(learning_rate), weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, n_epochs + 1):
        model.train()
        loss = train_epoch(train_dataloader, model, encoder_optimizer, decoder_optimizer, criterion, epoch, note_loss_weight, duration_loss_weight, gap_loss_weight)
        plot_loss_total += loss

        print(f"Epoch {epoch}, training Loss: {loss}")

        plot_loss_avg = plot_loss_total / plot_every
        plot_losses.append(plot_loss_avg)
        plot_loss_total = 0

        model.eval()
        val_loss = evaluate_model(model, val_dataloader, criterion, note_loss_weight, duration_loss_weight, gap_loss_weight)
        print(f"Epoch {epoch}, Validation Loss: {val_loss}")

def evaluate_model(model, dataloader, criterion, note_loss_weight, duration_loss_weight, gap_loss_weight):
    with torch.no_grad():
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Validation: ")
        print_data = True
        for data in progress_bar:
            input_tensor = data['input_ids'].to(device)
            attn_mask = data['attention_mask'].to(device)
            target_notes = data['labels']['notes'].to(device)
            target_durations = data['labels']['durations'].to(device)
            target_gaps = data['labels']['gaps'].to(device)
            decoder_outputs_notes, decoder_outputs_durations, decoder_outputs_gaps, logits_notes, logits_durations, logits_gaps = model.generate(input_tensor, attn_mask)

            loss_notes     = criterion(logits_notes.transpose(1, 2), target_notes[:, 1:])
            loss_durations = criterion(logits_durations.transpose(1, 2), target_durations[:, 1:])
            loss_gaps      = criterion(logits_gaps.transpose(1, 2), target_gaps[:, 1:])
            loss = note_loss_weight * loss_notes + duration_loss_weight * loss_durations + gap_loss_weight * loss_gaps

            if print_data:
                print_data = False
                print('Notes:', decoder_outputs_notes[0][:20].cpu().numpy())
                print('Target Notes:', target_notes[0][1:20].cpu().numpy())
                print('\nDurations:', decoder_outputs_durations[0][:20].cpu().numpy())
                print('Target Durations:', target_durations[0][1:21].cpu().numpy())
                print('\nGaps:', decoder_outputs_gaps[0][:20].cpu().numpy())
                print('Target Gaps:', target_gaps[0][1:21].cpu().numpy())

            total_loss += loss.item()
            progress_bar.set_postfix({'Validation Loss': total_loss / len(dataloader)})

        return total_loss / len(dataloader)


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def main():
    with open(HYPS_FILE, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Set seed
    set_seed(config['seed'])

    train_encoder = config['training']['train_encoder']

    # Load model and tokenizer
    encoder_name = 't5-small'
    encoder = T5EncoderModel.from_pretrained(encoder_name)
    tokenizer = T5Tokenizer.from_pretrained(encoder_name, legacy=False)

    # Freeze encoder
    if not train_encoder:
        for param in encoder.parameters():
            param.requires_grad = False
        encoder.eval()

    model = CustomModelTransformer(
        encoder=encoder, 
        device=device, 
        SOS_token=0, 
        MAX_LENGTH=config['data']['max_sequence_length'], 
        train_encoder=train_encoder, 
        dropout_p=config['model']['dropout'],
        expansion_factor=config['model']['expansion_factor'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers']
        )
    
    for p in model.decoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # Create dataset and collator
    batch_size = config['training']['batch_size']

    train_dataset = SongsDataset(config['data']['data_dir'], split='train')
    valid_dataset = SongsDataset(config['data']['data_dir'], split='valid')
    test_dataset  = SongsDataset(config['data']['data_dir'], split='test')
    collator = SongsCollator_v3(tokenizer=tokenizer, output_eos=config['data']['output_eos'], max_length=config['data']['max_sequence_length'], use_syllables=config['data']['use_syllables'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator, pin_memory=True, num_workers=0)
    val_loader   = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator, pin_memory=True, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator, pin_memory=True,  num_workers=0)

    train(
        train_loader, 
        val_loader, 
        model, 
        n_epochs=config['training']['epochs'], 
        learning_rate=config['training']['lr'], 
        weight_decay=config['training']['weight_decay'], 
        train_encoder=train_encoder,
        note_loss_weight=config['training']['note_loss_weight'],
        duration_loss_weight=config['training']['duration_loss_weight'],
        gap_loss_weight=config['training']['gap_loss_weight']
        )
    
    # Save model
    torch.save(model.state_dict(), 'model.pth')

if __name__ == "__main__":
    main()