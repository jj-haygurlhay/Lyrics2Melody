import os
from tqdm import tqdm
import yaml
import torch
from torch.utils.data import DataLoader
from dataloader.collator import SongsCollator
from dataloader.vocab import Lang
from transformers import (
    set_seed
)
from dataloader import SongsDataset
from models import CustomModelRNN
from torch.optim import AdamW
import torch.nn as nn
from datetime import datetime
import matplotlib.pyplot as plt
import csv


HYPS_FILE = './config/hyps.yaml'
EOS_token = 1
SOS_token = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(dataloader, model, encoder_optimizer,
          decoder_optimizer, criterion, epoch, note_loss_weight, duration_loss_weight, gap_loss_weight):

    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for data in progress_bar:
        input_tensor = data['input_ids'].to(device)
        target_notes = data['labels']['notes'].to(device)
        target_durations = data['labels']['durations'].to(device)
        target_gaps = data['labels']['gaps'].to(device)

        target_tensor = torch.cat([target_notes.unsqueeze(-1), target_durations.unsqueeze(-1), target_gaps.unsqueeze(-1)], dim=-1)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        decoder_outputs_notes, decoder_outputs_durations, decoder_outputs_gaps, _, _ = model(input_tensor, target_tensor)

        loss_notes = criterion(
            decoder_outputs_notes.view(-1, decoder_outputs_notes.size(-1)),
            target_notes.view(-1)
        )
        loss_durations = criterion( 
            decoder_outputs_durations.view(-1, decoder_outputs_durations.size(-1)),
            target_durations.view(-1)
        )
        loss_gaps = criterion(
            decoder_outputs_gaps.view(-1, decoder_outputs_gaps.size(-1)),
            target_gaps.view(-1)
        )
        loss = note_loss_weight * loss_notes + duration_loss_weight * loss_durations + gap_loss_weight * loss_gaps

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'Training Loss': total_loss / len(dataloader)})

    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, criterion, note_loss_weight, duration_loss_weight, gap_loss_weight):
    with torch.no_grad():
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Validation: ")
        is_printed = False # Put to False to print the first prediction
        for data in progress_bar:
            input_tensor = data['input_ids'].to(device)
            target_notes = data['labels']['notes'].to(device)
            target_durations = data['labels']['durations'].to(device)
            target_gaps = data['labels']['gaps'].to(device)

            decoder_outputs_notes, decoder_outputs_durations, decoder_outputs_gaps, _, _ = model(input_tensor)

            loss_notes = criterion(
                decoder_outputs_notes.view(-1, decoder_outputs_notes.size(-1)),
                target_notes.view(-1)
            )
            loss_durations = criterion( 
                decoder_outputs_durations.view(-1, decoder_outputs_durations.size(-1)),
                target_durations.view(-1)
            )
            loss_gaps = criterion(
                decoder_outputs_gaps.view(-1, decoder_outputs_gaps.size(-1)),
                target_gaps.view(-1)
            )
            loss = note_loss_weight * loss_notes + duration_loss_weight * loss_durations + gap_loss_weight * loss_gaps

            total_loss += loss.item()

            if not is_printed:
                print(f"Input: {data['input_ids'][0]}")
                print(f"Target: {data['labels']['notes'][0]}")
                print(f"Prediction: {decoder_outputs_notes.argmax(-1).cpu().numpy()[0]}")
                is_printed = True
            progress_bar.set_postfix({'Validation Loss': total_loss / len(dataloader)})

        return total_loss / len(dataloader)

def train(train_dataloader, val_dataloader, model, n_epochs, learning_rate=0.001, weight_decay=0.01,
               note_loss_weight=0.8, duration_loss_weight=0.2, gap_loss_weight=0.2, output_folder='/log'):
    train_losses, val_losses = [], []

    encoder_optimizer = AdamW(model.encoder.parameters(), lr=float(learning_rate), weight_decay=weight_decay)
    decoder_optimizer = AdamW(model.decoder.parameters(), lr=float(learning_rate), weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    csv_file = os.path.join(output_folder, 'training_log.csv')

    with open(csv_file, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file)

        csv_writer.writerow(['Epoch', 'Training Loss', 'Validation Loss'])
        for epoch in range(1, n_epochs + 1):
            model.train()
            loss = train_epoch(train_dataloader, model, encoder_optimizer, decoder_optimizer, criterion, epoch, note_loss_weight, duration_loss_weight, gap_loss_weight)
            train_losses.append(loss)
            print(f"Epoch {epoch}, training Loss: {loss}")

            model.eval()
            val_loss = evaluate_model(model, val_dataloader, criterion, note_loss_weight, duration_loss_weight, gap_loss_weight)
            val_losses.append(val_loss)
            print(f"Epoch {epoch}, Validation Loss: {val_loss}")

            csv_writer.writerow([epoch, loss, val_loss])
            csv_file.flush()
            
        return train_losses, val_losses

def main():
    with open(HYPS_FILE, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Create output file
    out_dir = config['out_dir']
    run_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = os.path.join(out_dir, run_dir)
    os.makedirs(output_folder, exist_ok=True)
    
    # Set seed
    set_seed(config['seed'])

    
    # Create language objects
    syllables = Lang('syllables')
    lines = open('./vocab/syllables.txt', 'r', encoding='utf-8').read().strip().split('\n')
    for syllable in lines:
        syllables.addWord(syllable)
    print(f"Number of syllables: {syllables.n_words}")


    model = CustomModelRNN(
        input_size=syllables.n_words,
        hidden_size=config['model']['hidden_size'], 
        SOS_token=0, 
        MAX_LENGTH=config['data']['max_sequence_length'], 
        dropout_p=config['model']['dropout'],
        device=device, 
        )

    batch_size = config['training']['batch_size']

        

    # Create dataset and collator
    train_dataset = SongsDataset(config['data']['data_dir'], split='train')
    valid_dataset = SongsDataset(config['data']['data_dir'], split='valid')
    test_dataset  = SongsDataset(config['data']['data_dir'], split='test')
    collator = SongsCollator(syllables_lang=syllables, output_eos=EOS_token, max_length=config['data']['max_sequence_length'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator, pin_memory=True, num_workers=0)
    val_loader   = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator, pin_memory=True, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator, pin_memory=True,  num_workers=0)

    train_loss, val_loss = train(
        train_loader, 
        val_loader, 
        model, 
        n_epochs=config['training']['epochs'], 
        learning_rate=config['training']['lr'], 
        weight_decay=config['training']['weight_decay'], 
        note_loss_weight=config['training']['note_loss_weight'],
        duration_loss_weight=config['training']['duration_loss_weight'],
        gap_loss_weight=config['training']['gap_loss_weight'],
        output_folder=output_folder
    )
    
    # Plot training and validation loss
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.grid()
    plt.savefig(os.path.join(output_folder, 'loss_plot.png'))

    # Save model
    torch.save(model.state_dict(), os.path.join(output_folder, 'model.pt'))

if __name__ == "__main__":
    main()